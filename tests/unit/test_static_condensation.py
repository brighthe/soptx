import unittest

import numpy as np

from fealpy.backend import backend_manager as bm

from soptx.fem.substructure import FEAStaticCondensation


def _make_spd_batch(shape_prefix: tuple[int, ...], n_dof: int, seed: int) -> np.ndarray:
    """构造逐位对称的随机正定刚度矩阵批次.

    参数:
        shape_prefix: 前导批量维, 空元组表示单个矩阵.
        n_dof: 局部自由度总数.
        seed: 随机数种子.

    返回:
        K: 形状 ``(*shape_prefix, n_dof, n_dof)`` 的对称正定矩阵.

    说明:
        以 ``0.5 * (M + M^T)`` 显式对称化, 保证测试输入本身不带舍入级非对称,
        使被测对象的对称性表现可归因于缩聚实现而非输入.
    """
    rng = np.random.default_rng(seed=seed)
    samples = rng.standard_normal((*shape_prefix, n_dof, n_dof))
    gram = np.swapaxes(samples, -1, -2) @ samples
    symmetric = 0.5 * (gram + np.swapaxes(gram, -1, -2))
    return symmetric + 2.0 * np.eye(n_dof)


class TestStaticCondensation(unittest.TestCase):
    """验证精确 Schur 补缩聚在标量与任意前导批量维下的一致行为."""

    @classmethod
    def setUpClass(cls) -> None:
        bm.set_backend("numpy")
        cls.i_dofs = bm.asarray([0, 2, 4], dtype=bm.int64)
        cls.b_dofs = bm.asarray([1, 3, 5], dtype=bm.int64)
        cls.n_dof = 6
        cls.K_local_batch = _make_spd_batch((4,), cls.n_dof, seed=20260814)

    def _condensor(self) -> FEAStaticCondensation:
        return FEAStaticCondensation(self.i_dofs, self.b_dofs)

    def test_batch_matches_scalar(self) -> None:
        """批量缩聚与逐个标量缩聚在机器精度内一致."""
        K_s_batch, N_batch = self._condensor().condense(self.K_local_batch)

        self.assertEqual(K_s_batch.shape, (4, 3, 3))
        self.assertEqual(N_batch.shape, (4, 3, 3))

        for batch_index, K_local in enumerate(self.K_local_batch):
            K_s_scalar, N_scalar = self._condensor().condense(K_local)
            np.testing.assert_allclose(
                bm.to_numpy(K_s_batch[batch_index]),
                bm.to_numpy(K_s_scalar),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                bm.to_numpy(N_batch[batch_index]),
                bm.to_numpy(N_scalar),
                rtol=1.0e-12,
                atol=1.0e-12,
            )

    def test_scalar_input_has_empty_leading_dim(self) -> None:
        """二维输入被视为前导维为空的特例, 输出不带批量维."""
        K_local = _make_spd_batch((), self.n_dof, seed=7)
        K_s, N = self._condensor().condense(K_local)

        self.assertEqual(K_s.shape, (3, 3))
        self.assertEqual(N.shape, (3, 3))

    def test_supports_multiple_leading_dims(self) -> None:
        """前导维不限于单个批量轴, 任意多维前缀均被保留."""
        K_local = _make_spd_batch((2, 3), self.n_dof, seed=11)
        K_s, N = self._condensor().condense(K_local)

        self.assertEqual(K_s.shape, (2, 3, 3, 3))
        self.assertEqual(N.shape, (2, 3, 3, 3))

    def test_recovery_satisfies_internal_equilibrium(self) -> None:
        """``N`` 与 ``K_s`` 满足静力缩聚的定义式, 与实现路径无关."""
        condensor = self._condensor()
        condensor.condense(self.K_local_batch)

        rng = np.random.default_rng(seed=2026)
        u_b = rng.standard_normal((4, 3))
        u_i = bm.to_numpy(condensor.recover(u_b))
        self.assertEqual(u_i.shape, (4, 3))

        i_dofs = bm.to_numpy(self.i_dofs)
        b_dofs = bm.to_numpy(self.b_dofs)
        K_ii = self.K_local_batch[:, i_dofs[:, None], i_dofs]
        K_ib = self.K_local_batch[:, i_dofs[:, None], b_dofs]
        K_bi = self.K_local_batch[:, b_dofs[:, None], i_dofs]
        K_bb = self.K_local_batch[:, b_dofs[:, None], b_dofs]

        # 内部自由度无外载: K_ii u_i + K_ib u_b = 0.
        internal_residual = (
            np.einsum('bij, bj -> bi', K_ii, u_i)
            + np.einsum('bij, bj -> bi', K_ib, u_b)
        )
        np.testing.assert_allclose(
            internal_residual, np.zeros((4, 3)), rtol=0.0, atol=1.0e-10
        )

        # 接口反力: K_bi u_i + K_bb u_b = K_s u_b.
        interface_force = (
            np.einsum('bij, bj -> bi', K_bi, u_i)
            + np.einsum('bij, bj -> bi', K_bb, u_b)
        )
        np.testing.assert_allclose(
            interface_force,
            np.einsum('bij, bj -> bi', bm.to_numpy(condensor.K_s), u_b),
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_recover_broadcasts_single_interface_vector(self) -> None:
        """单个接口位移向量可按广播规则作用于批量 ``N``."""
        condensor = self._condensor()
        condensor.condense(self.K_local_batch)

        rng = np.random.default_rng(seed=99)
        u_b = rng.standard_normal(3)
        u_i = bm.to_numpy(condensor.recover(u_b))

        self.assertEqual(u_i.shape, (4, 3))
        for batch_index in range(4):
            np.testing.assert_allclose(
                u_i[batch_index],
                bm.to_numpy(condensor.N)[batch_index] @ u_b,
                rtol=1.0e-12,
                atol=1.0e-12,
            )

    def test_recover_before_condense_raises(self) -> None:
        """未调用 ``condense`` 时恢复内部位移应失败而非返回错误结果."""
        with self.assertRaises(RuntimeError):
            self._condensor().recover(np.zeros(3))

    def test_preserves_input_dtype(self) -> None:
        """缩聚不强制 dtype, 输出精度与输入一致, 以免污染 float32 代理路径."""
        K_local = _make_spd_batch((4,), self.n_dof, seed=5).astype(np.float32)
        K_s, N = self._condensor().condense(K_local)

        self.assertEqual(bm.to_numpy(K_s).dtype, np.float32)
        self.assertEqual(bm.to_numpy(N).dtype, np.float32)

    def test_condensed_stiffness_is_symmetric(self) -> None:
        """对称输入下缩聚刚度的非对称量应停留在求解与矩阵乘法的舍入水平."""
        K_s, _ = self._condensor().condense(self.K_local_batch)
        K_s_np = bm.to_numpy(K_s)

        asymmetry = np.max(np.abs(K_s_np - np.swapaxes(K_s_np, -1, -2)))
        self.assertLess(asymmetry, 1.0e-12 * np.max(np.abs(K_s_np)))

    def test_rejects_shape_mismatch(self) -> None:
        """局部刚度矩阵形状与自由度划分不一致时立即失败."""
        condensor = self._condensor()
        with self.assertRaises(ValueError):
            condensor.condense(np.ones(6))
        with self.assertRaises(ValueError):
            condensor.condense(np.ones((2, 5, 6)))
        with self.assertRaises(ValueError):
            condensor.condense(np.eye(8))

    def test_rejects_invalid_dof_partition_at_construction(self) -> None:
        """自由度划分的校验在构造时完成, 不进入缩聚热路径."""
        with self.assertRaises(ValueError):
            FEAStaticCondensation([-1, 2, 4], [1, 3, 5])
        with self.assertRaises(ValueError):
            FEAStaticCondensation([0, 2, 4], [1, 2, 5])
        with self.assertRaises(ValueError):
            FEAStaticCondensation([0, 2], [1, 3, 5])
        with self.assertRaises(ValueError):
            FEAStaticCondensation([0, 0, 4], [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
