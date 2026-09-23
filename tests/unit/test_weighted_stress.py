# -*- coding: utf-8 -*-
"""逐点算子 D 的对称性.

``weighted_stress`` 在每个积分点上作用 w |J| rho S^T D S, 本构矩阵 D 对称时它必须是
对称算子, 即对任意 g 与 h

    <D g, h> = <g, D h>

与 ``test_gradients`` 里 B 与 B^T 的伴随测试配对: 两者都成立, PA 算子 B^T D B 才对称,
CG 才可用. 与那边一样只用合成数组, 不经网格与空间; 覆盖 2D / 3D 与末尾带批量维 NB
的多列情形.
"""

import unittest

from fealpy.backend import backend_manager as bm

from soptx.fem.kernels import strain_map, weighted_stress


class TestWeightedStressSymmetry(unittest.TestCase):
    """D 在欧氏内积下自伴"""

    RTOL = 1.0e-12

    # (NC, NQ, GD)
    SHAPES = ((5, 3, 2),
            (4, 8, 3))

    @classmethod
    def setUpClass(cls) -> None:
        bm.set_backend('numpy')

    @staticmethod
    def _deterministic(shape, frequency: float):
        """由固定三角函数生成的数组, 不用随机数, 失败时可以直接复现"""
        size = 1
        for n in shape:
            size *= n
        t = bm.arange(size, dtype=bm.float64)

        return bm.reshape(bm.sin(frequency * t) + 0.3 * bm.cos(2.1 * frequency * t),
                        shape)

    def _elastic_matrix(self, geo_dim: int):
        """对称正定的 (NS, NS) 本构矩阵: 合成矩阵 A 取 A A^T 再加对角"""
        n_strain = geo_dim * (geo_dim + 1) // 2
        a = self._deterministic((n_strain, n_strain), 0.9)

        return a @ bm.swapaxes(a, 0, 1) + n_strain * bm.eye(n_strain, dtype=bm.float64)

    def _check(self, n_cells, n_quad, geo_dim, batch=()) -> None:
        elastic_matrix = self._elastic_matrix(geo_dim)
        smap = strain_map(geo_dim, dtype=bm.float64)
        weighted_coef = 0.5 + bm.abs(self._deterministic((n_cells, n_quad), 1.1))
        g = self._deterministic((n_cells, n_quad, geo_dim, geo_dim) + batch, 1.7)
        h = self._deterministic((n_cells, n_quad, geo_dim, geo_dim) + batch, 2.9)

        kwargs = dict(weighted_coef=weighted_coef, elastic_matrix=elastic_matrix,
                    strain_map=smap)
        d_g = weighted_stress(g, **kwargs)
        d_h = weighted_stress(h, **kwargs)

        self.assertEqual(tuple(d_g.shape), tuple(g.shape))

        # 除批量维外全部求和, 多列时逐列比较
        lhs = bm.sum(bm.reshape(d_g * h, (-1, ) + batch), axis=0)
        rhs = bm.sum(bm.reshape(g * d_h, (-1, ) + batch), axis=0)

        scale = float(bm.max(bm.abs(lhs)))
        difference = float(bm.max(bm.abs(lhs - rhs))) / scale
        self.assertLessEqual(difference, self.RTOL)

    def test_single_column(self) -> None:
        """单列: <D g, h> = <g, D h>"""
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                self._check(*shape)

    def test_multi_column(self) -> None:
        """末尾带批量维 NB 时, 对称性逐列成立"""
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                self._check(*shape, batch=(3, ))


if __name__ == '__main__':
    unittest.main()
