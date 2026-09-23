# -*- coding: utf-8 -*-
"""基函数算子 B 与 B^T 的伴随一致性.

``physical_gradient`` (B) 与 ``physical_gradient_transpose`` (B^T) 必须严格互为转置,
否则 PA 算子不对称, CG 失效. 对任意 u_E 与 s_Q 检查

    <B u, s> = <u, B^T s>

两个 kernel 只吃数组, 因此这里不经网格与空间, 直接用合成数组; 伴随关系对任意数组都
成立, 不依赖它们来自真实的几何. 覆盖 TD = GD 的 2D / 3D, TD < GD 的曲面情形, 以及
末尾带批量维 NB 的多列情形.
"""

import unittest

from fealpy.backend import backend_manager as bm

from soptx.fem.kernels import physical_gradient, physical_gradient_transpose


class TestGradientAdjointness(unittest.TestCase):
    """B 与 B^T 在欧氏内积下互为伴随"""

    RTOL = 1.0e-12

    # (NC, NQ, ldof, TD, GD)
    SHAPES = ((5, 3, 6, 2, 2),
            (4, 8, 27, 3, 3),
            (3, 4, 6, 2, 3))

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

    def _check(self, n_cells, n_quad, local_dofs, top_dim, geo_dim, batch=()) -> None:
        reference_grad = self._deterministic((n_quad, local_dofs, top_dim), 0.7)
        jacobi_inverse = self._deterministic((n_cells, n_quad, top_dim, geo_dim), 1.3)
        u_E = self._deterministic((n_cells, local_dofs, geo_dim) + batch, 1.9)
        s_Q = self._deterministic((n_cells, n_quad, geo_dim, geo_dim) + batch, 2.3)

        grad_u = physical_gradient(u_E, reference_grad=reference_grad,
                                jacobi_inverse=jacobi_inverse)
        y_E = physical_gradient_transpose(s_Q, reference_grad=reference_grad,
                                        jacobi_inverse=jacobi_inverse)

        self.assertEqual(tuple(grad_u.shape), tuple(s_Q.shape))
        self.assertEqual(tuple(y_E.shape), tuple(u_E.shape))

        # 除批量维外全部求和, 多列时逐列比较
        lhs = bm.sum(bm.reshape(grad_u * s_Q, (-1, ) + batch), axis=0)
        rhs = bm.sum(bm.reshape(u_E * y_E, (-1, ) + batch), axis=0)

        scale = float(bm.max(bm.abs(lhs)))
        difference = float(bm.max(bm.abs(lhs - rhs))) / scale
        self.assertLessEqual(difference, self.RTOL)

    def test_single_column(self) -> None:
        """单列: <B u, s> = <u, B^T s>"""
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                self._check(*shape)

    def test_multi_column(self) -> None:
        """末尾带批量维 NB 时, 伴随关系逐列成立"""
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                self._check(*shape, batch=(3, ))


if __name__ == '__main__':
    unittest.main()
