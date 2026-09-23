# -*- coding: utf-8 -*-
"""FA / EA / PA 三个装配层级的等价性.

三者是同一个离散算子 K = sum_e R_e^T K_e R_e 的三种常驻形式: FA 存全局稀疏矩阵,
EA 存单元矩阵, PA 只存积分点上的几何与材料数据. 因此在同一个空间同一个积分子上,
三者的 ``@`` 与 ``diagonal()`` 必须一致到舍入误差 (求和次序不同, 不逐位相同).

覆盖 tri (2D) 与 hex (3D) x p = 1, 2 x coef 取 None / (NC, ) / (NC, NQ). 只测层级
本身, 不经过 analyzer, 因此边界条件与求解器都不参与.
"""

import unittest

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import HexahedronMesh, TriangleMesh

from soptx.fem.integrators import LinearElasticIntegrator
from soptx.fem.kernels import physical_basis_gradients
from soptx.fem.levels import available_levels, create_level
from soptx.materials import IsotropicLinearElasticMaterial


class TestAssemblyLevelEquivalence(unittest.TestCase):
    """以 EA 为基准, 校验 FA 与 PA 给出同一个离散算子"""

    RTOL = 1.0e-11

    # (网格名, 几何维数, 每方向单元数); 规模只要够暴露下标错位即可
    MESHES = (('tri', 2, 3), ('hex', 3, 2))
    DEGREES = (1, 2)
    HYPOTHESES = {2: 'plane_strain', 3: '3D'}

    @classmethod
    def setUpClass(cls) -> None:
        bm.set_backend('numpy')

    # ------------------------------------------------------------------ 夹具
    @staticmethod
    def _build_mesh(kind: str, n: int):
        if kind == 'tri':
            return TriangleMesh.from_box([0, 1, 0, 1], nx=n, ny=n)
        if kind == 'hex':
            return HexahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)
        raise ValueError(f"未知网格类型 {kind!r}")

    def _make_fixture(self, kind: str, dimension: int, n: int, p: int,
                    shape=None) -> dict:
        """造出一套空间, 材料与确定性测试数据.

        测试向量与 coef 都用固定的三角函数生成而不用随机数, 失败时可以直接复现.
        ``shape`` 决定张量空间的自由度排序, 默认 (-1, GD); 传 (GD, -1) 得到另一种.
        """
        mesh = self._build_mesh(kind, n)
        scalar_space = LagrangeFESpace(mesh, p=p, ctype='C')
        space = TensorFunctionSpace(scalar_space,
                                    shape=(-1, dimension) if shape is None else shape)
        material = IsotropicLinearElasticMaterial(
                        hypothesis=self.HYPOTHESES[dimension],
                        lame_lambda=1.0,
                        shear_modulus=0.75,
                        device=bm.get_device(mesh),
                    )

        # PA 的 build 在 integrator.q 为 None 时取 p + 3, 这里跟着取同一个
        _, ws = mesh.quadrature_formula(p + 3).get_quadrature_points_and_weights()
        n_quad = int(ws.shape[0])
        n_cells = mesh.number_of_cells()
        gdof = int(space.number_of_global_dofs())

        t = bm.linspace(-1.0, 1.0, gdof, **bm.context(ws))

        return {
            'mesh': mesh,
            'scalar_space': scalar_space,
            'space': space,
            'material': material,
            'n_quad': n_quad,
            'n_cells': n_cells,
            'x': bm.sin(13.0 * t) + 0.3 * bm.cos(7.0 * t),
            'x2': bm.cos(5.0 * t) - 0.7 * bm.sin(3.0 * t),
            'coef_cell': 0.4 + 0.5 * bm.abs(
                            bm.sin(bm.linspace(0.0, 9.0, n_cells, **bm.context(ws)))),
            'coef_quad': bm.reshape(
                            0.3 + 0.6 * bm.abs(bm.cos(bm.linspace(
                                0.0, 11.0, n_cells * n_quad, **bm.context(ws)))),
                            (n_cells, n_quad)),
        }

    @staticmethod
    def _make_levels(fixture: dict, coef) -> dict:
        """现造积分子再建三个层级.

        单元矩阵按 ``const`` 的积分子实例缓存, 复用同一个积分子换 coef 会吃到旧值,
        所以每种 coef 都新建一个.
        """
        integrator = LinearElasticIntegrator(material=fixture['material'], coef=coef)

        return {name: create_level(name,
                                space=fixture['space'],
                                integrator=integrator,
                                pattern=None)
                for name in ('fa', 'ea', 'pa')}

    def _relative_difference(self, actual, expected) -> float:
        scale = float(bm.max(bm.abs(expected)))
        if scale == 0.0:
            scale = 1.0

        return float(bm.max(bm.abs(actual - expected))) / scale

    def _coefficients(self, fixture: dict):
        return (('coef=None', None),
                ('coef=(NC,)', fixture['coef_cell']),
                ('coef=(NC,NQ)', fixture['coef_quad']))

    def _iterate_fixtures(self):
        for kind, dimension, n in self.MESHES:
            for p in self.DEGREES:
                yield kind, p, self._make_fixture(kind, dimension, n, p)

    # ------------------------------------------------------------------ 用例
    def test_pa_is_registered(self) -> None:
        """三个层级都在注册表里, analyzer 的 operator_level 校验依赖它"""
        for name in ('fa', 'ea', 'pa'):
            self.assertIn(name, available_levels())

    def test_basis_gradients_match_space(self) -> None:
        """PA 由参考梯度加 J^{-1} 合成的物理梯度, 必须等于空间直接给的.

        这是 PA 存储方案的前提: 常驻的是 (NQ, ldof, TD) 的参考梯度加每单元 J^{-1},
        而不是每单元的物理梯度 (NC, NQ, ldof, GD).
        """
        for kind, p, fixture in self._iterate_fixtures():
            with self.subTest(mesh=kind, p=p):
                bcs, _ = fixture['mesh'].quadrature_formula(
                                p + 3).get_quadrature_points_and_weights()
                levels = self._make_levels(fixture, None)

                pa = levels['pa']
                difference = self._relative_difference(
                                physical_basis_gradients(
                                    reference_grad=pa.reference_basis.grad,
                                    jacobi_inverse=pa.geometric_factors.jacobi_inverse),
                                fixture['scalar_space'].grad_basis(bcs, variable='x'))

                self.assertLessEqual(difference, self.RTOL)

    def test_matvec_matches_element_assembly(self) -> None:
        """FA 与 PA 对同一向量的作用必须等于 EA"""
        for kind, p, fixture in self._iterate_fixtures():
            for label, coef in self._coefficients(fixture):
                levels = self._make_levels(fixture, coef)
                x = fixture['x']
                expected = levels['ea'] @ x

                for name in ('fa', 'pa'):
                    with self.subTest(mesh=kind, p=p, coef=label, level=name):
                        difference = self._relative_difference(levels[name] @ x,
                                                            expected)
                        self.assertLessEqual(difference, self.RTOL)

    def test_multi_column_matches_single_column(self) -> None:
        """多列右端项按 (gdof, B) 布局, 每列必须等于单列分别作用的结果.

        算子的规范布局是自由度维在前: ``CG`` 在 ``batch_first=True`` 时先转成
        (gdof, B) 再进迭代, ``ConstrainedOperator`` 也按行取 Dirichlet 自由度. 三个
        层级里只有 FA 天然满足, EA 与 PA 的 gather / scatter-add 要显式对齐, 因此这
        条同时是对 ``ElementRestriction`` 批量维约定的回归.
        """
        for kind, p, fixture in self._iterate_fixtures():
            for label, coef in self._coefficients(fixture):
                levels = self._make_levels(fixture, coef)
                columns = (fixture['x'], fixture['x2'])
                batch = bm.stack(columns, axis=1)

                for name in ('fa', 'ea', 'pa'):
                    with self.subTest(mesh=kind, p=p, coef=label, level=name):
                        actual = levels[name] @ batch

                        self.assertEqual(tuple(actual.shape),
                                        (batch.shape[0], len(columns)))

                        for j, column in enumerate(columns):
                            difference = self._relative_difference(
                                            actual[:, j], levels[name] @ column)
                            self.assertLessEqual(difference, self.RTOL)

    def test_pa_matches_ea_for_both_dof_orderings(self) -> None:
        """两种自由度排序下, PA 的作用与对角都必须等于 EA.

        PA 的 E 向量是 (NC, ldof, GD) 的分量布局, 由 ``ElementRestriction`` 按
        ``dof_priority`` 重排 ``cell2dof`` 得到; EA 仍用扁平布局直接乘 K_e. 重排方向一
        旦弄反, 只有其中一种排序会暴露, 因此两种都要测.
        """
        for kind, dimension, n in self.MESHES:
            for p in self.DEGREES:
                for shape in ((-1, dimension), (dimension, -1)):
                    fixture = self._make_fixture(kind, dimension, n, p, shape=shape)
                    levels = self._make_levels(fixture, fixture['coef_cell'])
                    priority = fixture['space'].dof_priority

                    with self.subTest(mesh=kind, p=p, dof_priority=priority,
                                    quantity='matvec'):
                        difference = self._relative_difference(
                                        levels['pa'] @ fixture['x'],
                                        levels['ea'] @ fixture['x'])
                        self.assertLessEqual(difference, self.RTOL)

                    with self.subTest(mesh=kind, p=p, dof_priority=priority,
                                    quantity='diagonal'):
                        difference = self._relative_difference(
                                        levels['pa'].diagonal(),
                                        levels['ea'].diagonal())
                        self.assertLessEqual(difference, self.RTOL)

    def test_diagonal_matches_element_assembly(self) -> None:
        """三个层级取出的对角必须一致.

        PA 的对角走闭式而不是对单位向量作用, 是 Jacobi 预条件子在 PA 下可用的前提.
        """
        for kind, p, fixture in self._iterate_fixtures():
            for label, coef in self._coefficients(fixture):
                levels = self._make_levels(fixture, coef)
                expected = levels['ea'].diagonal()

                for name in ('fa', 'pa'):
                    with self.subTest(mesh=kind, p=p, coef=label, level=name):
                        difference = self._relative_difference(
                                        levels[name].diagonal(), expected)
                        self.assertLessEqual(difference, self.RTOL)

    def test_update_matches_rebuilt_level(self) -> None:
        """PA 就地更新 coef 后, 必须等于直接用该 coef 建的 EA.

        PA 的 update 只重算每积分点一个标量, 不碰几何因子也不重新积分; 这条是拓扑
        优化循环里复用 PA 实例的前提.
        """
        for kind, p, fixture in self._iterate_fixtures():
            with self.subTest(mesh=kind, p=p):
                coef = fixture['coef_cell']
                updated = self._make_levels(fixture, None)['pa']
                updated.update(coef)
                reference = self._make_levels(fixture, coef)['ea']
                x = fixture['x']

                difference = self._relative_difference(updated @ x, reference @ x)

                self.assertLessEqual(difference, self.RTOL)

    def test_partial_assembly_stores_less_at_second_order(self) -> None:
        """p = 2 起 PA 的常驻量必须少于 EA.

        PA 每单元常驻 O(NQ * GD^2), EA 每单元常驻 O((GD * ldof)^2); 存储优势正是
        选 PA 的理由, 交叉点在 p = 2. 若哪天改成常驻每单元物理梯度, 这条会先失败.
        """
        for kind, dimension, n in self.MESHES:
            with self.subTest(mesh=kind):
                fixture = self._make_fixture(kind, dimension, n, 2)
                levels = self._make_levels(fixture, None)

                self.assertLess(levels['pa'].persistent_bytes(),
                                levels['ea'].persistent_bytes())


if __name__ == '__main__':
    unittest.main()
