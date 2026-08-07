"""
SOPTX 原生子结构静态缩聚模块 (兼容 FEALPy 4.0)
=======================================================================
基于 SOPTX 原生的 QuadrangleMesh (四边形网格)、TensorFunctionSpace (张量有限元空间)、
IsotropicLinearElasticMaterial (各向同性线弹性材料) 与 LinearElasticIntegrator (线弹性积分器)，
用于通用有限元局部刚度阵组装与精确 Schur 补静态缩聚。

作者: Liang He (大连理工大学博士后) & Antigravity Assistant
日期: 2026-08-06
"""

from typing import Tuple, List, Union, Any, Optional
from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm
from soptx.fem.integrators.linear_elastic_integrator import LinearElasticIntegrator
from soptx.materials import IsotropicLinearElasticMaterial


class SubstructureMesh:
    """
    基于 SOPTX 原生 FEALPy 4.0 QuadrangleMesh (2D) / HexahedronMesh (3D) 管理的子结构区域。
    """
    def __init__(
        self,
        sub_id: int,
        *args: Any,
        E_base: float = 1.0,
        nu: float = 0.3
    ) -> None:
        self.sub_id: int = sub_id
        self.sub_id: int = sub_id

        # 灵活解析 2D/3D 参数 (支持元组输入及传统 2D/3D 位置参数)
        rest = args
        if isinstance(rest[0], (tuple, list)) and isinstance(rest[0][0], (tuple, list)):
            self.box_span: Tuple[Tuple[float, float], ...] = tuple(rest[0])
            self.n_fine: Tuple[int, ...] = tuple(rest[1])
            self.E_base = rest[2] if len(rest) > 2 else E_base
            self.nu = rest[3] if len(rest) > 3 else nu
        elif len(rest) >= 6 and isinstance(rest[0], (tuple, list)) and isinstance(rest[1], (tuple, list)) and isinstance(rest[2], (tuple, list)):
            self.box_span = (rest[0], rest[1], rest[2])
            self.n_fine = (rest[3], rest[4], rest[5])
            self.E_base = rest[6] if len(rest) > 6 else E_base
            self.nu = rest[7] if len(rest) > 7 else nu
        elif isinstance(rest[0], (tuple, list)) and isinstance(rest[1], (tuple, list)):
            self.box_span = (rest[0], rest[1])
            self.n_fine = (rest[2], rest[3])
            self.E_base = rest[4] if len(rest) > 4 else E_base
            self.nu = rest[5] if len(rest) > 5 else nu
        else:
            raise ValueError("无法解析的子结构网格输入参数")

        self.dim: int = len(self.box_span)

        # 属性别名
        self.x_span = self.box_span[0]
        self.y_span = self.box_span[1]
        self.n_fine_x = self.n_fine[0]
        self.n_fine_y = self.n_fine[1]
        if self.dim == 3:
            self.z_span = self.box_span[2]
            self.n_fine_z = self.n_fine[2]

        box = [coord for span in self.box_span for coord in span]
        if self.dim == 2:
            self.mesh = QuadrangleMesh.from_box(box=box, nx=self.n_fine[0], ny=self.n_fine[1])
            self.material = IsotropicLinearElasticMaterial(youngs_modulus=self.E_base, poisson_ratio=self.nu, hypothesis='plane_stress')
        else:
            self.mesh = HexahedronMesh.from_box(box=box, nx=self.n_fine[0], ny=self.n_fine[1], nz=self.n_fine[2])
            self.material = IsotropicLinearElasticMaterial(youngs_modulus=self.E_base, poisson_ratio=self.nu)

        self.sspace: LagrangeFESpace = LagrangeFESpace(self.mesh, p=1, ctype='C')
        self.space: TensorFunctionSpace = TensorFunctionSpace(self.sspace, shape=(-1, self.dim))
        self.integrator: LinearElasticIntegrator = LinearElasticIntegrator(material=self.material)

        self.n_nodes_x: int = self.n_fine[0] + 1
        self.n_nodes_y: int = self.n_fine[1] + 1
        if self.dim == 3:
            self.n_nodes_z: int = self.n_fine[2] + 1

        self.n_total_nodes: int = self.mesh.number_of_nodes()
        self.n_total_dofs: int = self.space.number_of_global_dofs()

        # 通用维度节点分类: 内部节点 (i) 与 边界/界面节点 (b)
        node_coords = self.mesh.entity('node')
        eps = 1e-7
        is_boundary = bm.zeros(node_coords.shape[0], dtype=bm.bool)
        for d in range(self.dim):
            is_boundary |= (bm.abs(node_coords[:, d] - self.box_span[d][0]) < eps)
            is_boundary |= (bm.abs(node_coords[:, d] - self.box_span[d][1]) < eps)

        self.boundary_nodes: Any = bm.nonzero(is_boundary)[0]
        self.internal_nodes: Any = bm.nonzero(~is_boundary)[0]

        self.i_dofs: Any = bm.sort(bm.concat([self.dim * self.internal_nodes + k for k in range(self.dim)]))
        self.b_dofs: Any = bm.sort(bm.concat([self.dim * self.boundary_nodes + k for k in range(self.dim)]))

        self.n_i: int = len(self.i_dofs)
        self.n_b: int = len(self.b_dofs)

    def assemble_local_stiffness(self, density_field: Any) -> Any:
        """
        使用 SOPTX 原生 BilinearForm 和 SIMP 变密度惩罚模型组装局部刚度阵 K_local。
        """
        simp_coef = bm.asarray(density_field.flatten()**3.0, dtype=bm.float64)
        self.integrator.coef = simp_coef

        bform = BilinearForm(self.space)
        bform.add_integrator(self.integrator)

        K_tensor = bform.assembly()
        if hasattr(K_tensor, 'toarray'):
            K_local = K_tensor.toarray()
        else:
            K_local = bm.asarray(K_tensor)

        return bm.asarray(K_local, dtype=bm.float64)


class FEAStaticCondensation:
    """
    有限元精确 Schur 补静态缩聚模块 (SOPTX 核心有限元基准解)。
    """
    def __init__(self, i_dofs: Any, b_dofs: Any) -> None:
        self.i_dofs: Any = bm.asarray(i_dofs, dtype=bm.int64)
        self.b_dofs: Any = bm.asarray(b_dofs, dtype=bm.int64)
        self.n_i: int = len(self.i_dofs)
        self.n_b: int = len(self.b_dofs)

        self.K_s: Optional[Any] = None
        self.N: Optional[Any] = None

    def condense(self, K_local: Any) -> Tuple[Any, Any]:
        """
        计算精确的 Schur 补缩聚刚度阵 K_s 及形函数矩阵 N：
            K_ii u_i + K_ib u_b = 0  =>  u_i = - inv(K_ii) K_ib u_b = N u_b
            K_s = K_bb - K_bi inv(K_ii) K_ib
        """
        K_ii = K_local[self.i_dofs[:, None], self.i_dofs]
        K_ib = K_local[self.i_dofs[:, None], self.b_dofs]
        K_bi = K_local[self.b_dofs[:, None], self.i_dofs]
        K_bb = K_local[self.b_dofs[:, None], self.b_dofs]

        # 求解 K_ii^{-1} K_ib
        invK_ii_K_ib = bm.linalg.solve(K_ii, K_ib)

        self.N = -invK_ii_K_ib
        self.K_s = K_bb - K_bi @ invK_ii_K_ib

        return self.K_s, self.N

    def recover(self, u_b: Any) -> Any:
        """根据界面位移 u_b 恢复内部位移 u_i = N * u_b。"""
        if self.N is None:
            raise RuntimeError("必须先调用 condense() 方法才能进行 recover() 计算")
        return self.N @ u_b

