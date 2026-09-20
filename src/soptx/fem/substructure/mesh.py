"""子结构网格与局部刚度装配.

同一模型中的全部子结构在几何上互为平移, 因此离散结构 (网格拓扑, 函数空间,
内部/接口自由度划分, 单位密度单元刚度阵) 与位置无关, 由 ``SubstructurePrototype``
持有并只构造一次. ``SubstructureMesh`` 是原型加上一个全局位置的轻量封装.

局部刚度装配接口接受任意可变前导维 ``...``: 单个子结构是前导维为空的特例,
批量子结构对应前导维 ``B``, 与 ``condensation`` 模块的形状约定一致.
"""

from typing import Tuple, Any, Optional, Sequence, List, Dict, Iterator

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.fem.integrators.linear_elastic_integrator import LinearElasticIntegrator
from soptx.materials import IsotropicLinearElasticMaterial


class SubstructurePrototype:
    """
    参考子结构: 同构子结构共享的离散结构与单位密度单元刚度阵.

    子结构刚度矩阵在平移下不变, 只取决于子结构的物理尺寸 ``cell_size``, 细网格
    划分 ``n_fine`` 和材料参数. 因此一个模型中的全部同构子结构共用一个原型,
    网格, 函数空间与单元刚度阵只需构造一次.

    SIMP 插值对单元刚度是线性的, 单元刚度阵可写成 ``coef(rho_e) * KE_unit[e]``,
    其中 ``KE_unit`` 是单位密度下的单元刚度阵. 批量装配据此避免逐个子结构重复
    调用有限元装配.

    属性:
        i_dofs: 内部自由度的局部编号, 形状 ``(n_i,)``.
        b_dofs: 接口自由度的局部编号, 形状 ``(n_b,)``.
        KE_unit: 单位密度单元刚度阵, 形状 ``(NC, n_edof, n_edof)``.
        cell2dof: 单元到局部自由度的映射, 形状 ``(NC, n_edof)``.
    """

    # 接口自由度上刚体模态基, 其正交补, 以及刚体运动下内部自由度取值的惰性缓存.
    _rigid_basis: Optional[Any] = None
    _deformation_basis: Optional[Any] = None
    _rigid_interior: Optional[Any] = None

    # 迹空间表示刚体模态的相对残量上限. 容许迹空间下该残量是舍入量级, 阈值只用于
    # 拦截不容许的迹空间, 不参与任何精度判定.
    _TRACE_RIGID_TOL: float = 1.0e-10

    def __init__(
        self,
        cell_size: Sequence[float],
        n_fine: Sequence[int],
        E_base: float = 1.0,
        nu: float = 0.3,
        *,
        degree: int = 1,
        p: Optional[int] = None,
        integration_order: Optional[int] = None,
        penal: float = 3.0,
        rho_min: float = 0.0,
    ) -> None:
        """
        构造参考子结构.

        参数:
            cell_size: 单个子结构在各方向的物理尺寸, 长度为 2 或 3.
            n_fine: 单个子结构在各方向的细单元数, 长度与 ``cell_size`` 相同.
            E_base: 实体材料的杨氏模量.
            nu: 泊松比.
            degree: 有限元位移空间的多项式插值次数, 缺省为 ``1``.
            p: ``degree`` 的别名, 显式给出时优先于 ``degree``.
            integration_order: 单元数值积分阶数; ``None`` 使用积分器缺省值.
            penal: SIMP 惩罚指数.
            rho_min: SIMP 刚度下限比值, 取 ``0.0`` 时插值为 ``rho**penal``,
                取正值时为 ``rho_min + (1 - rho_min) * rho**penal``.

        异常:
            ValueError: 当维数不是 2 或 3, ``cell_size`` 与 ``n_fine`` 长度不一致,
                ``degree`` 非正, 或存在非正的尺寸与单元数时抛出.
            RuntimeError: 当张量函数空间的自由度排布不是节点优先, 或细网格单元
                无法与结构化网格下标一一对应时抛出.
        """
        if len(cell_size) != len(n_fine):
            raise ValueError(
                f"cell_size 与 n_fine 的长度必须一致; "
                f"当前分别为 {len(cell_size)} 与 {len(n_fine)}."
            )
        self.dim: int = len(cell_size)
        if self.dim not in (2, 3):
            raise ValueError(f"仅支持 2D 与 3D 子结构; 当前维数为 {self.dim}.")
        if any(float(s) <= 0.0 for s in cell_size):
            raise ValueError(f"cell_size 的各分量必须为正; 当前为 {tuple(cell_size)}.")
        if any(int(n) <= 0 for n in n_fine):
            raise ValueError(f"n_fine 的各分量必须为正整数; 当前为 {tuple(n_fine)}.")

        self.degree: int = int(p if p is not None else degree)
        if self.degree <= 0:
            raise ValueError(f"degree 必须为正整数; 当前为 {self.degree}.")
        self.p: int = self.degree
        self.integration_order = (
            None if integration_order is None else int(integration_order)
        )
        if self.integration_order is not None and self.integration_order <= 0:
            raise ValueError(
                "integration_order 必须为正整数或 None; "
                f"当前为 {self.integration_order}."
            )

        self.cell_size: Tuple[float, ...] = tuple(float(s) for s in cell_size)
        self.n_fine: Tuple[int, ...] = tuple(int(n) for n in n_fine)
        self.E_base: float = float(E_base)
        self.nu: float = float(nu)
        self.penal: float = float(penal)
        self.rho_min: float = float(rho_min)

        # 原型网格取在原点处; 位置不影响刚度矩阵, 由 SubstructureMesh 单独记录.
        box = [c for s in self.cell_size for c in (0.0, s)]
        if self.dim == 2:
            self.mesh: Any = QuadrangleMesh.from_box(
                box=box, nx=self.n_fine[0], ny=self.n_fine[1]
            )
            self.material = IsotropicLinearElasticMaterial(
                youngs_modulus=self.E_base,
                poisson_ratio=self.nu,
                hypothesis='plane_stress',
            )
        else:
            self.mesh = HexahedronMesh.from_box(
                box=box, nx=self.n_fine[0], ny=self.n_fine[1], nz=self.n_fine[2]
            )
            self.material = IsotropicLinearElasticMaterial(
                youngs_modulus=self.E_base, poisson_ratio=self.nu
            )

        self.sspace: LagrangeFESpace = LagrangeFESpace(
            self.mesh, p=self.degree, ctype='C'
        )
        self.space: TensorFunctionSpace = TensorFunctionSpace(
            self.sspace, shape=(-1, self.dim)
        )
        self.integrator: LinearElasticIntegrator = LinearElasticIntegrator(
            material=self.material,
            q=self.integration_order,
        )

        # i_dofs/b_dofs 按 dim * node + k 编号, 只有节点优先排布才成立.
        if self.space.dof_priority:
            raise RuntimeError(
                "自由度划分假定节点优先排布 (dof = dim * node + k), "
                "需要 TensorFunctionSpace(shape=(-1, dim))."
            )

        self.n_total_nodes: int = int(self.sspace.number_of_global_dofs())
        self.n_total_dofs: int = int(self.space.number_of_global_dofs())
        self.n_cells: int = self.mesh.number_of_cells()

        self._classify_nodes()
        self._build_node_mapping()
        self._build_cell_mappings()

        # 单位密度单元刚度阵; coef 为 None 即密度恒为 1.
        self.integrator.coef = None
        self.KE_unit: Any = self.integrator.assembly(self.space)
        self.n_element_dofs: int = self.KE_unit.shape[-1]

        # 局部刚度矩阵展平后的散加位置, 全体子结构共用.
        flat_index = (
            self.cell2dof[:, :, None] * self.n_total_dofs + self.cell2dof[:, None, :]
        )
        self._scatter_index: Any = bm.reshape(flat_index, (-1,))

        self._corner_nodes: Optional[Any] = None
        self._linear_boundary_matrix: Optional[Any] = None

    @property
    def corner_nodes(self) -> Any:
        """子结构几何角节点的局部编号, 形状 ``(2**dim,)``.

        排布顺序:
            2D (4 节点): (0,0), (1,0), (1,1), (0,1)
            3D (8 节点): (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
        """
        if self._corner_nodes is not None:
            return self._corner_nodes

        ipoints = self.sspace.interpolation_points()
        if self.dim == 2:
            corners = [
                (0.0, 0.0),
                (self.cell_size[0], 0.0),
                (self.cell_size[0], self.cell_size[1]),
                (0.0, self.cell_size[1]),
            ]
        else:
            corners = [
                (0.0, 0.0, 0.0),
                (0.0, 0.0, self.cell_size[2]),
                (0.0, self.cell_size[1], 0.0),
                (0.0, self.cell_size[1], self.cell_size[2]),
                (self.cell_size[0], 0.0, 0.0),
                (self.cell_size[0], 0.0, self.cell_size[2]),
                (self.cell_size[0], self.cell_size[1], 0.0),
                (self.cell_size[0], self.cell_size[1], self.cell_size[2]),
            ]
        corner_indices = []
        for pt in corners:
            diff = ipoints - bm.asarray(pt, dtype=bm.float64)[None, :]
            dist = bm.sum(diff ** 2, axis=-1)
            idx = int(bm.argmin(dist))
            corner_indices.append(idx)

        self._corner_nodes = bm.asarray(corner_indices, dtype=bm.int64)
        return self._corner_nodes

    @property
    def corner_dofs(self) -> Any:
        """子结构几何角节点的局部自由度编号, 形状 ``(dim * 2**dim,)``."""
        corners = self.corner_nodes
        dofs = [self.dim * corners + k for k in range(self.dim)]
        return bm.concat(dofs, axis=0)

    @property
    def linear_boundary_matrix(self) -> Any:
        """线性边界插值矩阵 L, 形状 ``(n_b, dim * 2**dim)`` (Huang 2023 式 16).

        说明:
            将 8 个角节点 (2D 为 4 个) 的粗尺度位移向量 u_c 线性插值映射至子结构外表面全部
            n_b 个细边界节点自由度: u_b = L u_c.
            各行权重严格满足单位分解性 sum(W, axis=1) == 1.
        """
        if self._linear_boundary_matrix is not None:
            return self._linear_boundary_matrix

        ipoints = self.sspace.interpolation_points()
        b_pts = ipoints[self.boundary_nodes]
        n_bnodes = len(self.boundary_nodes)

        xi = b_pts[:, 0] / self.cell_size[0]
        eta = b_pts[:, 1] / self.cell_size[1]

        if self.dim == 2:
            w0 = (1.0 - xi) * (1.0 - eta)
            w1 = xi * (1.0 - eta)
            w2 = xi * eta
            w3 = (1.0 - xi) * eta
            W = bm.stack([w0, w1, w2, w3], axis=-1)
            n_corners = 4
        else:
            zeta = b_pts[:, 2] / self.cell_size[2]
            w0 = (1.0 - xi) * (1.0 - eta) * (1.0 - zeta)
            w1 = (1.0 - xi) * (1.0 - eta) * zeta
            w2 = (1.0 - xi) * eta * (1.0 - zeta)
            w3 = (1.0 - xi) * eta * zeta
            w4 = xi * (1.0 - eta) * (1.0 - zeta)
            w5 = xi * (1.0 - eta) * zeta
            w6 = xi * eta * (1.0 - zeta)
            w7 = xi * eta * zeta
            W = bm.stack([w0, w1, w2, w3, w4, w5, w6, w7], axis=-1)
            n_corners = 8

        L = bm.zeros((self.n_b, self.dim * n_corners), dtype=bm.float64)
        for c in range(n_corners):
            for k in range(self.dim):
                row_idx = bm.arange(n_bnodes, dtype=bm.int64) * self.dim + k
                col_idx = c * self.dim + k
                L = bm.set_at(L, (row_idx, col_idx), W[:, c])

        self._linear_boundary_matrix = L
        return self._linear_boundary_matrix

    def _classify_nodes(self) -> None:
        """按几何位置划分内部节点与接口节点, 并展开为自由度索引.

        说明:
            判定容差取最小细单元高阶步长的相对量, 使分类不随模型的物理量纲变化.
        """
        ipoints = self.sspace.interpolation_points()
        h_min = min(
            s / (n * self.degree) for s, n in zip(self.cell_size, self.n_fine)
        )
        eps = 1.0e-6 * h_min

        is_boundary: Any = bm.zeros((ipoints.shape[0],), dtype=bm.bool)
        for d in range(self.dim):
            is_boundary |= bm.abs(ipoints[:, d]) < eps
            is_boundary |= bm.abs(ipoints[:, d] - self.cell_size[d]) < eps

        self.boundary_nodes: Any = bm.nonzero(is_boundary)[0]
        self.internal_nodes: Any = bm.nonzero(~is_boundary)[0]

        self.i_dofs: Any = bm.sort(
            bm.concat([self.dim * self.internal_nodes + k for k in range(self.dim)])
        )
        self.b_dofs: Any = bm.sort(
            bm.concat([self.dim * self.boundary_nodes + k for k in range(self.dim)])
        )
        self.n_i: int = len(self.i_dofs)
        self.n_b: int = len(self.b_dofs)

    def _build_node_mapping(self) -> None:
        """由插值点坐标反解每个局部节点的结构化网格下标.

        异常:
            RuntimeError: 当局部节点无法与结构化高阶网格下标一一对应时抛出.
        """
        ipoints = self.sspace.interpolation_points()
        n_nodes_per_dir = tuple(n * self.degree + 1 for n in self.n_fine)

        columns = []
        linear_index: Any = bm.zeros((self.n_total_nodes,), dtype=bm.int64)
        for d in range(self.dim):
            h_d = self.cell_size[d] / (self.n_fine[d] * self.degree)
            idx_d = bm.astype(bm.round(ipoints[:, d] / h_d), bm.int64)
            idx_d = bm.clip(idx_d, 0, self.n_fine[d] * self.degree)
            columns.append(idx_d)
            linear_index = linear_index * n_nodes_per_dir[d] + idx_d

        if not bool(
            bm.all(
                bm.sort(linear_index)
                == bm.arange(self.n_total_nodes, dtype=bm.int64)
            )
        ):
            raise RuntimeError(
                "局部插值节点未能与结构化网格下标一一对应, 无法建立子结构到全局的映射."
            )

        # (n_nodes, dim): 第 n 行是局部插值节点 n 在子结构高阶结构化网格中的整数下标.
        self.node_grid_index: Any = bm.stack(columns, axis=-1)
        self.n_nodes_per_dir: Tuple[int, ...] = n_nodes_per_dir

    def _build_cell_mappings(self) -> None:
        """建立单元自由度映射, 以及结构化网格下标到单元编号的对应关系.

        异常:
            RuntimeError: 当细单元的重心无法与 ``n_fine`` 给出的结构化下标一一
                对应时抛出.

        说明:
            密度场常按 ``n_fine`` 形状的结构化数组给出, 而单元刚度按网格自身的
            单元编号排列. 两者的次序由网格生成器决定, 不应假定一致. 这里用单元
            重心反解结构化下标, 把该对应关系显式算出来并校验为一个置换.
        """
        self.cell2dof: Any = self.space.cell_to_dof()

        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        barycenter = bm.mean(node[cell], axis=1)

        # 由重心反解各方向的结构化下标, 再按 C 序压成线性下标.
        linear_index: Any = bm.zeros((self.n_cells,), dtype=bm.int64)
        for d in range(self.dim):
            h_d = self.cell_size[d] / self.n_fine[d]
            idx_d = bm.astype(bm.floor(barycenter[:, d] / h_d), bm.int64)
            idx_d = bm.clip(idx_d, 0, self.n_fine[d] - 1)
            linear_index = linear_index * self.n_fine[d] + idx_d

        if not bool(
            bm.all(
                bm.sort(linear_index) == bm.arange(self.n_cells, dtype=bm.int64)
            )
        ):
            raise RuntimeError(
                "细单元重心未能与结构化网格下标一一对应, "
                "无法建立密度场到单元编号的映射."
            )
        # ``linear_index[cell_id] = grid_id``. 两个方向都显式保存, 避免调用方
        # 猜测 FEALPy 单元编号与结构化 C 序是否一致.
        self._cell_grid_index: Any = linear_index
        inverse: Any = bm.zeros((self.n_cells,), dtype=bm.int64)
        inverse = bm.set_at(
            inverse,
            linear_index,
            bm.arange(self.n_cells, dtype=bm.int64),
        )
        self._grid_cell_index: Any = inverse

    ### 接口自由度上的刚体模态与变形子空间 ###

    @property
    def n_rigid(self) -> int:
        """刚体模态数: 二维为 3, 三维为 6."""
        return self.dim * (self.dim + 1) // 2

    @property
    def rigid_basis(self) -> Any:
        """接口自由度上刚体模态的标准正交基, 形状 ``(n_b, n_rigid)``.

        说明:
            自由漂浮子结构的 Schur 补 ``K_s`` 恰以刚体模态为零空间: 若整体位移
            ``(u_b, u_i)`` 是刚体运动则 ``K @ (u_b, u_i) = 0``, 由此
            ``u_i = N u_b`` 且 ``K_s u_b = 0``. 一次线性单元能精确表示刚体运动,
            因此该零空间是精确的而非近似的.

            基由接口节点坐标解析构造, **与密度无关**, 全体同构子结构共用, 只在
            首次访问时计算并缓存. 返回的是 QR 正交化后的基, 张成的子空间与物理
            平动/转动模态相同, 但各列不逐一对应某个物理模态.
        """
        if self._rigid_basis is None:
            self._build_interface_bases()
        return self._rigid_basis

    @property
    def deformation_basis(self) -> Any:
        """刚体子空间的标准正交补, 形状 ``(n_b, n_b - n_rigid)``.

        说明:
            ``K_s`` 限制到该子空间上严格正定 (其特征值即 ``K_s`` 的全部非零特征值),
            无需正则化即可取 Cholesky 因子. 代理若按
            ``K_s = R_perp L L^T R_perp^T`` 参数化, 则 ``K_s`` 的秩亏结构成为构造
            性质: 拟合误差无法泄漏进刚体模态, 而刚体模态正是装配后子结构位移的
            主要成分.
        """
        if self._deformation_basis is None:
            self._build_interface_bases()
        return self._deformation_basis

    @property
    def rigid_interior_modes(self) -> Any:
        """刚体运动下内部自由度的取值, 形状 ``(n_i, n_rigid)``.

        说明:
            记 ``R_rigid`` 为接口刚体模态基, 该量满足 ``N R_rigid = Phi_i``, 其中
            ``N`` 是精确内部位移恢复矩阵: 子结构做刚体运动时内部位移完全由接口位移
            决定, 且**与密度无关**. 因此它由网格解析给出, 无需任何有限元装配或缩聚.

            该性质是形函数代理参数化 ``N = Phi_i R_rigid^T + M R_perp^T`` 的前提,
            使刚体分量成为构造性质而不进入网络输出. 与 ``rigid_basis`` 共用同一
            转动中心, 两者必须来自同一个刚体位移场, 否则该恒等式不成立.
        """
        if self._rigid_interior is None:
            self._build_interface_bases()
        return self._rigid_interior

    def trace_interface_bases(
        self,
        trace: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any]:
        """构造给定接口迹空间上的刚体基, 变形子空间基与刚体内部取值.

        参数:
            trace: ``TraceBasis`` 实例, 提供形状 ``(n_b, n_q)`` 的迹矩阵 ``T``,
                迹坐标 ``q`` 与完整接口位移满足 ``u_b = T q``. 为 ``None`` 时取
                完整接口 ``T = I``, 返回值即 ``rigid_basis``,
                ``deformation_basis`` 与 ``rigid_interior_modes`` 三个属性本身.

        返回:
            (R_q, R_perp, Phi_i): 形状分别为 ``(n_q, n_rigid)``,
                ``(n_q, n_q - n_rigid)`` 与 ``(n_i, n_rigid)``. 三者满足
                ``B R_q = Phi_i``, 其中 ``B = N T`` 是迹空间下的内部延拓.

        异常:
            ValueError: ``T`` 的行数与接口自由度数 ``n_b`` 不符时抛出.
            RuntimeError: 迹自由度数少于刚体模态数, 或迹空间不能精确表示刚体
                运动时抛出. 后者意味着该迹空间不是容许的宏观位移空间, 装配后
                子结构无法自由做刚体运动, 缩聚刚度的零空间结构随之失效.

        说明:
            迹空间下的刚体坐标由 ``T Q = R_rigid`` 解出并校验残量: 完整接口上的
            刚体模态必须落在 ``T`` 的列空间内. 角点线性迹满足这一条件, 因为刚体
            位移在每条棱上是线性的, 而双线性插值沿棱退化为线性插值.

            ``Phi_i`` 不经任何求逆得到: ``T R_q`` 仍是刚体边界场, 它在标准正交基
            ``R_rigid`` 下的坐标为 ``R_rigid^T T R_q``, 于是
            ``Phi_i = N T R_q = (N R_rigid)(R_rigid^T T R_q)``, 括号内即完整接口
            上的 ``rigid_interior_modes``, 与密度无关.
        """
        if trace is None:
            return (
                self.rigid_basis,
                self.deformation_basis,
                self.rigid_interior_modes,
            )

        T = bm.asarray(trace.matrix, dtype=bm.float64)
        if int(T.shape[0]) != self.n_b:
            raise ValueError(
                f"迹矩阵的行数应为接口自由度数 {self.n_b}; "
                f"当前形状为 {tuple(T.shape)}."
            )
        n_q = int(T.shape[1])
        if n_q < self.n_rigid:
            raise RuntimeError(
                f"迹自由度数 {n_q} 少于刚体模态数 {self.n_rigid}, "
                f"无法构造迹空间刚体基."
            )

        full_rigid = self.rigid_basis
        q_rigid = bm.linalg.pinv(T) @ full_rigid
        residual = float(
            bm.linalg.norm(T @ q_rigid - full_rigid)
            / bm.linalg.norm(full_rigid)
        )
        if not residual <= self._TRACE_RIGID_TOL:
            raise RuntimeError(
                f"迹空间不能精确表示接口刚体模态, 相对残量 {residual:.3e} "
                f"超过 {self._TRACE_RIGID_TOL:.3e}."
            )

        # 完整 QR: 前 n_rigid 列张成迹空间的刚体子空间, 其余列构成其正交补.
        Q, _ = bm.linalg.qr(q_rigid, mode="complete")
        R_q = Q[:, :self.n_rigid]
        R_perp = Q[:, self.n_rigid:]
        Phi_i = self.rigid_interior_modes @ (
            bm.matrix_transpose(full_rigid) @ T @ R_q
        )
        return R_q, R_perp, Phi_i

    def _rigid_modes_on(self, dofs: Any, centroid: Any) -> Any:
        """构造刚体位移场在给定自由度集合上的取值.

        参数:
            dofs: 局部自由度编号, 形状 ``(n,)``.
            centroid: 转动中心, 形状 ``(dim,)``. 接口与内部自由度必须传入同一中心,
                否则两组取值对应的不是同一个刚体位移场.

        返回:
            modes: 形状 ``(n, n_rigid)``, 前 ``dim`` 列为平动, 其余为转动.

        说明:
            自由度按 ``dim * node + k`` 编号, 由此反解每个自由度所属的节点与分量.
            平动模态在对应分量上取 1; 转动模态取 ``e_a x (x - x_c)``, 每个自由度
            只取该模态在自身分量方向上的分量.
        """
        node = self.mesh.entity('node')
        d_node = dofs // self.dim
        d_comp = dofs % self.dim
        offset = node[d_node] - centroid[None, :]

        n_dof = len(dofs)
        rows = bm.arange(n_dof, dtype=bm.int64)
        columns = [bm.astype(d_comp == k, bm.float64) for k in range(self.dim)]

        zero = bm.zeros((n_dof,), dtype=bm.float64)
        if self.dim == 2:
            # 绕 z 轴转动: u = (-dy, dx).
            rotations = [bm.stack([-offset[:, 1], offset[:, 0]], axis=-1)]
        else:
            # 绕各坐标轴转动: u = e_a x d.
            rotations = [
                bm.stack([zero, -offset[:, 2], offset[:, 1]], axis=-1),
                bm.stack([offset[:, 2], zero, -offset[:, 0]], axis=-1),
                bm.stack([-offset[:, 1], offset[:, 0], zero], axis=-1),
            ]
        columns.extend(vec[rows, d_comp] for vec in rotations)

        return bm.stack(columns, axis=-1)

    def _build_interface_bases(self) -> None:
        """解析构造接口刚体模态基, 其正交补, 以及刚体运动下的内部自由度取值.

        异常:
            RuntimeError: 当接口自由度数不足以容纳全部刚体模态时抛出.

        说明:
            形心取接口节点而非全部节点, 与基的构造对象一致. 形心的选取只影响基的
            表示, 不影响张成的子空间——平动模态已在基中, 任何常向量平移都被吸收;
            但接口与内部两组模态必须共用同一形心, 否则 ``N R_rigid = Phi_i`` 不再
            成立.

            ``Phi_i`` 由 QR 的上三角因子换基得到: 由 ``modes_b = R_rigid R_up`` 与
            ``N modes_b = modes_i`` 得 ``Phi_i = N R_rigid = modes_i R_up^{-1}``,
            全程不涉及局部刚度矩阵.
        """
        if self.n_b < self.n_rigid:
            raise RuntimeError(
                f"接口自由度数 {self.n_b} 少于刚体模态数 {self.n_rigid}, "
                f"无法构造刚体模态基."
            )

        node = self.mesh.entity('node')
        centroid = bm.mean(node[self.boundary_nodes], axis=0)
        modes_b = self._rigid_modes_on(self.b_dofs, centroid)

        # 完整 QR: Q 的前 n_rigid 列张成刚体子空间, 其余列构成其正交补.
        Q, R = bm.linalg.qr(modes_b, mode='complete')
        self._rigid_basis = Q[:, :self.n_rigid]
        self._deformation_basis = Q[:, self.n_rigid:]

        # 解 Phi_i R_up = modes_i, 即换到 R_rigid 这组标准正交基下的内部取值.
        modes_i = self._rigid_modes_on(self.i_dofs, centroid)
        R_up = R[:self.n_rigid, :]
        self._rigid_interior = bm.matrix_transpose(
            bm.linalg.solve(bm.matrix_transpose(R_up), bm.matrix_transpose(modes_i))
        )

    def grid_to_cell_field(self, grid_field: Any) -> Any:
        """把局部结构化网格场重排为 prototype FE cell 顺序.

        参数:
            grid_field: 局部结构化场, 末尾若干维必须为 ``n_fine``; 允许携带
                任意前导批量维 ``...``.

        返回:
            cell_field: 形状 ``(..., NC)`` 的场, 最后一维按 FEALPy prototype
                单元编号排列.

        异常:
            ValueError: 当末尾维度不是 ``n_fine`` 时抛出.
        """
        field = bm.asarray(grid_field)
        n_grid = len(self.n_fine)
        if field.ndim < n_grid or tuple(field.shape[-n_grid:]) != self.n_fine:
            raise ValueError(
                f"局部结构化场形状 {tuple(field.shape)} 不匹配 n_fine={self.n_fine}."
            )

        leading = tuple(field.shape[:-n_grid])
        flat = bm.reshape(field, leading + (self.n_cells,))
        return flat[..., self._cell_grid_index]

    def cell_to_grid_field(self, cell_field: Any) -> Any:
        """把 prototype FE cell 顺序的场重排为局部结构化网格场.

        参数:
            cell_field: 按 FEALPy prototype 单元编号排列的场, 形状
                ``(..., NC)``.

        返回:
            grid_field: 末尾若干维为 ``n_fine`` 的局部结构化场, 按 C 序解释.

        异常:
            ValueError: 当最后一维不是 ``NC`` 时抛出.
        """
        field = bm.asarray(cell_field)
        if field.ndim < 1 or field.shape[-1] != self.n_cells:
            raise ValueError(
                f"FE cell 场形状 {tuple(field.shape)} 的末维必须为 NC={self.n_cells}."
            )

        leading = tuple(field.shape[:-1])
        grid_flat = field[..., self._grid_cell_index]
        return bm.reshape(grid_flat, leading + self.n_fine)

    def to_cell_density(self, density: Any) -> Any:
        """把密度场统一为按单元编号排列的形式.

        参数:
            density: 密度场. 末尾若干维为 ``n_fine`` 时按结构化网格解释, 并重排
                到单元编号次序; 末维为 ``NC`` 时视为已按单元编号排列. 两种形式
                都允许携带任意前导维 ``...``.

        返回:
            rho: 形状 ``(..., NC)`` 的单元密度.

        异常:
            ValueError: 当密度场的形状既不匹配 ``n_fine`` 也不匹配 ``NC`` 时抛出.
        """
        rho = bm.asarray(density)
        grid_shape = self.n_fine
        n_grid = len(grid_shape)

        if rho.ndim >= n_grid and tuple(rho.shape[-n_grid:]) == grid_shape:
            return self.grid_to_cell_field(rho)

        if rho.ndim >= 1 and rho.shape[-1] == self.n_cells:
            return rho

        raise ValueError(
            f"密度场形状 {tuple(rho.shape)} 无法解释: "
            f"末尾若干维应为 {grid_shape}, 或末维应为 {self.n_cells}."
        )

    def assemble_local_stiffness_batch(
        self,
        density: Any,
        *,
        chunk_size: Optional[int] = None,
    ) -> Any:
        """批量装配局部刚度矩阵.

        参数:
            density: 密度场, 形状约定见 ``to_cell_density``.
            chunk_size: 每批处理的子结构数. 为 ``None`` 时一次处理全部子结构.
                散加过程需要 ``B * NC * n_edof**2`` 量级的索引与数值缓冲, 子结构
                数很大时用该参数限制峰值内存.

        返回:
            K_local: 局部刚度矩阵, 形状 ``(..., n_dof, n_dof)``, 前导维与
                ``density`` 去掉密度轴后的前导维一致.

        异常:
            ValueError: 当 ``chunk_size`` 非正时抛出.

        说明:
            单元刚度由 ``coef(rho) * KE_unit`` 得到, 再一次性散加到局部矩阵.
            散加使用 ``bm.bincount`` 而非 ``bm.add_at``: 后者在 NumPy 后端走的是
            无缓冲的 ``np.add.at``, 在 PyTorch 后端则被上游标注为非确定性.

            单元刚度阵由积分子以 ``float64`` 产出, 因此装配结果恒为 ``float64``.
        """
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正整数; 当前为 {chunk_size}.")

        rho_cells = self.to_cell_density(density)
        leading = tuple(rho_cells.shape[:-1])
        rho_flat = bm.reshape(rho_cells, (-1, self.n_cells))
        n_batch = rho_flat.shape[0]

        step = n_batch if chunk_size is None else min(chunk_size, n_batch)
        if step == 0:
            blocks = [bm.zeros((0, self.n_total_dofs, self.n_total_dofs),
                               dtype=bm.float64)]
        else:
            blocks = [
                self._assemble_chunk(rho_flat[start:start + step])
                for start in range(0, n_batch, step)
            ]

        stacked = blocks[0] if len(blocks) == 1 else bm.concat(blocks, axis=0)
        return bm.reshape(
            stacked, leading + (self.n_total_dofs, self.n_total_dofs)
        )

    def iter_local_stiffness_batches(
        self,
        density: Any,
        *,
        chunk_size: int,
    ) -> Iterator[Tuple[int, int, Any]]:
        """按子结构批次流式装配局部刚度矩阵.

        参数:
            density: 密度场, 形状约定见 ``to_cell_density``. 去掉密度轴后的
                全部前导维会按 C 序展平为子结构批量维.
            chunk_size: 单次装配的最大子结构数, 必须为正整数.

        生成:
            (start, end, K_local_chunk): 当前批次在展平子结构批量中的半开区间
            ``[start, end)`` 及其局部刚度矩阵, 后者形状为
            ``(end - start, n_dof, n_dof)``.

        异常:
            ValueError: 当 ``chunk_size`` 非正时抛出.

        说明:
            与 ``assemble_local_stiffness_batch`` 不同, 本方法不保存已生成批次,
            也不在末尾执行 ``bm.concat``. 调用方消费一个批次后即可释放局部刚度
            矩阵, 从而把峰值内存限制在 ``chunk_size`` 对应的规模.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正整数; 当前为 {chunk_size}.")

        rho_cells = self.to_cell_density(density)
        rho_flat = bm.reshape(rho_cells, (-1, self.n_cells))
        n_batch = rho_flat.shape[0]

        for start in range(0, n_batch, chunk_size):
            end = min(start + chunk_size, n_batch)
            yield start, end, self._assemble_chunk(rho_flat[start:end])

    def _assemble_chunk(self, rho_chunk: Any) -> Any:
        """装配一批局部刚度矩阵.

        参数:
            rho_chunk: 按单元编号排列的密度, 形状 ``(b, NC)``.

        返回:
            K_local: 形状 ``(b, n_dof, n_dof)`` 的局部刚度矩阵.
        """
        n_chunk = rho_chunk.shape[0]
        n_dof = self.n_total_dofs

        # SIMP 插值: rho_min 为 0 时退化为纯幂律, 与 soptx 的 'simp' 方案一致.
        coef = rho_chunk ** self.penal
        if self.rho_min != 0.0:
            coef = self.rho_min + (1.0 - self.rho_min) * coef

        KE = bm.einsum('be, eij -> beij', coef, self.KE_unit)

        # 把 batch 编号并入平坦索引, 一次 bincount 完成全批散加.
        offsets = bm.arange(n_chunk, dtype=bm.int64) * (n_dof * n_dof)
        indices = bm.reshape(offsets[:, None] + self._scatter_index[None, :], (-1,))
        accumulated = bm.bincount(
            indices,
            weights=bm.reshape(KE, (-1,)),
            minlength=n_chunk * n_dof * n_dof,
        )
        K_local = bm.reshape(accumulated, (n_chunk, n_dof, n_dof))

        return K_local

    def compute_element_strain_energy(
        self,
        u_full: Any,
        cell_to_dof: Any,
    ) -> Any:
        """计算全场所有细单元在单位刚度下的变形应变能: (u_e)^T K_0 u_e.

        参数:
            u_full: 全场细观恢复位移向量, 形状 ``(total_full_dofs,)``.
            cell_to_dof: 全局细单元到全尺度自由度的映射, 形状 ``(n_elem_total, n_edof)``.

        返回:
            energy: 各细单元的单位基准变形应变能, 形状 ``(n_elem_total,)``.

        说明:
            全场所有单元为同构规则单元, 共用单个基准单元刚度矩阵 ``K_0 = self.KE_unit[0]``.
            通过向量化点乘直接计算应变能, 峰值内存仅取决于 ``uhe`` (数百 MB),
            彻底避免组装数十 GB 的全场刚度导数张量.
        """
        K0 = self.KE_unit[0]
        uhe = u_full[cell_to_dof]
        energy = bm.sum((uhe @ K0) * uhe, axis=-1)
        return energy


class SubstructureMesh:
    """
    单个子结构: 参考子结构加上它在全局模型中的位置.

    离散结构全部由 ``SubstructurePrototype`` 提供并可跨子结构共享; 本类只额外
    记录 ``sub_id`` 与 ``box_span``. 常用的离散属性以委托形式转发到原型, 因此
    调用方无需区分两者.
    """

    def __init__(
        self,
        sub_id: int,
        *args: Any,
        E_base: float = 1.0,
        nu: float = 0.3,
        prototype: Optional[SubstructurePrototype] = None,
        penal: float = 3.0,
        rho_min: float = 0.0,
    ) -> None:
        """
        构造单个子结构.

        参数:
            sub_id: 子结构编号.
            *args: 依次为各方向的坐标区间 (2 个或 3 个二元组), 各方向的细单元数
                (与区间个数相同的整数), 以及可选的 ``E_base`` 与 ``nu``.
            E_base: 实体材料的杨氏模量, 被 ``*args`` 中的同名位置参数覆盖.
            nu: 泊松比, 被 ``*args`` 中的同名位置参数覆盖.
            prototype: 共享的参考子结构. 为 ``None`` 时按本子结构的尺寸新建一个;
                同一模型中的全部子结构应传入同一个原型以避免重复构造.
            penal: SIMP 惩罚指数, 仅在新建原型时生效.
            rho_min: SIMP 刚度下限比值, 仅在新建原型时生效.

        异常:
            ValueError: 当 ``*args`` 无法解析为坐标区间加细单元数, 或传入的原型
                与本子结构的尺寸, 细划分, 材料参数不一致时抛出.
        """
        self.sub_id: int = sub_id

        spans, n_fine, E_base, nu = self._parse_geometry(args, E_base, nu)
        self.box_span: Tuple[Tuple[float, float], ...] = spans
        self.dim: int = len(spans)
        cell_size = tuple(hi - lo for lo, hi in spans)

        if prototype is None:
            prototype = SubstructurePrototype(
                cell_size, n_fine, E_base, nu, penal=penal, rho_min=rho_min
            )
        else:
            self._check_prototype(prototype, cell_size, n_fine, E_base, nu)
        self.prototype: SubstructurePrototype = prototype

        # 位置相关的属性别名; 尺寸与离散相关的属性一律委托给原型.
        self.x_span = self.box_span[0]
        self.y_span = self.box_span[1]
        if self.dim == 3:
            self.z_span = self.box_span[2]

    @staticmethod
    def _parse_geometry(
        args: Tuple[Any, ...],
        E_base: float,
        nu: float,
    ) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[int, ...], float, float]:
        """解析坐标区间, 细单元数与可选的材料参数.

        参数:
            args: ``__init__`` 收到的可变位置参数.
            E_base: 关键字形式给出的杨氏模量默认值.
            nu: 关键字形式给出的泊松比默认值.

        返回:
            (spans, n_fine, E_base, nu): 坐标区间元组, 各方向细单元数, 以及解析
                后的材料参数.

        异常:
            ValueError: 当区间个数不是 2 或 3, 或细单元数个数与区间个数不符时抛出.
        """
        spans: list = []
        cursor = 0
        while cursor < len(args) and isinstance(args[cursor], (tuple, list)):
            span = tuple(args[cursor])
            if len(span) != 2:
                raise ValueError(f"坐标区间必须是二元组; 当前为 {span}.")
            spans.append((float(span[0]), float(span[1])))
            cursor += 1

        dim = len(spans)
        if dim not in (2, 3):
            raise ValueError(
                f"必须给出 2 个或 3 个坐标区间; 当前解析到 {dim} 个."
            )
        if len(args) - cursor < dim:
            raise ValueError(
                f"{dim}D 子结构需要 {dim} 个细单元数; "
                f"当前只余 {len(args) - cursor} 个位置参数."
            )

        n_fine = tuple(int(v) for v in args[cursor:cursor + dim])
        cursor += dim

        rest = args[cursor:]
        if len(rest) > 0:
            E_base = float(rest[0])
        if len(rest) > 1:
            nu = float(rest[1])
        return tuple(spans), n_fine, E_base, nu

    @staticmethod
    def _check_prototype(
        prototype: SubstructurePrototype,
        cell_size: Tuple[float, ...],
        n_fine: Tuple[int, ...],
        E_base: float,
        nu: float,
    ) -> None:
        """校验共享原型与本子结构一致.

        参数:
            prototype: 待校验的参考子结构.
            cell_size: 本子结构在各方向的物理尺寸.
            n_fine: 本子结构在各方向的细单元数.
            E_base: 本子结构的杨氏模量.
            nu: 本子结构的泊松比.

        异常:
            ValueError: 当尺寸, 细划分或材料参数与原型不一致时抛出. 尺寸按相对
                容差 ``1e-12`` 比较, 其余按精确相等比较.
        """
        if prototype.n_fine != n_fine:
            raise ValueError(
                f"原型的细划分为 {prototype.n_fine}, 与本子结构的 {n_fine} 不一致."
            )
        for proto_size, size in zip(prototype.cell_size, cell_size):
            if abs(proto_size - size) > 1.0e-12 * max(abs(size), 1.0):
                raise ValueError(
                    f"原型的尺寸为 {prototype.cell_size}, "
                    f"与本子结构的 {cell_size} 不一致; "
                    f"共享原型要求全部子结构同构."
                )
        if prototype.E_base != float(E_base) or prototype.nu != float(nu):
            raise ValueError(
                f"原型的材料参数为 (E={prototype.E_base}, nu={prototype.nu}), "
                f"与本子结构的 (E={E_base}, nu={nu}) 不一致."
            )

    ### 委托给原型的离散属性 ###

    @property
    def mesh(self) -> Any:
        """参考子结构的细网格."""
        return self.prototype.mesh

    @property
    def material(self) -> Any:
        """参考子结构的材料模型."""
        return self.prototype.material

    @property
    def sspace(self) -> LagrangeFESpace:
        """参考子结构的标量拉格朗日空间."""
        return self.prototype.sspace

    @property
    def space(self) -> TensorFunctionSpace:
        """参考子结构的张量位移空间."""
        return self.prototype.space

    @property
    def integrator(self) -> LinearElasticIntegrator:
        """参考子结构的线弹性积分子."""
        return self.prototype.integrator

    @property
    def n_fine(self) -> Tuple[int, ...]:
        """各方向的细单元数."""
        return self.prototype.n_fine

    @property
    def n_fine_x(self) -> int:
        """x 方向的细单元数."""
        return self.prototype.n_fine[0]

    @property
    def n_fine_y(self) -> int:
        """y 方向的细单元数."""
        return self.prototype.n_fine[1]

    @property
    def n_fine_z(self) -> int:
        """z 方向的细单元数, 仅 3D 可用."""
        return self.prototype.n_fine[2]

    @property
    def n_nodes_x(self) -> int:
        """x 方向的节点数."""
        return self.prototype.n_fine[0] + 1

    @property
    def n_nodes_y(self) -> int:
        """y 方向的节点数."""
        return self.prototype.n_fine[1] + 1

    @property
    def n_nodes_z(self) -> int:
        """z 方向的节点数, 仅 3D 可用."""
        return self.prototype.n_fine[2] + 1

    @property
    def n_total_nodes(self) -> int:
        """子结构的节点总数."""
        return self.prototype.n_total_nodes

    @property
    def n_total_dofs(self) -> int:
        """子结构的局部自由度总数."""
        return self.prototype.n_total_dofs

    @property
    def internal_nodes(self) -> Any:
        """内部节点编号."""
        return self.prototype.internal_nodes

    @property
    def boundary_nodes(self) -> Any:
        """边界或接口节点编号."""
        return self.prototype.boundary_nodes

    @property
    def node_grid_index(self) -> Any:
        """各局部节点在子结构结构化网格中的整数下标, 形状 ``(n_nodes, dim)``."""
        return self.prototype.node_grid_index

    @property
    def i_dofs(self) -> Any:
        """内部自由度的局部编号."""
        return self.prototype.i_dofs

    @property
    def b_dofs(self) -> Any:
        """接口自由度的局部编号."""
        return self.prototype.b_dofs

    @property
    def n_i(self) -> int:
        """内部自由度数."""
        return self.prototype.n_i

    @property
    def n_b(self) -> int:
        """接口自由度数."""
        return self.prototype.n_b

    @property
    def n_rigid(self) -> int:
        """刚体模态数."""
        return self.prototype.n_rigid

    @property
    def rigid_basis(self) -> Any:
        """接口自由度上刚体模态的标准正交基."""
        return self.prototype.rigid_basis

    @property
    def deformation_basis(self) -> Any:
        """刚体子空间的标准正交补."""
        return self.prototype.deformation_basis

    @property
    def rigid_interior_modes(self) -> Any:
        """刚体运动下内部自由度的取值, 形状 ``(n_i, n_rigid)``."""
        return self.prototype.rigid_interior_modes

    def assemble_local_stiffness(self, density_field: Any) -> Any:
        """装配本子结构的局部刚度矩阵.

        参数:
            density_field: 密度场, 形状为 ``n_fine`` 或 ``(NC,)``.

        返回:
            K_local: 形状 ``(n_dof, n_dof)`` 的局部刚度矩阵.

        说明:
            等价于对原型调用批量装配并取唯一一个元素; 需要一次装配多个子结构时,
            直接使用 ``prototype.assemble_local_stiffness_batch`` 可避免逐个调用.
        """
        return self.prototype.assemble_local_stiffness_batch(density_field)


def build_substructures(
    assembler: Any,
    *,
    integration_order: Optional[int] = None,
) -> Tuple[SubstructurePrototype, List[SubstructureMesh], List[Tuple[int, ...]]]:
    """按装配器的布局铺开全部子结构, 共享同一个参考子结构.

    参数:
        assembler: 已构造的全局装配器, 提供求解域尺寸与子结构划分.
        integration_order: 单元数值积分阶数; ``None`` 使用积分器缺省值.

    返回:
        (prototype, sub_meshes, positions): 共享的参考子结构, 按 x 优先字典序排列的
            子结构列表, 以及各子结构在子结构网格中的整数位置 ``(sx, sy)``. 位置与
            ``sub_meshes`` 同序, 供 ``get_substructure_global_dofs`` 把局部自由度映射
            到全局编号. 全部子结构同构, 因此离散结构, 自由度划分与单位密度单元刚度
            只构造一次.
    """
    sub_size = tuple(
        assembler.domain_size[d] / assembler.n_sub[d] for d in range(assembler.dim)
    )
    prototype = SubstructurePrototype(
        sub_size,
        assembler.n_fine,
        assembler.E_base,
        assembler.nu,
        degree=assembler.degree,
        integration_order=integration_order,
    )

    sub_meshes: List[SubstructureMesh] = []
    positions: List[Tuple[int, ...]] = []
    sub_id = 0
    if assembler.dim == 2:
        for sx in range(assembler.n_sub[0]):
            for sy in range(assembler.n_sub[1]):
                spans = (
                    (sx * sub_size[0], (sx + 1) * sub_size[0]),
                    (sy * sub_size[1], (sy + 1) * sub_size[1]),
                )
                sub_meshes.append(
                    SubstructureMesh(
                        sub_id, *spans, *assembler.n_fine,
                        E_base=assembler.E_base, nu=assembler.nu, prototype=prototype,
                    )
                )
                positions.append((sx, sy))
                sub_id += 1
    elif assembler.dim == 3:
        for sx in range(assembler.n_sub[0]):
            for sy in range(assembler.n_sub[1]):
                for sz in range(assembler.n_sub[2]):
                    spans = (
                        (sx * sub_size[0], (sx + 1) * sub_size[0]),
                        (sy * sub_size[1], (sy + 1) * sub_size[1]),
                        (sz * sub_size[2], (sz + 1) * sub_size[2]),
                    )
                    sub_meshes.append(
                        SubstructureMesh(
                            sub_id, *spans, *assembler.n_fine,
                            E_base=assembler.E_base, nu=assembler.nu, prototype=prototype,
                        )
                    )
                    positions.append((sx, sy, sz))
                    sub_id += 1
    return prototype, sub_meshes, positions
