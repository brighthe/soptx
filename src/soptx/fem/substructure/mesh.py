"""子结构网格与局部刚度装配.

同一模型中的全部子结构在几何上互为平移, 因此离散结构 (网格拓扑, 函数空间,
内部/接口自由度划分, 单位密度单元刚度阵) 与位置无关, 由 ``SubstructurePrototype``
持有并只构造一次. ``SubstructureMesh`` 是原型加上一个全局位置的轻量封装.

局部刚度装配接口接受任意可变前导维 ``...``: 单个子结构是前导维为空的特例,
批量子结构对应前导维 ``B``, 与 ``condensation`` 模块的形状约定一致.
"""

from typing import Tuple, Any, Optional, Sequence

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

    # 接口自由度上刚体模态基与其正交补的惰性缓存.
    _rigid_basis: Optional[Any] = None
    _deformation_basis: Optional[Any] = None

    def __init__(
        self,
        cell_size: Sequence[float],
        n_fine: Sequence[int],
        E_base: float = 1.0,
        nu: float = 0.3,
        *,
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
            penal: SIMP 惩罚指数.
            rho_min: SIMP 刚度下限比值, 取 ``0.0`` 时插值为 ``rho**penal``,
                取正值时为 ``rho_min + (1 - rho_min) * rho**penal``.

        异常:
            ValueError: 当维数不是 2 或 3, ``cell_size`` 与 ``n_fine`` 长度不一致,
                或存在非正的尺寸与单元数时抛出.
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

        self.sspace: LagrangeFESpace = LagrangeFESpace(self.mesh, p=1, ctype='C')
        self.space: TensorFunctionSpace = TensorFunctionSpace(
            self.sspace, shape=(-1, self.dim)
        )
        self.integrator: LinearElasticIntegrator = LinearElasticIntegrator(
            material=self.material
        )

        # i_dofs/b_dofs 按 dim * node + k 编号, 只有节点优先排布才成立.
        if self.space.dof_priority:
            raise RuntimeError(
                "自由度划分假定节点优先排布 (dof = dim * node + k), "
                "需要 TensorFunctionSpace(shape=(-1, dim))."
            )

        self.n_total_nodes: int = self.mesh.number_of_nodes()
        self.n_total_dofs: int = self.space.number_of_global_dofs()
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

    def _classify_nodes(self) -> None:
        """按几何位置划分内部节点与接口节点, 并展开为自由度索引.

        说明:
            判定容差取最小细单元尺寸的相对量, 使分类不随模型的物理量纲变化;
            使用绝对容差时, 以毫米建模的大构件会把内部节点误判为接口节点.
        """
        node_coords = self.mesh.entity('node')
        h_min = min(s / n for s, n in zip(self.cell_size, self.n_fine))
        eps = 1.0e-6 * h_min

        # 显式标注为 Any: TensorLike 联合类型不支持 |= 运算符的类型推断.
        is_boundary: Any = bm.zeros((node_coords.shape[0],), dtype=bm.bool)
        for d in range(self.dim):
            is_boundary |= bm.abs(node_coords[:, d]) < eps
            is_boundary |= bm.abs(node_coords[:, d] - self.cell_size[d]) < eps

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
        """由节点坐标反解每个局部节点的结构化网格下标.

        异常:
            RuntimeError: 当局部节点无法与 ``n_fine`` 给出的结构化下标一一对应时抛出.

        说明:
            子结构在全局模型中的自由度编号由它的结构化下标加上子结构偏移得到.
            这里把局部节点编号到结构化下标的对应关系显式算出来并校验为置换,
            使装配不依赖网格生成器的节点编号次序.
        """
        node = self.mesh.entity('node')
        n_nodes_per_dir = tuple(n + 1 for n in self.n_fine)

        columns = []
        linear_index: Any = bm.zeros((self.n_total_nodes,), dtype=bm.int64)
        for d in range(self.dim):
            h_d = self.cell_size[d] / self.n_fine[d]
            idx_d = bm.astype(bm.round(node[:, d] / h_d), bm.int64)
            idx_d = bm.clip(idx_d, 0, self.n_fine[d])
            columns.append(idx_d)
            linear_index = linear_index * n_nodes_per_dir[d] + idx_d

        if not bool(
            bm.all(
                bm.sort(linear_index)
                == bm.arange(self.n_total_nodes, dtype=bm.int64)
            )
        ):
            raise RuntimeError(
                "局部节点未能与结构化网格下标一一对应, 无法建立子结构到全局的映射."
            )

        # (n_nodes, dim): 第 n 行是局部节点 n 在子结构结构化网格中的整数下标.
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
        self._grid_to_cell: Any = linear_index

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

    def _build_interface_bases(self) -> None:
        """解析构造接口自由度上的刚体模态基及其正交补.

        异常:
            RuntimeError: 当接口自由度数不足以容纳全部刚体模态时抛出.

        说明:
            接口自由度按 ``dim * node + k`` 编号, 由此反解每个自由度所属的节点与
            分量. 平动模态在对应分量上取 1; 转动模态取 ``e_a x (x - x_c)``, 其中
            ``x_c`` 为接口节点的形心, ``e_a`` 遍历各坐标轴. 形心的选取只影响基的
            表示, 不影响张成的子空间——平动模态已在基中, 任何常向量平移都被吸收.
        """
        if self.n_b < self.n_rigid:
            raise RuntimeError(
                f"接口自由度数 {self.n_b} 少于刚体模态数 {self.n_rigid}, "
                f"无法构造刚体模态基."
            )

        node = self.mesh.entity('node')
        b_node = self.b_dofs // self.dim
        b_comp = self.b_dofs % self.dim
        # 相对形心的坐标; 形心取接口节点而非全部节点, 与基的构造对象一致.
        offset = node[b_node] - bm.mean(node[self.boundary_nodes], axis=0)[None, :]

        rows = bm.arange(self.n_b, dtype=bm.int64)
        columns = [bm.astype(b_comp == k, bm.float64) for k in range(self.dim)]

        zero = bm.zeros((self.n_b,), dtype=bm.float64)
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
        # 每个接口自由度只取该转动模态在自身分量方向上的分量.
        columns.extend(vec[rows, b_comp] for vec in rotations)

        modes = bm.stack(columns, axis=-1)
        # 完整 QR: Q 的前 n_rigid 列张成刚体子空间, 其余列构成其正交补.
        Q = bm.linalg.qr(modes, mode='complete')[0]
        self._rigid_basis = Q[:, :self.n_rigid]
        self._deformation_basis = Q[:, self.n_rigid:]

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
            leading = tuple(rho.shape[:-n_grid])
            flat = bm.reshape(rho, leading + (self.n_cells,))
            return flat[..., self._grid_to_cell]

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
        return bm.reshape(accumulated, (n_chunk, n_dof, n_dof))


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
