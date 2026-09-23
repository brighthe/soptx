import warnings

from itertools import permutations
from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Threshold
from fealpy.functionspace import FunctionSpace as _FS
from fealpy.decorator import variantmethod
from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

from soptx.materials import LinearElasticMaterial

class JumpPenaltyIntegrator(LinearInt, OpInt, FaceInt):

    def __init__(self,
                q: Optional[int]=None,
                threshold: Optional[Threshold]=None,
                method: Optional[str]=None,
                material: Optional[LinearElasticMaterial]=None,
                penalty_scaling: Optional[str]='physical_h',
                density_shear_ratio: Optional[TensorLike]=None,
                density_coupling: str='harmonic',
            ) -> None:
        super().__init__()

        self.q = q
        self.threshold = threshold

        self.material = material

        # 逐单元相对剪切模量 mu(rho)/mu_0, 形状 (NC,); None 表示不做密度定标,
        # 惩罚系数退化为基材常数 (非密度型拓扑优化的原有行为). 详见
        # _face_penalty_scale 的 Notes.
        self.density_shear_ratio = density_shear_ratio
        self.density_coupling = density_coupling

        # 稳定化项缩放律选择. 默认 'physical_h' 为论文式物理量纲缩放
        # (α=μ/L0²·hF, 已实测在 k=1,2 恢复细层收敛, 复现论文表 5.2);
        # 'gamma_hinv' 为旧缩放 (γ/hF, 净效果 O(γ) 无 hF 缩放, 细层发散),
        # 仅保留作回归/对比.
        self.penalty_scaling = penalty_scaling

        self.assembly.set(method)

    def _face_penalty_scale(self, space: _FS) -> Optional[TensorLike]:
        """按相邻单元的相对剪切模量给每个待积分面的惩罚系数定标.

        Parameters
        ----------
        space : FunctionSpace
            位移空间, 提供网格与待积分面的索引.

        Returns
        -------
        TensorLike or None
            形状 ``(NF[index],)`` 的无量纲标度; ``density_shear_ratio`` 为 None
            时返回 None, 此时惩罚系数保持基材常数.

        Notes
        -----
        稳定化项必须与柔度块同步随密度缩放: 同一装配中柔度块为 ``O(1/E(rho))``,
        惩罚块应为 ``O(mu(rho))``. 若惩罚系数固定取基材值, SIMP 空区
        (``E(rho) ~ 1e-9 E_0``) 会被一个高出局部物理量级约 ``1/void_ratio`` 倍的
        惩罚项过约束, 产生虚假应力.

        惩罚系数逐面而相对刚度逐单元, 故需把两侧单元的相对剪切模量合成为面上的
        一个标度, 由 ``density_coupling`` 选取:

        - ``'harmonic'`` (默认): 调和平均 ``2 m_L m_R / (m_L + m_R)``, 异质界面的
          标准加权; 任一侧趋零则整体趋零, 实体-空区界面按弱侧定标.
        - ``'min'``: 取两侧较小者, 与调和平均同阶且更保守.
        - ``'mean'``: 算术平均; 实体-空区界面上仍保留约一半的基材惩罚, 不足以
          消除空区虚假应力, 仅供对照.

        边界面的 ``face_to_cell`` 两列相同, 三种规则都退化为该单元自身的值.
        ``rho ≡ 1`` 时三种规则均给出 1, 本方法逐位退化为原有的基材常数系数,
        故制造解算例的收敛阶不受影响.
        """
        if self.density_shear_ratio is None:
            return None

        mesh = space.mesh
        ratio = bm.asarray(self.density_shear_ratio)
        NC = mesh.number_of_cells()
        if ratio.shape[0] != NC:
            raise ValueError(
                f"density_shear_ratio 长度 {ratio.shape[0]} 与单元数 {NC} 不符"
            )

        index, _ = self.make_index(space)
        face2cell = mesh.face_to_cell()[index]
        m_left = ratio[face2cell[:, 0]]
        m_right = ratio[face2cell[:, 1]]

        if self.density_coupling == 'min':
            return bm.minimum(m_left, m_right)

        if self.density_coupling == 'mean':
            return 0.5 * (m_left + m_right)

        if self.density_coupling != 'harmonic':
            raise ValueError(
                f"不支持的密度耦合规则: {self.density_coupling}, "
                "可选 'harmonic' / 'min' / 'mean'"
            )

        total = m_left + m_right
        safe = bm.where(total > 0, total, bm.ones_like(total))
        return 2.0 * m_left * m_right / safe

    def _cell_to_face_sign(self, mesh):
        """FEALPy 4.0.0 二维下 face 即 edge, 接口名为 cell_to_edge_sign."""
        if mesh.top_dimension() == 2:
            return mesh.cell_to_edge_sign()
        return mesh.cell_to_face_sign()

    def _oriented_cell_basis(self, space: _FS, bcs: TensorLike, i: int) -> TensorLike:
        """把面上的积分点按各单元自身的局部面定向映入单元, 再取基函数值.

        Parameters
        ----------
        space : FunctionSpace
            位移空间.
        bcs : TensorLike
            面参考域上的重心坐标, 形状 ``(NQ, TD)``.
        i : int
            单元的局部面编号.

        Returns
        -------
        TensorLike
            形状 ``(NC, NQ, ldof, GD)`` 的基函数值.

        Notes
        -----
        ``bm.insert(bcs, i, 0, axis=1)`` 把面重心坐标按 **局部顶点索引递增** 的
        次序填入单元重心坐标, 得到的是该单元自己看到的局部面定向. 共享同一条内部
        面的两个单元, 这个局部定向与全局面的顶点次序未必一致 (实测结构化三角网格
        上约 1/3 的内部面不一致), 若两侧都直接套用同一组 ``bcs``, 则 ``w^+`` 与
        ``w^-`` 落在面上互为镜像的物理点, 装配出来的就不是跳量: 全局连续场的
        ``[[v]]`` 不为零, 稳定化项失去相容性. 该不相容误差进入离散平衡方程的右端,
        使 ``div sigma_h`` 掉一阶, 而位移与应力的 L2 阶不受影响, 故不易察觉.

        因此这里逐单元求出把"局部面顶点次序"对齐到"全局面顶点次序"的置换, 按其逆
        置换重排 ``bcs`` 的分量后再插值. 置换只有 ``TD!`` 种 (2D 为 2, 3D 为 6),
        按种类分组批量求值, 开销与原实现同阶.

        位移空间取 ``P_0`` 时单元内为常数, 镜像取点给出同一值, 本方法退化为恒等
        操作; 这也是 Hu-Zhang 次数 ``k = 1`` 未受上述缺陷影响的原因.
        """
        mesh = space.mesh
        TD = mesh.top_dimension()
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        NQ = bcs.shape[0]
        ldof = space.number_of_local_dofs()

        cell = mesh.entity('cell')
        face = mesh.entity('face')
        face_idx = mesh.cell_to_face()[:, i]

        # bm.insert 在局部面上的填充次序: 除 i 以外的局部顶点索引递增
        local_slots = [j for j in range(TD + 1) if j != i]
        local_vertices = cell[:, local_slots]      # (NC, TD)
        face_vertices = face[face_idx]             # (NC, TD)

        phi = bm.zeros((NC, NQ, ldof, GD), dtype=bm.float64)
        matched = bm.zeros(NC, dtype=bm.bool)

        for perm in permutations(range(TD)):
            # perm 满足 local_vertices[:, perm] == face_vertices;
            # 于是全局面第 m 个分量应落到局部槽位 perm[m], 即按逆置换重排 bcs
            sel = bm.all(local_vertices[:, list(perm)] == face_vertices, axis=1)
            sel = sel & (~matched)
            if not bool(bm.any(sel)):
                continue

            inv = [perm.index(j) for j in range(TD)]
            b = bm.insert(bcs[:, inv], i, 0, axis=1)
            phi_perm = bm.broadcast_to(space.basis(b), (NC, NQ, ldof, GD))
            phi = bm.where(sel[:, None, None, None], phi_perm, phi)
            matched = matched | sel

        if not bool(bm.all(matched)):
            raise RuntimeError(
                f"局部面 {i} 上有单元的顶点集合与其全局面不匹配, "
                "无法确定积分点定向; 请检查网格的 cell/face 拓扑一致性"
            )

        return phi

    def make_index(self, space: _FS):
        mesh = space.mesh
        NF = mesh.number_of_faces()
        
        face2cell = mesh.face_to_cell()
        is_internal_all = face2cell[:, 0] != face2cell[:, 1]

        if self.threshold is None:
            index = bm.arange(NF, dtype=bm.int64)
            is_internal_flag = is_internal_all
            return index, is_internal_flag
        
        elif isinstance(self.threshold, TensorLike): 
            index = self.threshold
            is_internal_flag = is_internal_all[index]
            return index, is_internal_flag
        
        else:
            raise ValueError(f"Unsupported threshold type: {self.threshold}")
    
    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        """待积分面与其相邻单元自由度之间的映射关系"""
        index, _ = self.make_index(space)
        mesh = space.mesh
        TD = mesh.top_dimension()
        NF = mesh.number_of_faces()
        ldof = space.number_of_local_dofs()

        cell2face = mesh.cell_to_face()
        cell2facesign = self._cell_to_face_sign(mesh)
        cell2dof = space.cell_to_dof()

        face2dof = bm.zeros((NF, 2*ldof), dtype=bm.int64)

        for i in range(TD+1):
            fidx = cell2face[:, i]
            pos  = cell2facesign[:, i]
            L = bm.nonzero(pos)[0]
            R = bm.nonzero(~pos)[0]

            if R.size > 0:
                face2dof[fidx[R], 0:ldof] = cell2dof[R]

            if L.size > 0:
                face2dof[fidx[L], ldof:2*ldof] = cell2dof[L]

        return face2dof[index]
    
    
    ########################################################################################
    # 变体方法
    ########################################################################################

    @enable_cache
    def fetch_matrix_jump(self, space: _FS):
        """计算矩阵跳量"""
        mesh = getattr(space, 'mesh', None)
        index, is_internal_flag = self.make_index(space)
        
        q = space.p + 3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        NC = mesh.number_of_cells()
        NF = mesh.number_of_faces()
        TD = mesh.top_dimension()
        GD = mesh.geo_dimension()
        NQ = len(ws)

        fm = mesh.entity_measure('face', index=index)
        if GD == 2:
            hF = fm  # 2D: 边长
        elif GD == 3:
            hF = bm.sqrt(fm)  # 3D: sqrt(面积) ≈ 面的特征尺度
            
        # 获取面的单位法向量
        fn = mesh.face_unit_normal(index=index)  # (NF(index), GD)

        cell2face = mesh.cell_to_face()
        # 单元内局部面的局部取向是否与该全局面的全局取向一致
        cell2facesign = self._cell_to_face_sign(mesh)      # (NC, TD+1)  True: "右/正" 侧; False: "左/负" 侧
        ldof = space.number_of_local_dofs()

        # 内部面 F 上, 基函数 w^+ 来自 L 侧单元, w^- 来自 R 侧单元
        w_plus  = bm.zeros((NF, NQ, ldof, GD), dtype=bm.float64)  
        w_minus = bm.zeros((NF, NQ, ldof, GD), dtype=bm.float64)  
        
        for i in range(TD+1):
            fidx = cell2face[:, i]
            pos  = cell2facesign[:, i]
            
            L = bm.nonzero(pos)[0]   # 左侧：pos=True，这是 w^+
            R = bm.nonzero(~pos)[0]  # 右侧：pos=False，这是 w^-
            
            # 按各单元自身的局部面定向映射积分点, 使面两侧取到同一物理点,
            # 否则装配出的不是跳量, 见 _oriented_cell_basis 的 Notes
            phi = self._oriented_cell_basis(space, bcs, i)
            
            # 存储原始基函数值（不带符号）
            if L.size > 0:
                w_plus[fidx[L]]  = phi[L]   # w^+
            if R.size > 0:
                w_minus[fidx[R]] = phi[R]   # w^-
        
        w_plus  = w_plus[index]   # (NF[index], NQ, ldof, GD)
        w_minus = w_minus[index]  # (NF[index], NQ, ldof, GD)
        
        # 构造矩阵跳量
        NF_local = len(index)
        matrix_jump = bm.zeros((NF_local, NQ, 2*ldof, GD, GD), dtype=bm.float64)
        
        # ============ 内部面 ============
        internal_idx = bm.nonzero(is_internal_flag)[0]
        if len(internal_idx) > 0:
            w_p = w_plus[internal_idx]
            w_m = w_minus[internal_idx]
            nu = fn[internal_idx]
            
            # R 侧
            M_R = 0.5 * (bm.einsum('fqdi, fj -> fqdij', w_m, -nu) + bm.einsum('fi, fqdj -> fqdij', -nu, w_m))
            matrix_jump[internal_idx, :, :ldof, :, :] = M_R
            # L 侧
            M_L = 0.5 * (bm.einsum('fqdi, fj -> fqdij', w_p, nu) + bm.einsum('fi, fqdj -> fqdij', nu, w_p))
            matrix_jump[internal_idx, :, ldof:, :, :] = M_L
        
        # ============ 边界面 ============
        boundary_idx = bm.nonzero(~is_internal_flag)[0]
        if len(boundary_idx) > 0:
            w_p = w_plus[boundary_idx]
            w_m = w_minus[boundary_idx]
            nu = fn[boundary_idx]
            
            # 判断并选择非零侧
            is_left = bm.any(w_p != 0, axis=(1, 2, 3))
            w = bm.where(is_left[:, None, None, None], w_p, w_m)

            # 计算矩阵跳量
            M = 0.5 * (bm.einsum('fqdi, fj -> fqdij', w, nu) + bm.einsum('fi, fqdj -> fqdij', nu, w))
            
            # 分别存储 L 侧和 R 侧
            left_idx = boundary_idx[is_left]
            right_idx = boundary_idx[~is_left]
            
            if len(left_idx) > 0:
                matrix_jump[left_idx, :, ldof:, :, :] = M[is_left]
            if len(right_idx) > 0:
                matrix_jump[right_idx, :, :ldof, :, :] = M[~is_left]
                
        return ws, matrix_jump, hF, fm

    @variantmethod('matrix_jump')
    def assembly(self, space: _FS) -> TensorLike:
        ws, matrix_jump, hF, fm = self.fetch_matrix_jump(space)
        integrand = bm.einsum('q, f, fqikl, fqjkl -> fij', ws, fm, matrix_jump, matrix_jump)
        
        # 构建缩放系数
        # k=1 用 E，k>=2 用 mu，反映不同次数对稳定化强度的不同需求
        # 应力空间的次数
        p = space.p + 1
        mesh = space.mesh
        node = mesh.entity('node')
        bbox_max = bm.max(node, axis=0)  
        bbox_min = bm.min(node, axis=0)  
        L0 = bm.max(bbox_max - bbox_min) # mm

        # ==================== 材料归一化检查与物理量纲提示 ====================
        E = self.material.youngs_modulus # MPa
        mu = self.material.shear_modulus # MPa
        
        if abs(E - 1.0) > 1e-6:
            msg = (
                f"\n[SOPTX 警告] 当前材料杨氏模量 E = {E} MPa。未进行归一化！\n"
                "在带有跳量稳定化项的胡张混合有限元中，柔度矩阵块量级为 O(1/E)，\n"
                "而稳定化惩罚块量级为 O(E)。直接代入真实高模量(如 7e4)将导致离散鞍点\n"
                "系统的条件数高达 O(E^2)，极易引发直接求解器(如 MUMPS)内存溢出(Error -9)。\n"
                "强烈建议：在材料定义中设置 E = 1.0 且泊松比 nu 保持物理真实值。\n"
                "【无量纲化物理映射规则】：\n"
                " 1. [应力 Sigma]: 混合元求解的表观应力即为真实物理应力，直接可用于屈服评估！\n"
                " 2. [位移 U]: 真实物理位移 U_real = 求解位移 U_calc / 真实的杨氏模量。\n"
                " 3. [柔顺度 C]: 真实柔顺度 C_real = 求解柔顺度 C_calc / 真实的杨氏模量。"
            )
            warnings.warn(msg, UserWarning)
        # ======================================================================
        
        # 跳量惩罚缩放律二选一:
        #
        # 1) 'physical_h' (默认, 论文式物理量纲缩放):
        #    c = Σ_F (μ/L0²)·hF·∫[[u]]:[[v]]ds, hF 幂次为 +1, 系数 α=μ/L0².
        #    integrand 已含面测度 fm(=hF, 2D), 故此处再乘 hF 一次方对齐论文.
        #    实测 (sinusoidal 混合边界制造解, k=1,2, nx=2..32) 恢复细层收敛:
        #    k=1: u→1 阶, σ→1.53 阶 (超收敛), H(div)→1 阶;
        #    k=2: u→2 阶, σ→2.02 阶, H(div)→1 阶 (降阶, 与论文表 5.2 逐格一致).
        #
        # 2) 'gamma_hinv' (旧缩放, 仅回归/对比): γ 取材料模量的小比例系数.
        #    由于 integrand 含 fm, 净效果是 O(γ) 常数系数、无 hF 缩放 ——
        #    与论文的 hF¹ 缩放律不同, 细层 (h -> 0) 位移/应力阶塌陷、div 发散.
        #    调参历史: 原 E/L0²·hF 与裸模量 γ/hF 都过大, 惩罚块量级远超柔度块,
        #    会压坏 div; γ ~ 1e-2·模量时粗层阶正常 (σ ~ 2.4, div ~ 1.6),
        #    但该缩放律本身仍不足以支撑细层收敛, 已由 'physical_h' 取代.
        if self.penalty_scaling == 'gamma_hinv':
            if p == 1:
                gamma = 0.01 * E
            else:
                gamma = 0.01 * mu

            coefficient = gamma * hF ** -1
        else:
            alpha = mu / L0 ** 2
            coefficient = alpha * hF

        # 两条缩放律的系数都取自基材; 密度型拓扑优化下再按两侧单元的相对剪切
        # 模量逐面定标, 使惩罚块与柔度块同步随密度缩放 (见 _face_penalty_scale).
        scale = self._face_penalty_scale(space)
        if scale is not None:
            coefficient = coefficient * scale

        KE = bm.einsum('f, fij -> fij', coefficient, integrand)

        return KE

    @enable_cache
    def fetch_vector_jump(self, space: _FS):
        """计算向量跳量"""
        mesh = getattr(space, 'mesh', None)
        index, is_internal_flag = self.make_index(space)
        
        q = space.p + 3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        NC = mesh.number_of_cells()
        NF = mesh.number_of_faces()
        TD = mesh.top_dimension()
        GD = mesh.geo_dimension()
        NQ = len(ws)

        fm = mesh.entity_measure('face', index=index) 
        if GD == 2:
            hF = fm  
        elif GD == 3:
            hF = bm.sqrt(fm)  
        else:
            raise ValueError(f"Unsupported dimension: {GD}")

        cell2face = mesh.cell_to_face()               # (NC, TD+1)
        # 单元内局部面的局部取向是否与该全局面的全局取向一致
        cell2facesign = self._cell_to_face_sign(mesh)      # (NC, TD+1)  True: "右/正" 侧; False: "左/负" 侧
        ldof = space.number_of_local_dofs()

        val_all = bm.zeros((NF, NQ, 2*ldof, GD), dtype=bm.float64) 
        # 内部面构建 [ -φ_R, +φ_L ]
        for i in range(TD+1):
            # 每个单元的第 i 个局部面对应的全局面号
            fidx = cell2face[:, i]                          # (NC,)
            pos  = cell2facesign[:, i]                      # (NC,)  True/False

            # 根据 cell2facesign 识别左侧单元(L, 对应 w^+)和右侧单元(R, 对应 w^-)
            L = bm.nonzero(pos)[0]                          
            R = bm.nonzero(~pos)[0]                         

            # 面上的积分点定义在 "面参考域", 基函数评估需要 "单元参考域" 的重心坐标;
            # 该映射按各单元自身的局部面定向进行, 见 _oriented_cell_basis 的 Notes
            phi = self._oriented_cell_basis(space, bcs, i)     # (NC, NQ, LDOF, GD)

            # [w] = w^+ - w^-，构建算子 [ -φ_R, +φ_L ]
            if R.size > 0:
                val_all[fidx[R], :, 0:ldof, :]   =  - phi[R, :, :, :]
            if L.size > 0:
                val_all[fidx[L], :, ldof:,  :]   =  + phi[L, :, :, :]

        val = val_all[index] # (NF[index], NQ, 2*LDOF, GD)

        boundary_indices_in_val = bm.nonzero(~is_internal_flag)[0]
        # 对于边界面, 跳量是迹本身即 [w] = w, 边界面值为 [-φ, 0] 或 [0, +φ]
        if len(boundary_indices_in_val) > 0:
            val[boundary_indices_in_val] = bm.abs(val[boundary_indices_in_val])

        return ws, val, hF, fm

    @assembly.register('vector_jump')
    def assembly(self, space: _FS) -> TensorLike:
        ws, vector_jump, hF, fm = self.fetch_vector_jump(space)
        # hF: (NF, )
        # ws: (NQ, )
        # fm: (NF, )
        # vector_jump: (NF, NQ, 2*LDOF, GD)

        integrand = bm.einsum('q, f, fqid, fqjd -> fij', ws, fm, vector_jump, vector_jump)
        KE = bm.einsum('f, fij -> fij', 1 / hF, integrand)

        return KE
    
