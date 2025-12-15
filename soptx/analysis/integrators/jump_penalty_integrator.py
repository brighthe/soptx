from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Threshold
from fealpy.functionspace import FunctionSpace as _FS
from fealpy.decorator import variantmethod
from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

class JumpPenaltyIntegrator(LinearInt, OpInt, FaceInt):

    def __init__(self, 
                q: Optional[int]=None,
                threshold: Optional[Threshold]=None,
                method: Optional[str]=None
            ) -> None:
        super().__init__()

        self.q = q
        self.threshold = threshold

        self.assembly.set(method)

    def make_index(self, space: _FS):
        mesh = space.mesh
        NF = mesh.number_of_faces()
        
        face2cell = mesh.face_to_cell()
        is_internal_all = face2cell[:, 0] != face2cell[:, 1]

        if self.threshold is None:
            index = bm.arange(NF, dtype=bm.int64)
            is_internal_flag = is_internal_all
            return index, is_internal_flag
        
        elif self.threshold == 'internal':
            index = bm.nonzero(is_internal_all)[0]
            is_internal_flag = bm.ones(len(index), dtype=bm.bool)
            return index, is_internal_flag

        elif callable(self.threshold):
            bc = mesh.entity_barycenter('face')
            index_bool = self.threshold(bc)
            index = bm.nonzero(index_bool)[0]
            is_internal_flag = is_internal_all[index]
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
        cell2facesign = mesh.cell_to_face_sign()
        cell2dof = space.cell_to_dof()

        face2dof = bm.full((NF, 2*ldof), -1, dtype=bm.int64)

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
            hF = fm  # 2D: 边长
        elif GD == 3:
            hF = bm.sqrt(fm)  # 3D: sqrt(面积) ≈ 面的特征尺度
        else:
            raise ValueError(f"Unsupported dimension: {GD}")

        cell2face = mesh.cell_to_face()               # (NC, TD+1)
        # 单元内局部面的局部取向是否与该全局面的全局取向一致
        cell2facesign = mesh.cell_to_face_sign()      # (NC, TD+1)  True: "右/正" 侧; False: "左/负" 侧
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

            # 面上的积分点是定义在 "面参考域" 里，而基函数评估需要 "单元参考域" 的重心坐标
            b = bm.insert(bcs, i, 0, axis=1)                # (NQ, TD+1)

            phi_ref = space.basis(b)                           # (1, NQ, LDOF, GD)
            phi = bm.broadcast_to(phi_ref, (NC, NQ, ldof, GD)) # (NC, NQ, LDOF, GD)

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

    @variantmethod('vector_jump')
    def assembly(self, space: _FS) -> TensorLike:
        ws, vector_jump, hF, fm = self.fetch_vector_jump(space)
        # hF: (NF, )
        # ws: (NQ, )
        # fm: (NF, )
        # vector_jump: (NF, NQ, 2*LDOF, GD)

        integrand = bm.einsum('q, f, fqid, fqjd -> fij', ws, fm, vector_jump, vector_jump)
        # KE = - bm.einsum('f, fij -> fij', 1 / hF, integrand)
        KE = - bm.einsum('f, fij -> fij', hF, integrand)

        return KE
    
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
        cell2facesign = mesh.cell_to_face_sign()      # (NC, TD+1)  True: "右/正" 侧; False: "左/负" 侧
        ldof = space.number_of_local_dofs()

        # 内部面 F 上, 基函数 w^+ 来自 L 侧单元, w^- 来自 R 侧单元
        w_plus  = bm.zeros((NF, NQ, ldof, GD), dtype=bm.float64)  
        w_minus = bm.zeros((NF, NQ, ldof, GD), dtype=bm.float64)  
        
        for i in range(TD+1):
            fidx = cell2face[:, i]
            pos  = cell2facesign[:, i]
            
            L = bm.nonzero(pos)[0]   # 左侧：pos=True，这是 w^+
            R = bm.nonzero(~pos)[0]  # 右侧：pos=False，这是 w^-
            
            b = bm.insert(bcs, i, 0, axis=1)
            phi_ref = space.basis(b)
            phi = bm.broadcast_to(phi_ref, (NC, NQ, ldof, GD))
            
            # 存储原始基函数值（不带符号）
            if L.size > 0:
                w_plus[fidx[L]]  = phi[L]   # w^+
            if R.size > 0:
                w_minus[fidx[R]] = phi[R]   # w^-
        
        w_plus  = w_plus[index]   # (NF[index], NQ, ldof, GD)
        w_minus = w_minus[index]  # (NF[index], NQ, ldof, GD)
        
        # 构造矩阵跳量
        matrix_jump = bm.zeros((NF, NQ, 2*ldof, GD, GD), dtype=bm.float64)
        
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

    @assembly.register('matrix_jump')
    def assembly(self, space: _FS) -> TensorLike:
        ws, matrix_jump, hF, fm = self.fetch_matrix_jump(space)
        # hF: (NF, )
        # ws: (NQ, )
        # fm: (NF, )
        # matrix_jump: (NF, NQ, 2*LDOF, GD, GD)
        integrand = bm.einsum('q, f, fqikl, fqjkl -> fij', ws, fm, matrix_jump, matrix_jump)
        KE = - bm.einsum('f, fij -> fij', hF, integrand)

        return KE