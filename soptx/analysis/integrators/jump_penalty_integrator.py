from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, CoefLike, Threshold
from fealpy.functionspace import FunctionSpace as _FS
from fealpy.utils import process_coef_func
from fealpy.functional import bilinear_integral
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
            
            if R.size>0:
                face2dof[fidx[R], 0:ldof]    = cell2dof[R]
            if L.size>0:
                face2dof[fidx[L], ldof:2*ldof] = cell2dof[L]

        # left  = face2dof[:, :ldof]
        # right = face2dof[:, ldof:]
        # # 用对侧 DOF 填补缺失侧（对应的矩阵块列/行乘以 jump 中的 0，不改变数值）
        # left_missing  = (left  < 0) & (right >= 0)
        # right_missing = (right < 0) & (left  >= 0)
        # left  = bm.where(left_missing,  right, left)
        # right = bm.where(right_missing, left,  right)
        # face2dof = bm.concatenate([left, right], axis=1)

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
        hF = fm

        cell2face = mesh.cell_to_face()               # (NC, TD+1)
        # 单元内局部面的局部取向是否与该全局面的全局取向一致
        cell2facesign = mesh.cell_to_face_sign()      # (NC, TD+1)  True: “右/正”侧
        ldof = space.number_of_local_dofs()

        val_all = bm.zeros((NF, NQ, 2*ldof, GD), dtype=bm.float64) 
        # 内部面构建 [ -φ_R, +φ_L ]; 边界面 [ -φ_R, 0 ] 或 [ 0, +φ_L ]
        for i in range(TD+1):
            # 每个单元的第 i 个局部面对应的全局面号
            fidx = cell2face[:, i]                          # (NC,)
            pos  = cell2facesign[:, i]                      # (NC,)  True/False

            # 哪几个单元第 i 个局部面与全局面的取向一致
            L = bm.nonzero(pos)[0]                          
            # 哪几个单元第 i 个局部面与全局面的取向相反
            R = bm.nonzero(~pos)[0]                         

            # 面上的积分点是定义在 "面参考域" 里，而基函数评估需要 "单元参考域" 的重心坐标
            b = bm.insert(bcs, i, 0, axis=1)                # (NQ, TD+1)

            phi_ref = space.basis(b)                           # (1, NQ, LDOF, GD)
            phi = bm.broadcast_to(phi_ref, (NC, NQ, ldof, GD)) # (NC, NQ, LDOF, GD)

            # 前半部分自由度是 R 侧, 后半部分自由度是 L 侧
            if R.size > 0:
                val_all[fidx[R], :, 0:ldof, :]   =  - phi[R, :, :, :]
            if L.size > 0:
                val_all[fidx[L], :, ldof:,  :]   =  + phi[L, :, :, :]

        val = val_all[index] # (NF, NQ, 2*LDOF, GD)

        boundary_indices_in_val = bm.nonzero(~is_internal_flag)[0]
        if len(boundary_indices_in_val) > 0:
            # 对于边界边, 跳量是迹, 基函数贡献应为 +φ, 直接取绝对值, 可以将 [-φ, 0] 变为 [+φ, 0], 而 [0, +φ] 保持不变
            val[boundary_indices_in_val] = bm.abs(val[boundary_indices_in_val])

        return hF, ws, fm, val

    @variantmethod('vector_jump')
    def assembly(self, space: _FS) -> TensorLike:
        hF, ws, fm, phi_jump = self.fetch_vector_jump(space)
        # hF: (NF, )
        # ws: (NQ, )
        # fm: (NF, )
        # phi_jump: (NF, NQ, 2*LDOF, GD)

        integrand = bm.einsum('q, f, fqid, fqjd -> fij', ws, fm, phi_jump, phi_jump)
        KE = - bm.einsum('f, fij -> fij', 1 / hF, integrand)

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
        hF = fm

        # 获取面的单位法向量
        fn = mesh.face_unit_normal(index=index)  # (len(index), GD)

        cell2face = mesh.cell_to_face()
        cell2facesign = mesh.cell_to_face_sign()
        ldof = space.number_of_local_dofs()

        # 分别存储两侧的基函数值（不带符号）
        w_plus  = bm.zeros((NF, NQ, ldof, GD), dtype=bm.float64)  # w^+
        w_minus = bm.zeros((NF, NQ, ldof, GD), dtype=bm.float64)  # w^-
        
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
        
        # 只取需要的面
        w_plus  = w_plus[index]   # (len(index), NQ, ldof, GD)
        w_minus = w_minus[index]  # (len(index), NQ, ldof, GD)
        
        NF_actual = len(index)
        
        # 构造矩阵跳量
        matrix_jump = bm.zeros((NF_actual, NQ, 2*ldof, GD, GD), dtype=bm.float64)
        
        for f in range(NF_actual):
            nu = fn[f]  # (GD,) 全局法向量
            
            if is_internal_flag[f]:  # 内部面
                # ν^+ = nu, ν^- = -nu
                for q in range(NQ):
                    for d in range(ldof):
                        w_p = w_plus[f, q, d]   # (GD,) w^+
                        w_m = w_minus[f, q, d]  # (GD,) w^-
                        
                        # R侧基函数（w^- 非零）：
                        # [[φ_R]] = 1/2(0⊗ν + ν⊗0 + φ_R⊗(-ν) + (-ν)⊗φ_R)
                        #         = 1/2(-φ_R⊗ν - ν⊗φ_R)
                        M_R = 0.5 * (bm.outer(w_m, -nu) + bm.outer(-nu, w_m))
                        matrix_jump[f, q, d, :, :] = M_R
                        
                        # L侧基函数（w^+ 非零）：
                        # [[φ_L]] = 1/2(φ_L⊗ν + ν⊗φ_L + 0⊗(-ν) + (-ν)⊗0)
                        #         = 1/2(φ_L⊗ν + ν⊗φ_L)
                        M_L = 0.5 * (bm.outer(w_p, nu) + bm.outer(nu, w_p))
                        matrix_jump[f, q, d + ldof, :, :] = M_L
                        
            else:  # 边界面
                # [[w]] = 1/2(w⊗ν + ν⊗w)
                # 判断哪一侧非零
                w = w_plus[f] if bm.any(w_plus[f] != 0) else w_minus[f]
                is_left = bm.any(w_plus[f] != 0)
                
                for q in range(NQ):
                    for d in range(ldof):
                        w_val = w[q, d]  # (GD,)
                        M = 0.5 * (bm.outer(w_val, nu) + bm.outer(nu, w_val))
                        
                        if is_left:
                            matrix_jump[f, q, d + ldof, :, :] = M
                        else:
                            matrix_jump[f, q, d, :, :] = M
        
        return ws, matrix_jump, hF, fm

    @assembly.register('matrix_jump')
    def assembly(self, space: _FS) -> TensorLike:
        hF, ws, fm, phi_jump = self.fetch_matrix_jump(space)
        # hF: (NF, )
        # ws: (NQ, )
        # fm: (NF, )
        # phi_jump: (NF, NQ, 2*LDOF, GD, GD)

        integrand = bm.einsum('q, f, fqikl, fqjkl -> fij', ws, fm, phi_jump, phi_jump)
        KE = - bm.einsum('f, fij -> fij', hF, integrand)

        return KE