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
                threshold: Optional[Threshold]=None
            ) -> None:
        super().__init__()

        self.q = q
        self.threshold = threshold

    def make_index(self, space: _FS):
        """生成所有内部面的全局整数索引"""
        mesh = space.mesh
        NF = mesh.number_of_faces()
        threshold = self.threshold

        if threshold is None:
            index = bm.arange(NF, dtype=bm.int64)
        
        elif isinstance(threshold, TensorLike):
            index = threshold
        
        elif callable(threshold):
            mesh = space.mesh
            face2cell = mesh.face_to_cell()
            index_bool = face2cell[:, 0] != face2cell[:, 1]
            if callable(threshold):
                bc = mesh.entity_barycenter('face', index=index_bool)
                index_bool = index_bool[threshold(bc)]
            index = bm.nonzero(index_bool)[0]

        return index
    
    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        index = self.make_index(space) # (NFs, )
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

        left  = face2dof[:, :ldof]
        right = face2dof[:, ldof:]
        # 用对侧 DOF 填补缺失侧（对应的矩阵块列/行乘以 jump 中的 0，不改变数值）
        left_missing  = (left  < 0) & (right >= 0)
        right_missing = (right < 0) & (left  >= 0)
        left  = bm.where(left_missing,  right, left)
        right = bm.where(right_missing, left,  right)
        face2dof = bm.concatenate([left, right], axis=1)

        return face2dof[index]

    @enable_cache
    def fetch(self, space: _FS):
        mesh = getattr(space, 'mesh', None)
        index = self.make_index(space)
        
        q = space.p + 3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        NC = mesh.number_of_cells()
        NF = mesh.number_of_faces()
        TD = mesh.top_dimension()
        GD = mesh.geo_dimension()
        NQ = len(ws)

        fm = mesh.entity_measure('face', index=index)  # (NFs, )

        cell2face = mesh.cell_to_face()               # (NC, TD+1)
        # 单元内局部面的局部取向是否与该全局面的全局取向一致
        cell2facesign = mesh.cell_to_face_sign()      # (NC, TD+1)  True: “右/正”侧
        ldof = space.number_of_local_dofs()

        val_all = bm.zeros((NF, NQ, 2*ldof, GD), dtype=bm.float64) 

        for i in range(TD+1):
            # 每个单元的第 i 个局部面对应的全局面号
            fidx = cell2face[:, i]                          # (NC,)
            pos  = cell2facesign[:, i]                      # (NC,)  True/False

            # 哪几个单元第 i 个局部面与全局面的取向一致
            L = bm.nonzero(pos)[0]                          
            # 哪几个单元第 i 个局部面与全局面的取向相反
            R = bm.nonzero(~pos)[0]                         

            # 面上的积分点是定义在“面参考域”里，而基函数评估需要“单元参考域”的重心坐标
            b = bm.insert(bcs, i, 0, axis=1)                # (NQ, TD+1)

            phi_ref = space.basis(b)                        # (1, NQ, ldof, GD)
            phi = bm.broadcast_to(phi_ref, (NC, NQ, ldof, GD))

            # 前半部分自由度是 R 侧, 后半部分自由度是 L 侧
            if R.size > 0:
                val_all[fidx[R], :, 0:ldof, :]   =  - phi[R, :, :, :]
            if L.size > 0:
                val_all[fidx[L], :, ldof:,  :]   =  + phi[L, :, :, :]

        val = val_all[index] # (NFs, NQ, 2*ldof, GD)

        return ws, val, fm

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:

        ws, phi_jump, fm = self.fetch(space)
        hF = fm

        KE = -bm.einsum('q, f, fqid, fqjd -> fij', ws, fm / hF, phi_jump, phi_jump)

        return KE