from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, CoefLike, Threshold
from fealpy.functionspace import FunctionSpace as _FS
from fealpy.utils import process_coef_func
from fealpy.functional import bilinear_integral
from fealpy.decorator import variantmethod
from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

class JumpPenaltyIntergrator2(LinearInt, OpInt, FaceInt):

    def __init__(self, q: Optional[int]=None, *,
                 threshold: Optional[Threshold]=None,
                 batched: bool=False):
        super().__init__()
        # self.coef = coef
        self.q = q
        self.threshold = threshold
        self.batched = batched

    def make_index(self, space: _FS):
        threshold = self.threshold

        if isinstance(threshold, TensorLike):
            index = threshold
        else:
            mesh = space.mesh
            face2cell = mesh.face_to_cell()
            index = face2cell[:, 0] != face2cell[:, 1]
            if callable(threshold):
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)]
        NF = mesh.number_of_faces()
        index = bm.arange(NF)
        return index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:

        index = self.make_index(space)
        mesh = space.mesh
        TD = mesh.top_dimension()
        NF = mesh.number_of_faces()

        cell2face = mesh.cell_to_face()
        cell2facesign = mesh.cell_to_face_sign()
        ldof = space.number_of_local_dofs()
        face2dof = bm.full((NF, 2*ldof), -1, dtype=bm.int64)
        cell2dof = space.cell_to_dof()

        for i in range(TD+1):

            lidx, = bm.nonzero(cell2facesign[:, i]) 
            ridx, = bm.nonzero(~cell2facesign[:, i]) 
            fidx = cell2face[:, i]   
            face2dof[fidx[lidx], 0:ldof] = cell2dof[lidx] 
            face2dof[fidx[ridx], ldof:2*ldof] = cell2dof[ridx]

        # 处理边界面：把未赋值的一侧复制为已有的一侧
        bc_faces = mesh.boundary_face_index()   # 你已有的边界索引
        for f in bc_faces:
            left = face2dof[f, 0:ldof]
            right = face2dof[f, ldof:2*ldof]
            # 若左侧未赋值（全为 -1），复制右侧；反之亦然
            if bm.all(left == -1):
                face2dof[f, 0:ldof] = right
            if bm.all(right == -1):
                face2dof[f, ldof:2*ldof] = left

        # print(face2dof)

        return face2dof[index]

    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        index = self.make_index(space)
        mesh = getattr(space, 'mesh', None)
        p = getattr(space, 'p', None)
        cm = mesh.entity_measure('face', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NF = mesh.number_of_faces()
        TD = mesh.top_dimension()
        GD = mesh.geo_dimension()
        NQ = len(ws)
        NC = mesh.number_of_cells()
        cell2face = mesh.cell_to_face()
        cell2facesign = mesh.cell_to_face_sign()
        ldof = space.number_of_local_dofs() # 单元上的所有的自由度的个数
        face2dof = bm.zeros((NF, 2*ldof),dtype=bm.int32)
        cell2dof = space.cell_to_dof()
        val = bm.zeros((NF, NQ, 2*ldof,GD),dtype=bm.float64) 
        n = mesh.face_unit_normal()
        for i in range(TD+1): # 循环单元每个面
            lidx, = bm.nonzero(cell2facesign[:, i]) # 单元是全局面的左边单元
            ridx, = bm.nonzero(~cell2facesign[:, i]) # 单元是全局面的右边单元
            fidx = cell2face[:, i] # 第 i 个面的全局编号)
            face2dof[fidx[lidx], 0:ldof] = cell2dof[lidx] 
            face2dof[fidx[ridx], ldof:2*ldof] = cell2dof[ridx]
            b = bm.insert(bcs, i, 0, axis=1)
            cval = space.basis(b)
            cval = bm.broadcast_to(cval, (NF, cval.shape[1], cval.shape[2], GD))
            # cval = bm.einsum('cql->cql', space.basis(b))
            val[fidx[ridx, None],:, 0:ldof,:] = +cval[ridx[:, None],:, :,:]
            val[fidx[lidx, None],:, ldof:2*ldof,:] = -cval[lidx[:, None],:, :,:]

        bc_faces = mesh.boundary_face_index()
        val[bc_faces] = -val[bc_faces]
        val = val[index]
        val0 = val[0, ...]

        return bcs, ws, val, cm, index

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        # coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
        # val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='face', index=index)
        #bilinear_integral(phi, phi, ws, cm, val, batched=self.batched)

        return bm.einsum('q, fqid, fqjd->fij', ws, phi, phi)