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
        face2dof = bm.zeros((NF, 2*ldof),dtype=bm.int32)
        cell2dof = space.cell_to_dof()

        for i in range(TD+1):

            lidx, = bm.nonzero(cell2facesign[:, i]) 
            ridx, = bm.nonzero(~cell2facesign[:, i]) 
            fidx = cell2face[:, i]   
            face2dof[fidx[lidx], 0:ldof] = cell2dof[lidx] 
            face2dof[fidx[ridx], ldof:2*ldof] = cell2dof[ridx]

        temp = face2dof[index]

        return temp

    @enable_cache
    def fetch(self, space: _FS):

        q = self.q
        index = self.make_index(space)
        mesh = getattr(space, 'mesh', None)
        GD = mesh.geo_dimension()
        p = getattr(space, 'p', None)

        cm = mesh.entity_measure('face', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        NF = mesh.number_of_faces()
        TD = mesh.top_dimension()
        NQ = len(ws)
        NC = mesh.number_of_cells()

        cell2face = mesh.cell_to_face()
        cell2facesign = mesh.cell_to_face_sign()

        ldof = space.number_of_local_dofs() # 单元上的所有的自由度的个数
        face2dof = bm.zeros((NF, 2*ldof),dtype=bm.int32)
        cell2dof = space.cell_to_dof()
        val = bm.zeros((NF, NQ, 2*ldof, GD),dtype=bm.float64) 
        n = mesh.face_unit_normal()

        for i in range(TD+1): # 循环单元每个面

            # from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
            # s_space = LagrangeFESpace(mesh=mesh, p=1, ctype='C')
            # t_space = TensorFunctionSpace(s_space, shape=(2, -1))

            lidx, = bm.nonzero(cell2facesign[:, i]) # 单元是全局面的左边单元
            ridx, = bm.nonzero(~cell2facesign[:, i]) # 单元是全局面的右边单元

            fidx = cell2face[:, i] # 第 i 个面的全局编号)
            face2dof[fidx[lidx], 0:ldof] = cell2dof[lidx] 
            face2dof[fidx[ridx], ldof:2*ldof] = cell2dof[ridx]
            b = bm.insert(bcs, i, 0, axis=1)
            cval = space.basis(b)
            # cval_test0 = s_space.basis(b)
            # cval_test = t_space.basis(b)
            cval = bm.broadcast_to(cval, (NC, cval.shape[1], cval.shape[2], cval.shape[3]))
            # cval = bm.einsum('cql->cql', space.basis(b))
            val[fidx[ridx, None], :, 0:ldof, :] =+ cval[ridx[:, None], ...]
            val[fidx[lidx, None], :, ldof:2*ldof, :] =- cval[lidx[:, None], ...]

        val = val[index]

        return bcs, ws, val, cm, index

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)

        KE = bm.einsum('q, f, fqid, fqjd -> fij', ws, cm, phi, phi)

        return KE

        # # val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='face', index=index)

        # return bilinear_integral(phi, phi, ws, cm*cm, val, batched=self.batched)



# import matplotlib.pyplot as plt

# from fealpy.backend import backend_manager as bm
# from fealpy.mesh import TriangleMesh
# from fealpy.functionspace.huzhang_fe_space import HuZhangFESpace 
# from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

# from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
# #from fealpy.fem.huzhang_displacement_integrator import HuZhangDisplacementIntegrator
# from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
# from fealpy.fem import VectorSourceIntegrator

# from fealpy.decorator import cartesian

# from fealpy.fem import BilinearForm,ScalarMassIntegrator
# from fealpy.fem import LinearForm, ScalarSourceIntegrator,BoundaryFaceSourceIntegrator
# from fealpy.fem import DivIntegrator
# from fealpy.fem import BlockForm,LinearBlockForm

# N = 2
# mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)
# space = LagrangeFESpace(mesh, p=2, ctype='D')
# bform = BilinearForm(space)
# bform.add_integrator(JumpPenaltyIntergrator(coef=1))
# M = bform.assembly()
# print(M.to_dense())