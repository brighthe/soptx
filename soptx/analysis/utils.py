from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import TensorFunctionSpace, Function
from fealpy.mesh import TriangleMesh

from ..analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from ..model.pde_base import PDEBase 

def _get_displacement_dof_component(uh: Function, space: TensorFunctionSpace) -> TensorLike:
    """获取位移自由度分量形式"""
    shape = space.shape
    scalar_space = space.scalar_space
    mesh = space.mesh
    gdof = scalar_space.number_of_global_dofs()
    GD = mesh.geo_dimension()

    if shape[1] == -1: # dof_priority
        uh_reshaped = uh.reshape(GD, gdof)  
        return uh_reshaped.T
    
    elif shape[1] == GD: # gd_priority
        return uh.reshape(GD, gdof)

def _scalar_disp_to_tensor_disp(dof_priority: bool,
                            uh: Function, 
                            uh0: Function, uh1: Function, uh2: Function = None
                        ) -> Function:
    """将标量位移转换为张量位移"""
    if uh2 is None:
        if dof_priority:
            uh[:] = bm.stack((uh0, uh1), axis=-1).T.flatten()
        else:
            uh[:] = bm.stack((uh0, uh1), axis=-1).flatten()
    else:
        if dof_priority:
            uh[:] = bm.stack((uh0, uh1, uh2), axis=-1).T.flatten()
        else:
            uh[:] = bm.stack((uh0, uh1, uh2), axis=-1).flatten()

    return uh

def project_solution_to_finer_mesh(pde: PDEBase,
                                nx: int, ny: int,
                                uh: Function,
                                lfa: LagrangeFEMAnalyzer,
                                source_refinement_level,
                                target_mesh: TriangleMesh
                            ) -> Function:
    # 重新生成源网格
    mesh = pde.init_mesh(nx=nx, ny=ny)
    for _ in range(source_refinement_level):
        mesh.bisect(isMarkedCell=None)

    # 只要当前网格的单元数还少于目标网格，就继续加密和插值
    while mesh.number_of_cells() < target_mesh.number_of_cells():
        sspace = lfa.get_scalar_space_from_mesh(mesh=mesh)
        scell2dof = sspace.cell_to_dof()
        tspace = lfa.get_tensor_space_from_scalar_space(scalar_space=sspace)
        uh_dof_component = _get_displacement_dof_component(uh=uh, space=tspace)

        uh0c2f = uh_dof_component[..., 0][scell2dof]
        uh1c2f = uh_dof_component[..., 1][scell2dof]
        data = {'uh0c2f': uh0c2f, 'uh1c2f': uh1c2f}

        options = mesh.bisect_options(data=data, disp=False)
        mesh.bisect(isMarkedCell=None, options=options)

        # 在新的、更细的网格上构建插值解
        sspace = lfa.get_scalar_space_from_mesh(mesh=mesh)
        tspace = lfa.get_tensor_space_from_scalar_space(scalar_space=sspace)
        uh_new = tspace.function()
        uh0_new = sspace.function()
        uh1_new = sspace.function()
        scell2dof_new = sspace.cell_to_dof()
        uh0_new[scell2dof_new.reshape(-1)] = options['data']['uh0c2f'].reshape(-1)
        uh1_new[scell2dof_new.reshape(-1)] = options['data']['uh1c2f'].reshape(-1)

        uh = _scalar_disp_to_tensor_disp(dof_priority=True, 
                                            uh=uh_new, uh0=uh0_new, uh1=uh1_new)
    
    return uh