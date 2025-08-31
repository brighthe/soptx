from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Tuple
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


def map_bcs_to_sub_elements(bcs_e: Tuple[TensorLike, TensorLike], n_sub: int):
    """将位移单元的积分点的重心坐标映射成各个子密度单元的积分点的重心坐标
    
    Parameters:
    -----------
    bcs_e : 位移单元积分点的重心坐标, 结构为 ( (NQ, GD), (NQ, GD) )
    n_sub : 子密度单元的总数
    子密度的顺序 (先行后列)
    | 3 | 2 | 
    | 0 | 1 |

    Returns:
    --------
    bcs_g : 子密度单元积分点的重心坐标, 结构为 ( (n_sub, NQ_x, GD), (n_sub, NQ_y, GD) )
    """
    if not isinstance(bcs_e, tuple):
        raise TypeError(
            "输入参数 'bcs_e' 必须是一个元组 (tuple),"
            "此函数仅适用于张量积网格."
        )

    bcs_xi, bcs_eta = bcs_e
    GD = bcs_xi.shape[1]

    sqrt_n_sub = bm.sqrt(n_sub)
    if sqrt_n_sub != int(sqrt_n_sub):
        raise ValueError("子密度单元个数 'n_sub' 必须是一个完全平方数")
    
    n_sub_x = int(sqrt_n_sub)
    n_sub_y = int(sqrt_n_sub)

    p_1d = bm.unique(bcs_xi[:, 0])
    NQ_x = len(p_1d)
    NQ_y = len(p_1d)

    bcs_g_xi = bm.zeros((n_sub, NQ_x, GD))
    bcs_g_eta = bm.zeros((n_sub, NQ_y, GD))

    for i in range(n_sub_y): # 遍历行 
        for j in range(n_sub_x): # 遍历列
            
            # 先列后行
            # sub_element_idx = j * n_sub_y + i
            # 先行后列
            sub_element_idx = i * n_sub_x + j

            # 计算当前密度单元的区间范围 (假设父单元为 [0, 1] x [0, 1])
            xi_start, xi_end = j / n_sub_x, (j + 1) / n_sub_x
            eta_start, eta_end = i / n_sub_y, (i + 1) / n_sub_y
            
            # 线性映射
            mapped_xi = xi_start + p_1d * (xi_end - xi_start)
            mapped_eta = eta_start + p_1d * (eta_end - eta_start)
            
            bcs_g_xi[sub_element_idx, :, 0] = mapped_xi[::-1] # 降序
            bcs_g_xi[sub_element_idx, :, 1] = mapped_xi       # 升序

            bcs_g_eta[sub_element_idx, :, 0] = mapped_eta[::-1] # 降序
            bcs_g_eta[sub_element_idx, :, 1] = mapped_eta       # 升序
            
    bcs_g = (bcs_g_xi, bcs_g_eta)

    return bcs_g

