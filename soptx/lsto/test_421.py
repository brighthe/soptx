import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

def toplsm(domain_width, domain_height, ele_num_per_row, ele_num_per_col, 
           lv, lcur, fea_interval, plot_interval, total_it_num):
    """
    Python implementation of TOPLSM for mean compliance optimization of structures in 2D
    with the classical level set method.
    
    Based on the 199-line MATLAB program by Michael Yu Wang, Shikui CHEN and Qi XIA (2004-2005)
    
    Parameters:
    -----------
    domain_width : float
        The width of the design domain
    domain_height : float
        The height of the design domain
    ele_num_per_row : int
        Number of finite elements in horizontal direction
    ele_num_per_col : int
        Number of finite elements in vertical direction
    lv : float
        Lagrange multiplier for volume constraint
    lcur : float
        Lagrange multiplier for perimeter constraint
    fea_interval : int
        Frequency of finite element analysis
    plot_interval : int
        Frequency of plotting
    total_it_num : int
        Total iteration number
    """
    # Step 1: Data initialization
    ew = domain_width / ele_num_per_row  # Width of each finite element
    eh = domain_height / ele_num_per_col  # Height of each finite element
    m = [ele_num_per_col + 1, ele_num_per_row + 1]  # Number of nodes in each dimension
    
    # Create mesh grid for level set function
    x, y = np.meshgrid(ew * np.arange(-0.5, ele_num_per_row + 1), 
                        eh * np.arange(-0.5, ele_num_per_col + 1))
    
    # Get coordinates of the finite element nodes and create elements
    fe_nd_x, fe_nd_y, first_node_per_col = make_nodes(ele_num_per_row, ele_num_per_col, ew, eh)
    ele_nodes_id = make_elements(ele_num_per_row, ele_num_per_col, first_node_per_col)
    
    # Create level set grid
    ls_grid_x = x.flatten()
    ls_grid_y = y.flatten()
    
    # Define the level set grid ID for each element
    ele_ls_grid_id = np.zeros(len(ele_nodes_id), dtype=int)
    for i in range(len(ele_nodes_id)):
        ele_ls_grid_id[i] = np.where(
            (np.abs(ls_grid_x - fe_nd_x[ele_nodes_id[i, 0]] - ew/2) < 1e-10) &
            (np.abs(ls_grid_y - fe_nd_y[ele_nodes_id[i, 0]] - eh/2) < 1e-10)
        )[0][0]
    
    # Define initial level set function
    cx = domain_width / 200 * np.array([33.33, 100, 166.67, 0, 66.67, 133.33, 200, 33.33, 100, 166.67, 0, 66.67, 133.33, 200, 33.33, 100, 166.67])
    cy = domain_height / 100 * np.array([0, 0, 0, 25, 25, 25, 25, 50, 50, 50, 75, 75, 75, 75, 100, 100, 100])
    
    tmp_phi = np.zeros((len(ls_grid_x), len(cx)))
    for i in range(len(cx)):
        tmp_phi[:, i] = -np.sqrt((ls_grid_x - cx[i])**2 + (ls_grid_y - cy[i])**2) + domain_height/10
    
    ls_grid_phi = -(np.max(tmp_phi, axis=1))
    
    # Set boundary conditions for the level set function
    boundary_idx = np.where(
        (ls_grid_x - np.min(ls_grid_x)) * (ls_grid_x - np.max(ls_grid_x)) * 
        (ls_grid_y - np.max(ls_grid_y)) * (ls_grid_y - np.min(ls_grid_y)) <= 1e-10
    )[0]
    ls_grid_phi[boundary_idx] = -1e-6
    
    # Project phi values to the finite element nodes
    points = np.column_stack((ls_grid_x, ls_grid_y))
    fe_nd_phi = griddata(points, ls_grid_phi, np.column_stack((fe_nd_x, fe_nd_y)), method='cubic')
    
    # Create force vector
    f = np.zeros(2 * (ele_num_per_row + 1) * (ele_num_per_col + 1))
    f[2 * (ele_num_per_row + 1) * (ele_num_per_col + 1) - ele_num_per_col - 1] = -1
    
    # Initialize arrays to store objective function, volume ratio, and compliance
    obj = np.zeros(total_it_num)
    vol_ratio = np.zeros(total_it_num)
    compliance = np.zeros(total_it_num)
    
    # Set up the figure for plotting
    fig = plt.figure(figsize=(15, 7))
    
    # Main iteration loop
    it_num = 0
    while it_num < total_it_num:
        print(f"\nFinite Element Analysis No. {it_num+1} starts...")
        
        # Step 2: Finite element analysis
        fe_nd_u, fe_nd_v, mean_compliance = fea(
            1, 1e-3, f, ele_num_per_row, ele_num_per_col, ew, eh, 
            fe_nd_x, fe_nd_y, fe_nd_phi, ele_nodes_id, 0.3
        )
        
        # Calculate geometric quantities
        ls_grid_phi_reshaped = ls_grid_phi.reshape(m[0]+1, m[1]+1)
        ls_grid_curv = calc_curvature(ls_grid_phi_reshaped, ew, eh).flatten()
        
        # Step 3: Shape sensitivity analysis and normal velocity field calculation
        ls_grid_beta = np.zeros_like(ls_grid_x)
        for e in range(ele_num_per_row * ele_num_per_col):
            ls_grid_beta[ele_ls_grid_id[e]] = sensi_analysis(
                1, 1e-3, ele_nodes_id, fe_nd_u, fe_nd_v, ew, eh, 0.3, e, 
                lv, lcur, ls_grid_phi[ele_ls_grid_id[e]], ls_grid_curv[ele_ls_grid_id[e]]
            )
        
        # Get velocity field
        ls_grid_vn = ls_grid_beta / np.max(np.abs(ls_grid_beta))
        
        # Step 4: Level set surface update and reinitialization
        ls_grid_phi = level_set_evolve(
            ls_grid_phi.reshape(m[0]+1, m[1]+1), 
            ls_grid_vn.reshape(m[0]+1, m[1]+1), 
            ew, eh, fea_interval
        ).flatten()
        
        if it_num == 0 or (it_num + 1) % 5 == 0:
            ls_grid_phi = reinitialize(
                ls_grid_phi.reshape(m[0]+1, m[1]+1), ew, eh, 20
            ).flatten()
        
        # Update phi values at finite element nodes
        fe_nd_phi = griddata(points, ls_grid_phi, np.column_stack((fe_nd_x, fe_nd_y)), method='cubic')
        
        # Step 5: Results visualization
        # Calculate objective function, volume ratio, and compliance
        obj[it_num], vol_ratio[it_num], compliance[it_num] = obj_fun(
            mean_compliance, ls_grid_phi, domain_width, domain_height, ew, eh, lv
        )
        
        if it_num == 0 or (it_num + 1) % plot_interval == 0:
            plt.clf()
            
            # Plot level set contour
            ax1 = fig.add_subplot(121)
            fe_nd_phi_reshaped = fe_nd_phi.reshape(m[0], m[1])
            contour = ax1.contourf(fe_nd_x.reshape(m[0], m[1]), fe_nd_y.reshape(m[0], m[1]), 
                           fe_nd_phi_reshaped, levels=[0, np.max(fe_nd_phi)])
            ax1.set_aspect('equal')
            ax1.grid(True)
            ax1.set_title('Level Set Contour')
            
            # Plot 3D surface
            ax2 = fig.add_subplot(122, projection='3d')
            surf = ax2.plot_surface(x, y, -ls_grid_phi.reshape(m[0]+1, m[1]+1), 
                            cmap=cm.coolwarm, alpha=0.8)
            ax2.contour(x, y, -ls_grid_phi.reshape(m[0]+1, m[1]+1), levels=[0], colors='k')
            ax2.view_init(30, 45)
            ax2.set_title('Level Set Function')
            
            plt.suptitle(f'Iteration {it_num+1}, Volume Ratio: {vol_ratio[it_num]:.3f}, Compliance: {compliance[it_num]:.3f}')
            plt.tight_layout()
            plt.pause(0.5)
        
        # Step 6: Go to step 2
        it_num += 1
    
    plt.show()
    return obj, vol_ratio, compliance, ls_grid_phi, fe_nd_phi

def add_bound_condition(k, ele_num_per_col):
    """
    Add boundary conditions to the stiffness matrix
    """
    n = np.arange(1, ele_num_per_col + 2)
    for i in range(len(n)):
        k[2 * n[i] - 2:2 * n[i], :] = 0
        k[:, 2 * n[i] - 2:2 * n[i]] = 0
        k[2 * n[i] - 2, 2 * n[i] - 2] = 1
        k[2 * n[i] - 1, 2 * n[i] - 1] = 1
    
    return k

def assemble(k, ke, elements_node_id, ele_id):
    """
    Assemble the element stiffness matrix into the global stiffness matrix
    """
    m = elements_node_id[ele_id, :]
    for i in range(len(m)):
        for j in range(len(m)):
            k[2*m[i]-2:2*m[i], 2*m[j]-2:2*m[j]] += ke[2*i:2*i+2, 2*j:2*j+2]
    
    return k

def basic_ke(e, nu, a, b):
    """
    Return the element stiffness matrix of a full/empty element
    """
    k = np.array([
        -1/6/a/b*(nu*a**2-2*b**2-a**2), 1/8*nu+1/8, -1/12/a/b*(nu*a**2+4*b**2-a**2), 3/8*nu-1/8,
        1/12/a/b*(nu*a**2-2*b**2-a**2), -1/8*nu-1/8, 1/6/a/b*(nu*a**2+b**2-a**2), -3/8*nu+1/8
    ])
    
    ke = e/(1-nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    
    return ke

def calc_curvature(phi, dx, dy):
    """
    Calculate curvature of the level set function
    """
    matrix = matrix4diff(phi)
    phix = (matrix['i_plus_1'] - matrix['i_minus_1'])/(2 * dx)
    phiy = (matrix['j_plus_1'] - matrix['j_minus_1'])/(2 * dy)
    
    phiyx_bk, phiyx_fw = upwind_diff(phiy, dx, 'x')
    phixy = (phiyx_bk + phiyx_fw)/2
    
    phixx = (matrix['i_plus_1'] - 2*phi + matrix['i_minus_1'])/dx**2
    phiyy = (matrix['j_plus_1'] - 2*phi + matrix['j_minus_1'])/dy**2
    
    denominator = (phix**2 + phiy**2)**1.5 + 100*np.finfo(float).eps
    curvature = (phixx * phiy**2 - 2 * phix * phiy * phixy + phiyy * phix**2) / denominator
    
    return curvature

def ele_stiff_matrix(ew, eh, e1, e0, nu, phi, ele_node_id, i):
    """
    Return the element stiffness matrix according to their relative position to the boundary
    """
    if np.min(phi[ele_node_id[i,:]]) > 0:  # Element is inside the boundary
        ke = basic_ke(e1, nu, ew, eh)
    elif np.max(phi[ele_node_id[i,:]]) < 0:  # Element is outside the boundary
        ke = basic_ke(e0, nu, ew, eh)
    else:  # Element is cut by the boundary
        s, t = np.meshgrid(np.arange(-1, 1.1, 0.1), np.arange(-1, 1.1, 0.1))
        s = s.flatten()
        t = t.flatten()
        
        tmp_phi = ((1 - s)*(1 - t)/4 * phi[ele_node_id[i,0]] + 
                  (1 + s)*(1 - t)/4 * phi[ele_node_id[i,1]] +
                  (1 + s)*(1 + t)/4 * phi[ele_node_id[i,2]] + 
                  (1 - s)*(1 + t)/4 * phi[ele_node_id[i,3]])
        
        area_ratio = np.sum(tmp_phi >= 0) / len(s)
        ke = area_ratio * basic_ke(e1, nu, ew, eh)
    
    return ke

def fea(e1, e0, f, ele_num_per_row, ele_num_per_col, ew, eh, fe_nd_x, fe_nd_y, fe_nd_phi, ele_nodes_id, nu):
    """
    Perform finite element analysis and return displacement field and mean compliance
    """
    n_dof = 2 * (ele_num_per_row + 1) * (ele_num_per_col + 1)
    k = np.zeros((n_dof, n_dof))
    
    for i in range(ele_num_per_row * ele_num_per_col):
        ke = ele_stiff_matrix(ew, eh, e1, e0, nu, fe_nd_phi, ele_nodes_id, i)
        k = assemble(k, ke, ele_nodes_id, i)
    
    k = add_bound_condition(k, ele_num_per_col)
    
    # Solve system K*U = F
    u = np.linalg.solve(k, f)
    
    # Calculate mean compliance
    force_idx = 2 * (ele_num_per_row + 1) * (ele_num_per_col + 1) - ele_num_per_col - 1
    mean_compliance = f[force_idx] * u[force_idx]
    
    # Extract u and v components
    u_comp = u[0::2]
    v_comp = u[1::2]
    
    return u_comp, v_comp, mean_compliance

def level_set_evolve(phi0, vn, dx, dy, loop_num):
    """
    Update the level set surface using a first order space convex scheme
    """
    det_t = 0.5 * min(dx, dy) / np.max(np.abs(vn))
    phi = phi0.copy()
    
    for i in range(loop_num):
        dx_l, dx_r = upwind_diff(phi, dx, 'x')
        dy_l, dy_r = upwind_diff(phi, dy, 'y')
        
        grad_plus = np.sqrt(np.maximum(dx_l, 0)**2 + np.minimum(dx_r, 0)**2 + 
                            np.maximum(dy_l, 0)**2 + np.minimum(dy_r, 0)**2)
        
        grad_minus = np.sqrt(np.minimum(dx_l, 0)**2 + np.maximum(dx_r, 0)**2 + 
                             np.minimum(dy_l, 0)**2 + np.maximum(dy_r, 0)**2)
        
        phi = phi - det_t * (np.maximum(vn, 0) * grad_plus + np.minimum(vn, 0) * grad_minus)
    
    return phi

def make_elements(ele_num_per_row, ele_num_per_col, first_node_per_col):
    """
    Create finite elements
    """
    ele_nodes_id = np.zeros((ele_num_per_row * ele_num_per_col, 4), dtype=int)
    
    for i in range(1, ele_num_per_row + 1):
        idx_range = np.arange(i * ele_num_per_col, (i-1) * ele_num_per_col, -1)
        node_range = np.arange(first_node_per_col[i-1], first_node_per_col[i-1] - ele_num_per_col, -1)
        ele_nodes_id[idx_range-1, 3] = node_range
    
    ele_nodes_id[:, 0] = ele_nodes_id[:, 3] - 1
    ele_nodes_id[:, 1] = ele_nodes_id[:, 0] + ele_num_per_col + 1
    ele_nodes_id[:, 2] = ele_nodes_id[:, 1] + 1
    
    return ele_nodes_id

def make_nodes(ele_num_per_row, ele_num_per_col, ele_width, ele_height):
    """
    Create nodes for finite elements
    """
    x, y = np.meshgrid(ele_width * np.arange(0, ele_num_per_row + 1), 
                       ele_height * np.arange(0, ele_num_per_col + 1))
    
    first_node_per_col = np.where(y.flatten() == np.max(y.flatten()))[0]
    
    return x.flatten(), y.flatten(), first_node_per_col

def matrix4diff(phi):
    """
    Prepare matrices for finite difference calculations
    """
    i_minus_1 = np.zeros_like(phi)
    i_plus_1 = np.zeros_like(phi)
    j_minus_1 = np.zeros_like(phi)
    j_plus_1 = np.zeros_like(phi)
    
    i_minus_1[:, 0] = phi[:, -1]
    i_minus_1[:, 1:] = phi[:, :-1]
    
    i_plus_1[:, -1] = phi[:, 0]
    i_plus_1[:, :-1] = phi[:, 1:]
    
    j_minus_1[0, :] = phi[-1, :]
    j_minus_1[1:, :] = phi[:-1, :]
    
    j_plus_1[-1, :] = phi[0, :]
    j_plus_1[:-1, :] = phi[1:, :]
    
    return {
        'i_minus_1': i_minus_1,
        'i_plus_1': i_plus_1,
        'j_minus_1': j_minus_1,
        'j_plus_1': j_plus_1
    }

def obj_fun(mean_compliance, ls_grid_phi, domain_width, domain_height, ew, eh, lv):
    """
    Calculate objective function, volume ratio, and compliance
    """
    # Calculate volume ratio
    volume = np.sum(ls_grid_phi > 0) * ew * eh
    total_volume = domain_width * domain_height
    vol_ratio = volume / total_volume
    
    # Calculate objective function
    obj = mean_compliance + lv * vol_ratio
    
    return obj, vol_ratio, mean_compliance

def reinitialize(phi0, dx, dy, loop_num):
    """
    Regularize the level set function to be a signed distance function
    """
    phi = phi0.copy()
    
    for i in range(loop_num + 1):
        dx_l, dx_r = upwind_diff(phi, dx, 'x')
        dy_l, dy_r = upwind_diff(phi, dy, 'y')
        
        dx_c = (dx_l + dx_r) / 2
        dy_c = (dy_l + dy_r) / 2
        
        s = phi / (np.sqrt(phi**2 + (dx_c**2 + dy_c**2) * dx**2) + np.finfo(float).eps)
        
        det_t = 0.5 * min(dx, dy) / np.max(np.abs(s))
        
        grad_plus = np.sqrt(np.maximum(dx_l, 0)**2 + np.minimum(dx_r, 0)**2 + 
                            np.maximum(dy_l, 0)**2 + np.minimum(dy_r, 0)**2)
        
        grad_minus = np.sqrt(np.minimum(dx_l, 0)**2 + np.maximum(dx_r, 0)**2 + 
                             np.minimum(dy_l, 0)**2 + np.maximum(dy_r, 0)**2)
        
        phi = phi - det_t * ((np.maximum(s, 0) * grad_plus + np.minimum(s, 0) * grad_minus) - s)
    
    return phi

def sensi_analysis(e1, e0, ele_nodes_id, u, v, ew, eh, nu, ele_id, l4vol, l4curv, phi, curvature):
    """
    Perform sensitivity analysis and calculate velocity field
    """
    ae = np.array([
        u[ele_nodes_id[ele_id, 0]], v[ele_nodes_id[ele_id, 0]],
        u[ele_nodes_id[ele_id, 1]], v[ele_nodes_id[ele_id, 1]],
        u[ele_nodes_id[ele_id, 2]], v[ele_nodes_id[ele_id, 2]],
        u[ele_nodes_id[ele_id, 3]], v[ele_nodes_id[ele_id, 3]]
    ])
    
    b = 0.5 * np.array([
        [-1/ew, 0, 1/ew, 0, 1/ew, 0, -1/ew, 0],
        [0, -1/eh, 0, -1/eh, 0, 1/eh, 0, 1/eh],
        [-1/eh, -1/ew, -1/eh, 1/ew, 1/eh, 1/ew, 1/eh, -1/ew]
    ])
    
    strain = b @ ae  # strain is a 3x1 vector
    
    if phi > 0.75 * ew:
        e = e1
    elif phi < -0.75 * ew:
        e = e0
    else:
        density_min = 1e-3
        xd = phi / (0.75 * ew)
        e = e1 * 0.75 * (1.0 - density_min) * (xd - xd**3 / 3.0) + 0.5 * (1 + density_min)
    
    d = e / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu)/2]
    ])
    
    strain_energy = strain.T @ d @ strain
    beta = l4vol - strain_energy - l4curv * curvature
    
    return beta

def upwind_diff(phi, dx, str_direction):
    """
    Calculate backward and forward finite differences
    """
    matrix = matrix4diff(phi)
    
    if str_direction == 'x':
        back_diff = (phi - matrix['i_minus_1']) / dx
        fawd_diff = (matrix['i_plus_1'] - phi) / dx
    elif str_direction == 'y':
        back_diff = (phi - matrix['j_minus_1']) / dx
        fawd_diff = (matrix['j_plus_1'] - phi) / dx
    
    return back_diff, fawd_diff

# Example usage
if __name__ == "__main__":
    from fealpy.backend import backend_manager as bm
    from soptx.solver import ElasticFEMSolver, AssemblyMethod
    from soptx.material import DensityBasedMaterialConfig, DensityBasedMaterialInstance
    from fealpy.mesh import UniformMesh2d
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from soptx.pde import Cantilever2dData1
    material_config = DensityBasedMaterialConfig(
                        elastic_modulus=1,            
                        minimal_modulus=1e-9,         
                        poisson_ratio=0.3,            
                        plane_assumption="plane_stress",    
                        interpolation_model="SIMP",    
                        penalty_factor=3
                    )
    materials = DensityBasedMaterialInstance(config=material_config)
    p = 1
    nx, ny = 32, 22
    h = [1, 1]
    domain_width = nx * h[0]  
    domain_height = ny * h[1]  
    mesh_fe = UniformMesh2d(extent=[0, nx, 0, ny], h=[1, 1], origin=[0.0, 0.0], 
                            ipoints_ordering='yx', device='cpu')
    mesh_ls = UniformMesh2d(extent=[0, nx+1, 0, ny+1], h=[1, 1], origin=[-0.5, -0.5], 
                            ipoints_ordering='yx', device='cpu')
    
    NN_ls = mesh_ls.number_of_nodes()
    node_ls = mesh_ls.entity('node')
    node_ls_x = node_ls[:, 0]
    node_ls_y = node_ls[:, 1]
    node_fe = mesh_fe.entity('node')

    # 定义初始结构的 "骨架" 点
    cx = domain_width / 200 * bm.array([33.33, 100, 166.67, 0, 66.67, 133.33, 200, 33.33, 100, 166.67, 0, 66.67, 133.33, 200, 33.33, 100, 166.67])
    cy = domain_height / 100 * bm.array([0, 0, 0, 25, 25, 25, 25, 50, 50, 50, 75, 75, 75, 75, 100, 100, 100])

    # 计算初始水平集函数
    phi_tmp = np.zeros((NN_ls, len(cx)))
    for i in range(len(cx)):
        phi_tmp[:, i] = -np.sqrt((node_ls_x - cx[i])**2 + (node_ls_y - cy[i])**2) + domain_height / 10
    phi_ls = -(bm.max(phi_tmp, axis=1))
    
    # 设置边界条件
    boundary_idx = bm.nonzero(mesh_ls.boundary_node_flag())[0]
    phi_ls[boundary_idx] = -1e-6
    
    node_fe_x = node_fe[:, 0]
    node_fe_y = node_fe[:, 1]

    # 将水平集函数投影到有限元节点
    from scipy.interpolate import griddata
    points = np.column_stack((node_ls_x, node_ls_y))
    phi_fe = griddata(points, phi_ls, np.column_stack((node_fe_x, node_fe_y)), method='cubic')
    
    fe_nd_phi_test = mesh_fe.cell_barycenter()

    mesh_fe.nodedata['phi_fe'] = phi_fe
    mesh_fe.to_vtk('/home/heliang/FEALPy_Development/soptx/soptx/vtu/fe_phi.vts')

    mesh_ls.nodedata['phi_ls'] = phi_ls
    mesh_ls.to_vtk('/home/heliang/FEALPy_Development/soptx/soptx/vtu/ls_phi.vts')
    
    pde = Cantilever2dData1(
            xmin=0, xmax=nx * h[0],
            ymin=0, ymax=ny * h[1],
            T = -1
            )
    GD = mesh.geo_dimension()
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
    solver_sta = ElasticFEMSolver(
                materials=materials,
                tensor_space=tensor_space_C,
                pde=pde,
                assembly_method=AssemblyMethod.STANDARD,
                solver_type='direct',
                solver_params={'solver_type': 'mumps'}, 
            )
        

    ew = 0.1  # 单元宽度（与TOPLSM中一致）
    eh = 0.1  # 单元高度
    ew_1 = 1  # 单元宽度（与TOPLSM中一致）
    eh_1 = 1  # 单元高度
    e1 = 1.0  # 实体材料的弹性模量
    e0 = 1e-3  # 空洞材料的弹性模量
    nu = 0.3  # 泊松比

    ke = basic_ke(e1, nu, ew, eh)
    ke_1 = basic_ke(e1, nu, ew_1, eh_1)
    from fealpy.backend import backend_manager as bm
    print(f"Element stiffness matrix (ke):\n{bm.max(ke - ke_1)}")



    # Parameters (example values)
    domain_width = 2.0
    domain_height = 1.0
    ele_num_per_row = 60
    ele_num_per_col = 30
    lv = 100  # Lagrange multiplier for volume constraint
    lcur = 0.01  # Lagrange multiplier for perimeter constraint
    fea_interval = 1
    plot_interval = 5
    total_it_num = 100
    
    # Run optimization
    obj, vol_ratio, compliance, ls_grid_phi, fe_nd_phi = toplsm(
        domain_width, domain_height, ele_num_per_row, ele_num_per_col,
        lv, lcur, fea_interval, plot_interval, total_it_num
    )
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, total_it_num+1), compliance)
    plt.title('Mean Compliance')
    plt.xlabel('Iteration')
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, total_it_num+1), vol_ratio)
    plt.title('Volume Ratio')
    plt.xlabel('Iteration')
    
    plt.tight_layout()
    plt.show()