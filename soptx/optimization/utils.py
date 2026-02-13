"""拓扑优化工具函数模块

该模块包含了拓扑优化中使用的通用工具函数，包括：
- MMA 子问题求解器
"""

from typing import Tuple
from numpy.linalg import solve

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from typing import Union, Optional, Literal
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import Function
from fealpy.mesh import HomogeneousMesh, SimplexMesh, TensorMesh

"""Method of Moving Asymptotes (MMA) 子问题求解器

实现了 MMA 算法中子问题的求解,使用原始-对偶内点法.
参考 : Svanberg, K. (1987). The method of moving asymptotes—a new method for 
structural optimization. International Journal for Numerical Methods in Engineering.
"""
def solve_mma_subproblem(m: int, n: int, 
                        epsimin: float,
                        low: TensorLike, upp: TensorLike,
                        alfa: TensorLike, beta: TensorLike,
                        p0: TensorLike, q0: TensorLike,
                        P: TensorLike, Q: TensorLike,
                        a0: float,
                        a: TensorLike = None,
                        b: TensorLike = None,
                        c: TensorLike = None,
                        d: TensorLike = None
                    ) -> Tuple[TensorLike, ...]:
    """求解 MMA 子问题
    
    使用 primal-dual Newton method 求解由 MMA 算法产生的凸近似子问题.
    
    Parameters
    ----------
    m : 约束数量, 约束函数 f_i(x) 的个数
    n : 设计变量数量, 变量 x_j 的个数
    epsimin : 最小收敛容差
    low : 下渐近线
    upp : 上渐近线
    alfa : 变量下界
    beta : 变量上界
    p0 (n, 1): 目标函数的正梯度项
    q0 (n, 1): 目标函数的负梯度项
    P (m, n): 约束函数的正梯度项
    Q (m, n): 约束函数的负梯度项
    a0 (float): 目标函数的线性项 a_0*z 的系数
    a (m, 1): 约束的线性项 a_i*z 的系数
    b (m, 1): -r_i
    c (m, 1): 约束的二次项 c_iy_i 的系数
    d (m, 1): 约束的二次项 0.5*d_i*y_i^2 的系数
    
    Returns
    - xmma (n, 1): 自然变量
    - ymma: 人工变量 
    - zmma: 人工变量
    - lam (m, 1): m 个约束的拉格朗日乘子 lambda
    - xsi (n, 1): n 个约束 alpha_j-x_j 的拉格朗日乘子 xi
    - eta : n 个约束 x_j-beta_j 的拉格朗日乘子 eta
    - mu : m 个约束 -y_i <= 0 的拉格朗日乘子 mu
    - zet : 1 个约束 -z <= 0 的拉格朗日乘子 zeta
    - s : m 个约束的松弛变量 s
    """

    # 变量初始化
    een = bm.ones((n, 1), dtype=bm.float64)
    eem = bm.ones((m, 1), dtype=bm.float64)
    x = 0.5 * (alfa + beta)
    #! A-1: 保证 x 远离渐近线/上下界 ----
    eps_x = 1e-12

    def clamp_x(x_in: TensorLike) -> TensorLike:
        xmin = bm.maximum(alfa, low) + eps_x
        xmax = bm.minimum(beta, upp) - eps_x
        xmax = bm.maximum(xmax, xmin + eps_x)
        return bm.minimum(bm.maximum(x_in, xmin), xmax)

    x = clamp_x(x)
    #! ------------------------------------------------------------
    y = bm.copy(eem)
    z = bm.array([[1.0]])
    lam = bm.copy(eem)
    
    #! A-2：用安全分母初始化 xsi / eta，避免 (x-alfa) 或 (beta-x) 为 0 ----
    xalfa = bm.maximum(x - alfa, bm.tensor(eps_x, dtype=bm.float64))
    betax = bm.maximum(beta - x, bm.tensor(eps_x, dtype=bm.float64))

    xsi = een / xalfa
    xsi = bm.maximum(xsi, een)

    eta = een / betax
    eta = bm.maximum(eta, een)
    # xsi = een / (x - alfa)
    # xsi = bm.maximum(xsi, een)

    # eta = een / (beta - x)
    # eta = bm.maximum(eta, een)
    #! ----------------------------------------------------------------------
    
    mu = bm.maximum(eem, 0.5*c)
    zet = bm.array([[1.0]])
    s = bm.copy(eem)

    epsi = 1 # 松弛参数, 每次外循环迭代中逐步减小   
    epsvecn = epsi * een # (n, 1)
    epsvecm = epsi * eem # (m, 1)
    # 外循环迭代: 逐步减小松弛参数 epsi
    itera = 0
    while epsi > epsimin:
        epsvecn = epsi * een # (n, 1)
        epsvecm = epsi * eem # (m, 1)
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1

        plam = p0 + bm.dot(P.T, lam)
        qlam = q0 + bm.dot(Q.T, lam)
        gvec = bm.dot(P, uxinv1) + bm.dot(Q, xlinv1)
        dpsidx = plam / ux2 - qlam / xl2

        # 1. 计算 KKT 残差
        rex = dpsidx - xsi + eta           # (n, 1)
        rey = c + d*y - mu - lam           # (m, 1) 
        rez = a0 - zet - bm.dot(a.T, lam)  # (m, 1)
        relam = gvec - a*z - y + s - b     # (m, 1)
        rexsi = xsi * (x - alfa) - epsvecn # (n, 1)
        reeta = eta * (beta - x) - epsvecn # (n, 1)
        remu = mu * y - epsvecm            # (m, 1)
        rezet = zet * z - epsi             # (1, 1)
        res = lam * s - epsvecm            # (m, 1)
    
        residu1 = bm.concatenate((rex, rey, rez), axis=0)   # (n+m+m, 1)
        residu2 = bm.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0) # (m+n+n+m+1+m, 1)
        residu = bm.concatenate((residu1, residu2), axis=0)
        residunorm = bm.sqrt((bm.dot(residu.T, residu)).item())
        residumax = bm.max(bm.abs(residu))
        
        ittt = 0
        # 内循环迭代: 在固定的 epsi 下求解 KKT 系统
        while (residumax > 0.9 * epsi) and (ittt < 200):
            ittt = ittt + 1
            itera = itera + 1
            
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2 # (n, 1)
            xlinv2 = een / xl2 # (n, 1)
            
            plam = p0 + bm.dot(P.T, lam)
            qlam = q0 + bm.dot(Q.T, lam)
            gvec = bm.dot(P, uxinv1) + bm.dot(Q, xlinv1)

            GG = bm.einsum('j, ij -> ij', uxinv2.flatten(), P) - \
                 bm.einsum('j, ij -> ij', xlinv2.flatten(), Q)          # (m, n) 
            
            # 2. 计算 Newton 方向的一阶残差 delta_x, delta_y, delta_z, delta_lambda
            dpsidx = plam / ux2 - qlam / xl2                            # (n, 1)
            #! A-3：Newton 迭代内安全分母（每次内循环都要更新，因为 x 在变）----
            xalfa = bm.maximum(x - alfa, bm.tensor(eps_x, dtype=bm.float64))
            betax = bm.maximum(beta - x, bm.tensor(eps_x, dtype=bm.float64))
            delx = dpsidx - epsvecn / xalfa + epsvecn / betax           # (n, 1)
            # delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x) # (n, 1)
            #! ----------------------------------------------------------------
            dely = c + d * y - lam - epsvecm / y                        # (m, 1)
            delz = a0 - bm.dot(a.T, lam) - epsi / z                     # (1, 1)
            dellam = gvec - a * z - y - b + epsvecm / lam               # (m, 1)
            
            # 3. 计算 Hessian 的对角线
            diagx = plam / ux3 + qlam / xl3
            #! A-3
            diagx = 2 * diagx + xsi / xalfa + eta / betax               # (n, 1)
            # diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x) # (n, 1)
            #! ----------------------------------------------------------------
            diagxinv = een / diagx
            diagy = d + mu / y                                      # (m, 1)
            diagyinv = eem/diagy
            diaglam = s / lam                                       # (m, 1)
            diaglamyi = diaglam + diagyinv
            
            # 4. 求解 KKT 线性系统
            if m < n:
                # 选择 (\Delta\lambda, \Delta z) 系统
                blam = dellam + dely / diagy - bm.dot(GG, (delx / diagx)) # (m, 1)
                bb = bm.concatenate((blam, delz), axis=0)                 # (m+1, 1)
                D_lamyi = diaglamyi * bm.eye(1)  
                GD_xG = bm.einsum('ik, k, jk -> ij', GG, diagxinv.flatten(), GG)  
                Alam = D_lamyi + GD_xG  # (m, 1)
                AAr1 = bm.concatenate((Alam, a), axis=1)     # (m, m+1)
                AAr2 = bm.concatenate((a, -zet/z), axis=0).T # (1, m+1)
                AA = bm.concatenate((AAr1, AAr2), axis=0)    # (m+1, m+1)
                solut = solve(AA, bb)
                dlam = solut[0:m]
                dz = solut[m:m+1]                                # (m, 1)
                dx = -delx / diagx - bm.dot(GG.T, dlam) / diagx  # (m, 1)
            else:
                # 选择 (\Delta x, \Delta z) 系统
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely/diagy
                D_x = diagx * bm.eye(1)
                GD_lamyiG = bm.einsum('ik, k, jk -> ij', GG, diaglamyiinv.flatten(), GG)
                Axx = D_x + GD_lamyiG
                azz = zet/z + bm.dot(a.T, (a/diaglamyi))
                axz = bm.dot(-GG.T, (a/diaglamyi))
                bx = delx + bm.dot(GG.T, (dellamyi/diaglamyi))
                bz = delz - bm.dot(a.T, (dellamyi/diaglamyi))
                AAr1 = bm.concatenate((Axx, axz), axis=1)
                AAr2 = bm.concatenate((axz.T, azz), axis=1)
                AA = bm.concatenate((AAr1, AAr2), axis=0) # (n+1, n+1)
                bb = bm.concatenate((-bx, -bz), axis=0)
                solut = solve(AA, bb)
                dx = solut[0:n]
                dz = solut[n:n+1]                                                               # (m, 1)
                dlam = bm.dot(GG, dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi # (m, 1)
                
            # 5.计算 Newton 方向 \Delta w
            dy = -dely / diagy + dlam / diagy                  # (m, 1)
            #! A-3
            dxsi = -xsi + epsvecn / xalfa - (xsi * dx) / xalfa  # (n, 1)
            deta = -eta + epsvecn / betax + (eta * dx) / betax  # (n, 1)
            # dxsi = -xsi + epsvecn/(x-alfa) - (xsi*dx)/(x-alfa) # (n, 1)
            # deta = -eta + epsvecn/(beta-x) + (eta*dx)/(beta-x) # (n, 1)
            #! ----------------------------------------------------------------
            dmu = -mu + epsvecm / y - (mu * dy) / y            # (m, 1)
            dzet = -zet + epsi / z - zet * dz / z              # (1, 1)
            ds = -s + epsvecm / lam - (s * dlam) / lam         # (m, 1)
            xx = bm.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)          # (m+n+n+m+1+m, 1)
            dxx = bm.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0) # (m+n+n+m+1+m, 1)
            
            # 6. 确定线搜索步长
            stepxx = -1.01 * dxx / xx
            stmxx = bm.max(stepxx)
            #! A-4
            stepalfa = -1.01 * dx / xalfa
            stmalfa = bm.max(stepalfa)
            stepbeta =  1.01 * dx / betax
            stmbeta = bm.max(stepbeta)
            # stepalfa = -1.01 * dx / (x - alfa)
            # stmalfa = bm.max(stepalfa)
            # stepbeta = 1.01 * dx / (beta - x)
            # stmbeta = bm.max(stepbeta)
            #! ----------------------------------------------------------------
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0 / stminv
            
            xold = bm.copy(x)
            yold = bm.copy(y)
            zold = bm.copy(z)
            lamold = bm.copy(lam)
            xsiold = bm.copy(xsi)
            etaold = bm.copy(eta)
            muold = bm.copy(mu)
            zetold = bm.copy(zet)
            sold = bm.copy(s)
            
            # 7. 线搜索更新变量
            itto = 0
            resinew = 2 * residunorm
            while (resinew > residunorm) and (itto < 50):
                itto = itto + 1
 
                x = xold + steg*dx
                #! A-5: 更新 x 后进行夹紧 ----
                x = clamp_x(x)   
                #! ---------------------------------------------
                y = yold + steg*dy
                z = zold + steg*dz
                lam = lamold + steg*dlam
                xsi = xsiold + steg*dxsi
                eta = etaold + steg*deta
                mu = muold + steg*dmu
                zet = zetold + steg*dzet
                s = sold + steg*ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = een / ux1
                xlinv1 = een / xl1

                plam = p0 + bm.dot(P.T, lam)
                qlam = q0 + bm.dot(Q.T, lam)
                gvec = bm.dot(P, uxinv1) + bm.dot(Q, xlinv1)
                dpsidx = plam/ux2 - qlam/xl2

                rex = dpsidx - xsi + eta
                rey = c + d*y - mu - lam
                rez = a0 - zet - bm.dot(a.T, lam)
                relam = gvec - a*z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu1 = bm.concatenate((rex, rey, rez), axis=0)
                residu2 = bm.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
                residu = bm.concatenate((residu1, residu2), axis=0)
                resinew = bm.sqrt(bm.dot(residu.T, residu))

                steg = steg / 2

            residunorm = resinew.copy()
            residumax = bm.max(bm.abs(residu))
            steg = 2 * steg
            
        epsi = 0.1 * epsi
    
    # 返回最优解
    xmma = bm.copy(x)
    ymma = bm.copy(y)
    zmma = bm.copy(z)
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma


def compute_volume(
                density: Union[Function, TensorLike],
                mesh: HomogeneousMesh,
                density_location: Literal['element', 'node'] = 'element',
                integration_order: Optional[int] = None
            ) -> float:
    """计算设计域内材料的总体积"""
    if density_location in ['element']:
        rho_element = density  # (NC, )
        cell_measure = mesh.entity_measure('cell')

        current_volume = bm.sum(cell_measure * rho_element[:])

        return current_volume
    
    elif density_location in ['element_multiresolution']:

        rho_sub_element = density # (NC, n_sub)

        NC, n_sub = rho_sub_element.shape
        cell_measure = mesh.entity_measure('cell')
        sub_cm = bm.tile(cell_measure.reshape(NC, 1), (1, n_sub)) / n_sub # (NC, n_sub)

        current_volume = bm.einsum('cn, cn -> ', sub_cm, rho_sub_element[:])

        return current_volume

    elif density_location in ['node']:
        #* 标准节点密度表征下的体积计算
        # 计算单元积分点处的重心坐标
        qf = mesh.quadrature_formula(q=integration_order)
        # bcs_e.shape = ( (NQ_x, GD), (NQ_y, GD) ), ws_e.shape = (NQ, )
        bcs, ws = qf.get_quadrature_points_and_weights()

        rho_q = density(bcs) # (NC, NQ)

        if isinstance(mesh, SimplexMesh):
            cm = mesh.entity_measure('cell')
            current_volume = bm.einsum('q, cq, c -> ', ws, rho_q, cm)
        
        elif isinstance(mesh, TensorMesh):
            J = mesh.jacobi_matrix(bcs)
            detJ = bm.abs(bm.linalg.det(J))
            current_volume = bm.einsum('q, cq, cq -> ', ws, rho_q, detJ)

        #* 简化节点密度表征下的体积计算
        # cell_measure = self._mesh.entity_measure('cell')
        # total_volume = bm.sum(cell_measure)

        # rho_node = density[:] # (NN, )
        # avg_rho = bm.sum(rho_node) / rho_node.shape[0]
        # current_volume = total_volume * avg_rho

        return current_volume
    
    elif density_location in ['node_multiresolution']:

        rho_sub_q = density # (NC, n_sub, NQ)
        NC, n_sub, NQ = rho_sub_q.shape

        if isinstance(mesh, SimplexMesh):
            cell_measure = mesh.entity_measure('cell')
            sub_cm = bm.tile(cell_measure.reshape(NC, 1), (1, n_sub)) / n_sub # (NC, n_sub)
            current_volume = bm.einsum('q, cnq, cn -> ', ws, rho_sub_q, sub_cm)
        
        elif isinstance(mesh, TensorMesh):
            # 计算位移单元积分点处的重心坐标
            qf_e = mesh.quadrature_formula(q=integration_order)
            # bcs_e.shape = ( (NQ, GD), (NQ, GD) ), ws_e.shape = (NQ, )
            bcs_e, ws_e = qf_e.get_quadrature_points_and_weights()

            # 把位移单元高斯积分点处的重心坐标映射到子密度单元 (子参考单元) 高斯积分点处的重心坐标 (仍表达在位移单元中)
            from soptx.analysis.utils import map_bcs_to_sub_elements
            # bcs_eg.shape = ( (n_sub, NQ, GD), (n_sub, NQ, GD) ), ws_e.shape = (NQ, )
            bcs_eg = map_bcs_to_sub_elements(bcs_e=bcs_e, n_sub=n_sub)
            bcs_eg_x, bcs_eg_y = bcs_eg[0], bcs_eg[1]

            detJ_eg = bm.zeros((NC, n_sub, NQ)) # (NC, n_sub, NQ)
            for s_idx in range(n_sub):
                sub_bcs = (bcs_eg_x[s_idx, :, :], bcs_eg_y[s_idx, :, :])  # ((NQ, GD), (NQ, GD))

                J_sub = mesh.jacobi_matrix(sub_bcs) # (NC, NQ, GD, GD)
                detJ_sub = bm.abs(bm.linalg.det(J_sub)) # (NC, NQ)

                detJ_eg[:, s_idx, :] = detJ_sub

            current_volume = bm.einsum('q, cnq, cnq -> ', ws_e, rho_sub_q, detJ_eg)

    else:
        raise ValueError(f"Unsupported density_location: {density_location}")