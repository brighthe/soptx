function dsi_dx = compute_stress_sensitivity(xPhys, U, cluster_idx, dsi_dvm, ...
                                           dvm_dstress, lambda, B, D, KE, ...
                                           penal_S, penal_K, H, Hs, nelx, nely, ft)

    nc = length(cluster_idx);
    nele = nelx * nely;
    dsi_dx = zeros(nc, nele);
    
    % 计算应力惩罚函数的导数 ∂η_S/∂ρ
    detaS_drho = penal_S * xPhys.^(penal_S - 1);

    % 计算 ∂K/∂ρ (SIMP)
    detaK_drho = penal_K * xPhys.^(penal_K - 1);

    for e = 1:nele
        ely = mod(e - 1, nely) + 1;
        elx = floor((e - 1) / nely) + 1;
        
        n1 = (nely+1)*(elx-1) + ely;
        n2 = (nely+1)*elx + ely;
        edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];

        % 单元位移
        Ue = U(edof);
        
        % 应变和应力
        strain_e = B * Ue;
        stress_solid = D * strain_e;

        for i = 1:nc
            % ===== 第 1 项：应力惩罚导数项 =====
            if ismember(e, cluster_idx{i})
                % ∂s_i/∂σ^vM * ∂σ^vM/∂σ (1×3)
                grad_vm = dsi_dvm(i, e) * dvm_dstress(e, :);
                
                % ∂η_S/∂ρ_b * E*B*u (3×1)
                stress_term = detaS_drho(ely, elx) * stress_solid;
                
                % 第 1 项
                term1 = grad_vm * stress_term;  
            else
                term1 = 0;
            end

            % ===== 第 2 项：刚度矩阵导数项 =====
            % η_S(ρ_e) * λ_i^T * ∂K/∂ρ_e * u
            etaS_e = xPhys(ely, elx)^penal_S;
            dK_drho_e = detaK_drho(ely, elx) * KE;
            term2 = -etaS_e * lambda(edof, i)' * dK_drho_e * Ue;

            dsi_drho_e = term1 + term2;

            if ft == 2
                % 密度过滤：∂s_i/∂x_b = Σ_e [∂s_i/∂ρ_e * ∂ρ_e/∂x_b]
                % 其中 ∂ρ_e/∂x_b = H(e,b) / Hs(e)
                [~, cols, vals] = find(H(e, :)); 
                
                for idx = 1:length(cols)
                    b = cols(idx);         
                    H_eb = vals(idx);      

                    dsi_dx(i, b) = dsi_dx(i, b) + dsi_drho_e * H_eb / Hs(e);
                end
            else
                dsi_dx(i, e) = dsi_drho_e;
            end
        end
    end
end