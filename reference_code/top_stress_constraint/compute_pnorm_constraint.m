function [sigmapn, fval, dsi_dvm] = compute_pnorm_constraint( ...
                                        cluster_vm, cluster_idx, ...
                                        Ni, sigmay, p, nele, nc)

    % 计算归一化 P-norm 应力约束值
    sigmapn = zeros(nc, 1);
    for i = 1:nc
        cluster_p = (cluster_vm{i} / sigmay).^p;
        sigmapn(i) = (sum(cluster_p) / Ni(i))^(1/p);
    end
    
    % 约束函数值 g_i <= 0
    fval = sigmapn - 1;
    
    % 计算 P-norm 约束对单个 von Mises 应力的导数 d(s_i) / d(sigma_a^vM)
    dsi_dvm = zeros(nc, nele);
    for i = 1:nc
        if sigmapn(i) > 1e-12  % 避免除零
            term_A = sigmapn(i)^(1-p);
            term_B = 1 / (Ni(i) * sigmay^p);
            stress_values = cluster_vm{i};
            dsi_dvm(i, cluster_idx{i}) = term_A * term_B * (stress_values.^(p-1));
        end
    end
end