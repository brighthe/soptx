function rhs = compute_adjoint_rhs(cluster_idx, dsi_dvm, dvm_dstress, B, D, nely, nelx)

    nc = length(cluster_idx);
    ndof = 2*(nely+1)*(nelx+1);
    rhs = zeros(ndof, nc);
    
    % 对每个聚类计算 RHS
    for i = 1:nc
        elements = cluster_idx{i};
        
        for elem_idx = elements'
            % 将线性索引转换为 (elx, ely)
            ely = mod(elem_idx - 1, nely) + 1;
            elx = floor((elem_idx - 1) / nely) + 1;
            
            % 节点索引
            n1 = (nely+1)*(elx-1) + ely;
            n2 = (nely+1)*elx + ely;
            edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
            
            % 计算：∂s_i/∂σ^vM * ∂σ^vM/∂σ (1×3)
            grad_term = dsi_dvm(i, elem_idx) * dvm_dstress(elem_idx, :);
            
            % B^T * E^T * [∂s_i/∂σ^vM * ∂σ^vM/∂σ]^T
            % B: (3×8), E: (3×3), grad_term': (3×1)
            rhs_elem = B' * (D' * grad_term');  % (8×1)
            
            % 组装到全局 RHS
            rhs(edof, i) = rhs(edof, i) + rhs_elem;
        end
    end
end