function dvm_dstress = compute_dvm_dsigma(stress_vm, stress_penalized)

    nele = length(stress_vm);
    dvm_dstress = zeros(nele, 3);
    
    vm_safe = max(stress_vm, 1e-12);
    
    % 提取应力分量
    sig_x = stress_penalized(:, 1);
    sig_y = stress_penalized(:, 2);
    tau_xy = stress_penalized(:, 3);
    
    % 计算导数
    dvm_dstress(:, 1) = (2*sig_x - sig_y) ./ (2*vm_safe);  % ∂σ^vM/∂σ_x
    dvm_dstress(:, 2) = (2*sig_y - sig_x) ./ (2*vm_safe);  % ∂σ^vM/∂σ_y
    dvm_dstress(:, 3) = (3*tau_xy) ./ vm_safe;             % ∂σ^vM/∂τ_xy
end