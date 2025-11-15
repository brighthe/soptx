clc; clear;
% --- 问题定义 ---
% (P1) 质量最小化 + 应力约束 

%% --- 输入参数准备 --- 
% --- 网格参数 ---
nelx = 120;
nely = 40;
nele = nelx * nely;

% --- 几何参数 ---
l = 300; % [mm]
b = 100; % [mm]
t = 1.0; % [mm]
unit_size_x = l / nelx;
unit_size_y = b / nely;

% --- 材料参数 ---
E = 71000;        % [MPa]
nu = 0.33;
density = 2.8e-9; % [ton/mm^3]

% --- 过滤参数 ---
ft = 2; % 0: 无过滤, 1: 灵敏度过滤, 2: 密度过滤
rmin = 2; % 索引空间中的两个单元

% --- 应力约束参数 ---
sigmay = 350; % 屈服应力限制 [MPa]
p = 8;        % p-norm 聚合参数
nc = 10;      % 应力约束数量

% --- 质量约束参数 ---
element_area = unit_size_x * unit_size_y;
element_volume = element_area * t;
m_e = density * element_volume; % 单个实体单元质量 [ton]

%% --- 定义不参与优化的单元（强制为实体）---
passive_elements = false(nely, nelx);

% 载荷施加在左上角，排除载荷点周围的单元
% 论文建议 3×2 个单元
load_elx = 1;
load_ely_top = nely;  % 40

% === 1. 排除载荷点附近单元（左上角）===
for ex = 1:min(3, nelx)
    for ey = max(1, nely-2):nely
        passive_elements(ey, ex) = true;
    end
end

% === 2. 排除支座附近单元（右下角）===
for ex = max(1, nelx-2):nelx
    for ey = 1:min(3, nely)
        passive_elements(ey, ex) = true;
    end
end

fprintf('Passive elements (fixed as solid): %d (%.1f%%)\n', ...
        sum(passive_elements(:)), 100*sum(passive_elements(:))/nele);

%% --- 施加过滤 ---
% 过滤矩阵
iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
  for j1 = 1:nely
    e1 = (i1-1)*nely+j1;
    for i2 = max(i1-(ceil(rmin)-1), 1):min(i1+(ceil(rmin)-1), nelx)
      for j2 = max(j1-(ceil(rmin)-1), 1):min(j1+(ceil(rmin)-1), nely)
        e2 = (i2-1)*nely+j2;
        k = k+1;
        iH(k) = e1;
        jH(k) = e2;
        sH(k) = max(0, rmin - sqrt((i1-i2)^2 + (j1-j2)^2));
      end
    end
  end
end
H = sparse(iH, jH, sH);
Hs = sum(H, 2);

x = repmat(0.5, [nely, nelx]); % 初始设计变量
xPhys = x; % 初始单元密度

if ft == 2
    xPhys(:) = (H * x(:)) ./ Hs;
    xPhys(passive_elements) = 1.0;  % 过滤后也要强制
elseif ft == 0 || ft == 1
    xPhys = x;
end

%% --- 有限元分析参数 ---
% 局部刚度矩阵
k = [ 1/2-nu/6   1/8+nu/8 -1/4-nu/12 -1/8+3*nu/8 ... 
     -1/4+nu/12 -1/8-nu/8  nu/6       1/8-3*nu/8];
KE = E/(1-nu^2)*[ k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
                  k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
                  k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
                  k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
                  k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
                  k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
                  k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
                  k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)]; 
% SIMP 参数
penal_K = 3; 
% 应力罚函数参数
penal_S = 0.5; 

% 应变位移矩阵 (单元中心点)
h = unit_size_x;
B = 1/(2*h) * [-1  0  1  0  1  0 -1  0; 
                0 -1  0 -1  0  1  0  1; 
               -1 -1 -1  1  1  1  1 -1];
% 本构矩阵
D = E/(1-nu^2)*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];

%% --- MMA 参数初始化 ---
maxiter = 2000;
change = 1;
loop = 0;

m = nc;
n = nelx * nely;
xmin = 0.001 * ones(n, 1);
xmax = 1.0 * ones(n, 1);
xold1 = x(:);
xold2 = x(:);
low = xmin;
upp = xmax;

% 设置重新聚类频率 
recluster_freq = 50;  % 1, 10, 50, 100, 或 inf (不重新聚类)

%% --- 优化迭代循环 ---
while change > 0.01 && loop < maxiter
    loop = loop + 1;

    % 强制被动单元为实体
    x(passive_elements) = 1.0;
    
    % --- 有限元分析 ---
    [F, U, K, fixeddofs] = FEA(nelx, nely, xPhys, penal_K, KE);

    % --- 质量目标函数及灵敏度 ---
    % 质量目标函数
    f0val = sum(sum(m_e * xPhys));
    % 质量对物理密度的灵敏度 ∂f0/∂rho
    df0drho = m_e * ones(nely, nelx);
    df0dx = df0drho;
    % 应用过滤得到对设计变量的敏感度 ∂f0/∂x
    if ft == 1
        df0dx(:) = H * (x(:) .* df0drho(:)) ./ Hs ./ max(1e-3, x(:));
    elseif ft == 2
        df0dx(:) = H * (df0drho(:) ./ Hs);
    end
    
    % --- 应力计算 ---
    % 初始 von Mises 应力计算
    [stress_vm, stress_penalized] = compute_von_mises(nelx, nely, xPhys, U, penal_S, B, D);

    % 应力聚类 (水平法)
    if mod(loop, recluster_freq) == 1
        % 重新聚类
        [cluster_idx, cluster_vm, Ni] = stress_clustering(stress_vm, nc, nele);
    else
        % 更新现有聚类
        [cluster_idx, cluster_vm, Ni] = stress_clustering(stress_vm, nc, nele, cluster_idx);
    end

    % 归一化 P-norm 应力约束及其灵敏度
    [sigmapn, fval, dsi_dvm] = compute_pnorm_constraint(cluster_vm, cluster_idx, Ni, sigmay, p, nele, nc);
    
    % von Mises 应力对其应力分量的导数 d(sigma_a^vM) / d(sigma_a)
    dvm_dstress = compute_dvm_dsigma(stress_vm, stress_penalized);
    
    % --- 伴随法计算应力分量对设计变量的导数 ---
    % 计算伴随方程右端项
    rhs_adjoint = compute_adjoint_rhs(cluster_idx, dsi_dvm, dvm_dstress, B, D, nely, nelx);
    % 求解伴随方程
    lambda = FEA_adjoint(K, rhs_adjoint, fixeddofs);
    % 计算应力约束对物理密度的灵敏度 d(s_i) / drho
    dsi_drho = compute_stress_sensitivity(xPhys, U, cluster_idx, dsi_dvm, ...
                                         dvm_dstress, lambda, B, D, KE, ...
                                         penal_S, penal_K, nelx, nely);

    % --- 应力灵敏度过滤 ---
    dsi_dx = zeros(nc, nele);
    if ft == 1
        for i = 1:nc
            dsi_dx_i = reshape(dsi_drho(i, :), nely, nelx);
            dsi_dx_i(:) = H * (x(:) .* dsi_dx_i(:)) ./ Hs ./ max(1e-3, x(:));
            dsi_dx(i, :) = dsi_dx_i(:);
        end
    elseif ft == 2
        for i = 1:nc
            dsi_dx_i = reshape(dsi_drho(i, :), nely, nelx);
            dsi_dx_i(:) = H * (dsi_dx_i(:) ./ Hs);
            dsi_dx(i, :) = dsi_dx_i(:);
        end
    else
        dsi_dx = dsi_drho;
    end

    % 在应力计算后添加
    if loop == 1
        fprintf('\n=== Detailed Stress Analysis (Iteration 1) ===\n');
        
        % 应力统计
        fprintf('von Mises stress statistics:\n');
        fprintf('  Min:  %.1f MPa\n', min(stress_vm));
        fprintf('  Max:  %.1f MPa\n', max(stress_vm));
        fprintf('  Mean: %.1f MPa\n', mean(stress_vm));
        fprintf('  Median: %.1f MPa\n', median(stress_vm));
        
        % 高应力单元统计
        high_stress = stress_vm > 500;
        fprintf('  Elements with σ > 500 MPa: %d (%.1f%%)\n', ...
                sum(high_stress), 100*sum(high_stress)/nele);
        
        very_high_stress = stress_vm > 1000;
        fprintf('  Elements with σ > 1000 MPa: %d (%.1f%%)\n', ...
                sum(very_high_stress), 100*sum(very_high_stress)/nele);
        
        % P-norm详情
        fprintf('\nP-norm constraint details:\n');
        for i = 1:nc
            fprintf('  Cluster %2d: σ^PN = %.0f MPa (%.2f×σ̄), Ni = %d, fval = %+.3f\n', ...
                    i, sigmapn(i)*sigmay, sigmapn(i), Ni(i), fval(i));
        end
        
        % 找出最高应力的位置
        [max_stress, max_idx] = max(stress_vm);
        ely_max = mod(max_idx-1, nely) + 1;
        elx_max = floor((max_idx-1) / nely) + 1;
        fprintf('\nMax stress location: element (%d, %d), σ = %.0f MPa, ρ = %.3f\n', ...
                elx_max, ely_max, max_stress, xPhys(ely_max, elx_max));
    end

    % --- 调用 MMA ---
    xval = x(:);              % 当前设计变量
    f0val_mma = f0val;        % 目标函数值
    df0dx_mma = df0dx(:);     % 目标函数梯度 (n×1)
    fval_mma = fval;          % 约束函数值 (m×1)
    dfdx_mma = dsi_dx;        % 约束函数梯度 (m×n)

    % MMA 参数
    a0 = 1.0;
    a = zeros(m, 1);     
    c = 1000*ones(m, 1); 
    d = zeros(m, 1);     
    
    [xmma, ~, ~, ~, ~, ~, ~, ~, ~, low, upp] = ...
        mmasub(m, n, loop, xval, xmin, xmax, xold1, xold2, ...
               f0val_mma, df0dx_mma, fval_mma, dfdx_mma, low, upp, ...
               a0, a, c, d);

    % --- 更新设计变量 ---
    xold2 = xold1;
    xold1 = xval;
    x(:) = xmma;

    % 应用过滤
    if ft == 2
        xPhys(:) = (H * x(:)) ./ Hs;
        xPhys(passive_elements) = 1.0;  % 过滤后也要强制
    elseif ft == 0 || ft == 1
        xPhys = x;
    end
    
    % --- 收敛判断 ---
    change = max(abs(xmma - xold1));
    
    % --- 输出 ---
    fprintf('It:%4i Obj:%7.3e      Vol:%7.3f      ch:%6.3f MaxCon:%6.3f\n', ...
              loop,   f0val,   sum(xPhys(:))/nele, change,   max(fval));
    
    % --- 可视化 ---
    colormap(gray); imagesc(1-xPhys); caxis([0 1]); axis equal; axis off; axis xy; drawnow;


end