clc; clear;
% --- 问题定义 ---
% (P1) 质量最小化 + 应力约束 

%% --- 输入参数准备 --- 
% --- 网格参数 ---
nelx = 80;
nely = 80;
nele = nelx * nely;

% --- 几何参数 ---
l = 200; % [mm]
b = 200; % [mm]
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

%% --- 定义不参与优化的单元（L形梁）---
passive_elements = false(nely, nelx);

% === 排除载荷和支座附近的单元 ===

% 1. 载荷点附近（右侧中间）
load_x_pos = nelx;
load_y_pos = round(nely * 2/5);
for ex = max(1, load_x_pos-2):nelx
    for ey = max(1, load_y_pos-1):min(nely, load_y_pos+1)
        passive_elements(ey, ex) = true;
    end
end

% 2. 支座附近（左上角）
for ex = 1:3
    for ey = max(1, round(nely*3/5)):nely
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

% === 定义L形的初始几何 ===
% 右上角设为空洞
void_x_start = round(nelx * 2/5) + 1;
void_y_start = round(nely * 2/5) + 1;
for ex = void_x_start:nelx
    for ey = void_y_start:nely
        x(ey, ex) = 0.001;  % 设为接近0（空洞）
    end
end

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
maxiter = 1000;
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

obj_history = zeros(maxiter, 1);          % 质量历史
change_history = zeros(maxiter, 1);       % 变化量历史
constraint_history = zeros(maxiter, 1);   % 约束历史
compliance_history = zeros(maxiter, 1);   % 柔顺度历史

% --- 创建可视化窗口 ---
fig = figure('Position', [100, 100, 1200, 400]);

%% --- 优化迭代循环 ---
while change > 0.01 && loop < maxiter
    loop = loop + 1;

    % 强制被动单元为实体
    x(passive_elements) = 1.0;
    
    % --- 有限元分析 ---
    [F, U, K, fixeddofs] = FEA_LBeam(nelx, nely, xPhys, penal_K, KE);

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

    % --- 柔顺度 ---
    compliance = 0.5 * F' * U;
    
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

    obj_history(loop) = f0val;
    change_history(loop) = change;
    constraint_history(loop) = max(fval);
    compliance_history(loop) = compliance;  
    
    % --- 输出 ---
    fprintf('It:%4i Mass:%7.3e Vol:%7.3f Comp:%8.0f ch:%6.3f MaxCon:%6.3f\n', ...
              loop, f0val, sum(xPhys(:))/nele, compliance, change, max(fval));

    % 将应力重塑为矩阵形式
    stress_matrix = reshape(stress_vm, nely, nelx);
    
    % 子图 1: 拓扑
    subplot(1, 2, 1);
    imagesc(1-xPhys);
    colormap(gca, gray);
    caxis([0 1]);
    axis equal; axis tight; axis off; axis xy;
    title(sprintf('Topology (Iter %d)', loop), 'FontSize', 12);
    
    % 子图 2: 应力
    subplot(1, 2, 2);
    imagesc(stress_matrix);
    colormap(gca, jet);
    caxis([0 600]);  % 应力范围 0-600 MPa
    axis equal; axis tight; axis off; axis xy;
    title(sprintf('von Mises Stress [MPa] (Iter %d)', loop), 'FontSize', 12);
    colorbar;

    drawnow;

end

%% --- 简单的最终结果输出 ---
fprintf('\n=== Final Results ===\n');
fprintf('Iterations: %d\n', loop);
fprintf('Mass: %.2f kg×10⁻³\n', f0val*1e6);
fprintf('Compliance: %.0f N·mm\n', compliance_history(loop)); 

%% --- 绘制收敛曲线 ---
iterations = 1:loop;
mass_hist_kg = obj_history(1:loop) * 1e6;  % 转换为 kg×10⁻³
comp_hist = compliance_history(1:loop);

figure('Position', [100, 100, 800, 500]);

% 质量收敛
subplot(2, 1, 1);
plot(iterations, mass_hist_kg, 'b-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Mass [kg × 10^{-3}]');
title('Mass Convergence');
grid on;

% 柔顺度收敛
subplot(2, 1, 2);
plot(iterations, comp_hist, 'g-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Compliance [N·mm]');
title('Compliance Convergence');
grid on;

% 保存
saveas(gcf, 'convergence.png');
fprintf('Convergence plot saved to: convergence.png\n');

