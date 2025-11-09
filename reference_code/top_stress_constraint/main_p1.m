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
rmin = 1.5;

% --- 应力约束参数 ---
sigmay = 350; % 屈服应力限制 [MPa]
p = 8;        % p-norm 聚合参数
nc = 10;      % 应力约束数量

% --- 质量约束参数 ---
element_area = unit_size_x * unit_size_y;
element_volume = element_area * t;
m_e = density * element_volume; % 单个实体单元质量 [ton]

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
    xPhys(:) = (H*x(:))./Hs;
elseif ft == 0 || ft == 1
    xPhys = x;
end

%% --- 有限元分析 ---
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

penal_K = 3; % SIMP 参数

% 初始位移计算
[F, U, K, fixeddofs] = FEA(nelx, nely, xPhys, penal_K, KE);

%% --- 应力处理 ---
penal_S = 0.5; % 应力罚函数

% 应变位移矩阵 (单元中心点)
h = unit_size_x;
B = 1/(2*h) * [-1  0  1  0  1  0 -1  0; 
                0 -1  0 -1  0  1  0  1; 
               -1 -1 -1  1  1  1  1 -1];
% 本构矩阵
D = E/(1-nu^2)*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];

% --- 初始 von Mises 应力计算 ---
[stress_vm, stress_penalized] = compute_von_mises(nelx, nely, xPhys, U, penal_S, B, D);

% --- 应力聚类 (水平法) --- 
[stress_vm_sorted, sort_idx] = sort(stress_vm, 'descend');
% 计算每个聚类的单元数
base_size = floor(nele / nc);
remainder = mod(nele, nc);
Ni = repmat(base_size, nc, 1);
Ni(1:remainder) = Ni(1:remainder) + 1;
% 分配单元到聚类
cluster_vm = cell(nc, 1);
cluster_idx = cell(nc, 1);
start_pos = 1;
for i = 1:nc
    end_pos = start_pos + Ni(i) - 1;
    cluster_idx{i} = sort_idx(start_pos:end_pos);
    cluster_vm{i} = stress_vm_sorted(start_pos:end_pos);
    start_pos = end_pos + 1;
end

% --- 归一化 P-norm 应力约束及其梯度 ---
% 归一化 P-norm 应力约束值
sigmapn = zeros(nc, 1);
for i = 1:nc
    cluster_p = (cluster_vm{i} / sigmay).^p;
    sigmapn(i) = (sum(cluster_p) / Ni(i))^(1/p);
end
fval = sigmapn - 1;
% 归一化 P-norm 应力约束对单个 von Mises 应力的导数 d(s_i) / d(sigma_a^vM)
dsi_dvm = zeros(nc, nele);
for i = 1:nc
    term_A = sigmapn(i)^(1-p);
    term_B = 1 / (Ni(i) * sigmay^p);
    stress_values = cluster_vm{i};
    dsi_dvm(i, cluster_idx{i}) = term_A * term_B * (stress_values.^(p-1));
end

% --- von Mises 应力对其应力分量的导数 d(sigma_a^vM) / d(sigma_a)  ---
dvm_dstress = compute_dvm_dsigma(stress_vm, stress_penalized);

% --- 伴随法计算应力分量对设计变量的导数 ---
% 计算伴随方程右端项
rhs_adjoint = compute_adjoint_rhs(cluster_idx, dsi_dvm, dvm_dstress, B, D, nely, nelx);

% 求解伴随方程
lambda = FEA_adjoint(K, rhs_adjoint, fixeddofs);

% 计算应力约束敏感度 d(s_i) / dx
dsi_dx = compute_stress_sensitivity(xPhys, U, cluster_idx, dsi_dvm, ...
                                   dvm_dstress, lambda, B, D, KE, ...
                                   penal_S, penal_K, H, Hs, nelx, nely, ft);

maxite = 200; % 最大迭代次数
tolx = 0.01;  % 收敛标准
change = 1.0; % 变化量


% --- MMA 初始化 (第 1 部分) ---
m = nc; % m 个应力约束
xval = x_design; % 当前的设计变量

% --- MMA 初始化 (第 2 部分) ---
xold1 = zeros(n, 1);
xold2 = zeros(n, 1);
low = zeros(n, 1);
upp = zeros(n, 1);
xmin = 1e-3 * ones(n, 1);
xmax = ones(n, 1);
c = 1000 * ones(m, 1); 
d = zeros(m, 1);                
a = zeros(m, 1);       
a0 = 1;                
df0dx2 = zeros(n, 1);
dfdx2 = zeros(m, n);

% 迭代计数器
iter = 0;

% --- *** 新增：开始主循环 *** ---
while change > tolx && iter < maxite
    
    iter = iter + 1;
    

    % --- 伴随法 (Adjoint Method) 灵敏度分析 ---
    
    % *** 更改 ***: 计算对 "物理密度 rho" 的导数
    dfdrho = zeros(m, n); 
    
    KE_solid = KE; 
    CB = C*B;
    BTC = (CB)';

    for i = 1:nc % --- 循环 nc 个应力约束 ---
        
        % --- 构造伴随载荷 R_i ---
        R_i = sparse(2*(nely+1)*(nelx+1), 1);
        
        % *** 修正：循环改为 "列优先" ***
        for elx = 1:nelx
            for ely = 1:nely
                % 'a' 是列优先索引
                a = (elx-1)*nely + ely; 
                
                val_dsi_dvm = dsi_dvm(i, a);
                if val_dsi_dvm == 0
                    continue;
                end
                
                val_dvm_dstress = dvm_dstress(a, :)';
                Re = BTC * (val_dvm_dstress * val_dsi_dvm);
                
                n1 = (nely+1)*(elx-1)+ely;
                n2 = (nely+1)* elx   +ely;
                edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
                R_i(edof) = R_i(edof) + Re;
            end
        end
        
        % --- 求解伴随方程 K * lambda_i = R_i ---
        lambda_i = zeros(2*(nely+1)*(nelx+1), 1);
        lambda_i(freedofs, :) = K(freedofs, freedofs) \ R_i(freedofs, :);
        lambda_i(fixeddofs, :) = 0;
        
        % --- *** 新增：计算 T1 - T2 (对 rho 的导数) *** ---
        % *** 修正：循环改为 "列优先" ***
        for elx = 1:nelx
            for ely = 1:nely
                % 'b' 是列优先索引
                b = (elx-1)*nely + ely;
                rhob = xPhys_2d(ely, elx); % 物理密度
                
                % --- T1 (局部项) ---
                val_dsi_dvm_b = dsi_dvm(i, b);
                if val_dsi_dvm_b == 0
                    T1 = 0;
                else
                    d_eta_s = 0.5 * (rhob^(-0.5));
                    val_dvm_dstress_b = dvm_dstress(b, :);
                    solid_sigma_b = solid_stresses(b, :)';
                    T1 = val_dsi_dvm_b * (val_dvm_dstress_b * solid_sigma_b) * d_eta_s;
                end
                
                % --- T2 (全局项) ---
                dK_drho = penal * (rhob^(penal-1));
                
                n1 = (nely+1)*(elx-1)+ely;
                n2 = (nely+1)* elx   +ely;
                edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
                
                Ue = U(edof);
                lambda_e = lambda_i(edof);
                eta_s = rhob^0.5;
                
                T2 = eta_s * dK_drho * (lambda_e' * KE_solid * Ue);
                
                % 存储 (未过滤的) 灵敏度
                dfdrho(i, b) = T1 - T2;
            end
        end
    end % 结束 nc 个约束循环

    % --- *** 新增：过滤应力灵敏度 *** ---
    dfdx = zeros(m, n); % 最终传递给 MMA 的 "过滤后" 的灵敏度
    if ft == 1
        for i = 1:m
            unfiltered_sens = dfdrho(i, :)';
            filtered_sens = H * (unfiltered_sens ./ Hs);
            dfdx(i, :) = filtered_sens';
        end
    else % ft == 0
        dfdx = dfdrho;
    end
    
    % --- *** 新增：调用 MMA 求解器 *** ---
    % m = nc (约束数)
    % n = nelx*nely (设计变量数)
    % f0val = 质量
    % df0dx = 质量灵敏度 (n x 1)
    % fval = 应力约束 (m x 1)
    % dfdx = 应力灵敏度 (m x n)
    
    [xmma, ~, ~, ~, ~, ~, ~, ~, ~, low, upp] = ...
            mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2, ...
            f0val, df0dx, df0dx2, fval, dfdx, dfdx2, ...
            low, upp, a0, a, c, d);
    % --- *** 新增：更新设计变量 *** ---
    xold2 = xold1;
    xold1 = xval;
    xval = xmma; % 更新设计变量
    
    change = max(abs(xval - xold1));
    
    % --- 打印和绘图 ---
    fprintf(' It.:%5i Obj.:%11.4f MaxCons.:%7.3f ch.:%7.3f\n', ...
        iter, f0val, max(fval), change);
    
    % 绘图
    figure(1);
    colormap(gray); 
    imagesc(1-xPhys_2d); 
    caxis([0 1]); 
    axis equal; 
    axis off; 
    drawnow;
    
end % --- 结束主循环 ---