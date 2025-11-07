clc; clear;
% --- 问题定义 ---
% (P1) 质量最小化 + 应力约束 

% 网格参数
nelx = 120;
nely = 40;

% 几何参数
l = 300; % [mm]
b = 100; % [mm]
t = 1.0; % [mm]
unit_size_x = l / nelx;
unit_size_y = b / nely;
% 过滤半径
rmin = 2.0 * unit_size_x;
% 材料参数
E = 71000; % [MPa]
nu = 0.33;
density = 2.8e-9; % [ton/mm^3]

% --- 应力约束参数 ---
% 屈服应力限制
sigmay = 350; % [MPa]
% p-norm 聚合参数
p = 8;
% 应力约束数量
nc = 10;

% --- 质量约束参数 ---
element_area = unit_size_x * unit_size_y;
element_volume = element_area * t;
% 单个实体单元质量
m_e = density * element_volume; % [ton]

% 优化参数
penal = 3;

% --- *** 新增：过滤和循环控制 *** ---
ft = 1; % 0 = 不过滤, 1 = 密度过滤
maxite = 200; % 最大迭代次数
tolx = 0.01;  % 收敛标准
change = 1.0; % 变化量
% ---

% 初始设计
n = nelx * nely; % 设计变量数量
x_design = 0.5 * ones(n, 1); % 初始设计 (1D 列向量)
xPhys = x_design;            % 物理密度 (初始迭代)

% 准备过滤 (列优先索引)
iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
  for j1 = 1:nely
    e1 = (i1-1)*nely+j1; 
    for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
      for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
        e2 = (i2-1)*nely+j2;
        k = k+1;
        iH(k) = e1;
        jH(k) = e2;
        sH(k) = max(0, rmin - sqrt((i1-i2)^2 + (j1 - j2)^2));
      end
    end
  end
end
H = sparse(iH, jH, sH);
Hs = sum(H, 2);

% 单元刚度矩阵
k = [ 1/2-nu/6   1/8+nu/8 -1/4-nu/12 -1/8+3*nu/8 ... 
     -1/4+nu/12 -1/8-nu/8  nu/6       1/8-3*nu/8];
KE = 1/(1-nu^2)*[ k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
                  k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
                  k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
                  k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
                  k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
                  k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
                  k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
                  k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)]; 
% 应变位移矩阵 (单元中心点)
B = (1/2/unit_size_x) * [-1  0  1  0  1  0 -1  0; 
                          0 -1  0 -1  0  1  0  1; 
                         -1 -1 -1  1  1  1  1 -1];
% 本构矩阵
C = E/(1-nu^2)*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];

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
    
    % --- *** 新增：应用密度过滤器 *** ---
    if ft == 1
        xPhys(:) = (H*xval)./Hs;
    else % ft == 0
        xPhys(:) = xval;
    end
    % 将 1D 列优先向量重塑为 2D 矩阵
    xPhys_2d = reshape(xPhys, [nely, nelx]);
    
    % --- 位移求解 ---
    K = sparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1));
    F = sparse(2*(nely+1)*(nelx+1),1); 
    U = zeros(2*(nely+1)*(nelx+1),1);

    % *** 修正：使用 xPhys_2d ***
    for elx = 1:nelx
      for ely = 1:nely
        n1 = (nely+1)*(elx-1)+ely; 
        n2 = (nely+1)* elx   +ely;
        edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
        K(edof, edof) = K(edof, edof) + xPhys_2d(ely, elx)^penal*KE;
      end
    end

    % --- 定义 MBB 梁的载荷和边界条件 ---
    load_node = (nely+1); 
    F(2*load_node, 1) = -1500.0; % [N]
    fixeddofs_x = 1:2:2*(nely+1);
    node_bottom_right = (nelx)*(nely+1) + 1;
    fixeddofs_y = 2*node_bottom_right;
    fixeddofs = unique([fixeddofs_x, fixeddofs_y]);
    alldofs     = [1:2*(nely+1)*(nelx+1)];
    freedofs    = setdiff(alldofs, fixeddofs);
    
    % SOLVING
    U(freedofs, :) = K(freedofs,freedofs) \ F(freedofs,:);      
    U(fixeddofs, :) = 0;

    % --- 计算质量目标函数和应力约束函数 ---
    
    % *** 修正：质量目标函数 (使用 xPhys) ***
    f0val = sum(xPhys * m_e);
    
    % *** 修正：质量目标函数灵敏度 (df0/drho) ***
    df0drho = m_e * ones(n, 1);
    
    % *** 新增：过滤质量灵敏度 ***
    if ft == 1
        df0dx = H' * (df0drho ./ Hs); % H' = H
    else % ft == 0
        df0dx = df0drho;
    end
    
    % --- 计算所有单元的 von Mises 应力 ---
    von_mises = zeros(n, 1);
    solid_stresses = zeros(n, 3);
    penalized_stresses = zeros(n, 3);
    
    % *** 修正：循环改为 "列优先" (elx 在外) ***
    for elx = 1:nelx
        for ely = 1:nely
            % *** 修正：'stress_counter' 现在是列优先索引 ***
            stress_counter = (elx-1)*nely + ely; 

            n1 = (nely+1)*(elx-1)+ely; 
            n2 = (nely+1)* elx   +ely;
            Ue = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1);
            
            % 单元的实体材料应力
            solid_stress_e = C*B*Ue;
            solid_stresses(stress_counter, :) = solid_stress_e;
            
            % *** 修正：使用 xPhys_2d ***
            penalized_stress_e = (xPhys_2d(ely, elx)^0.5) * solid_stress_e;
            penalized_stresses(stress_counter, :) = penalized_stress_e;
            
            % von Mises 等效应力 (平面应力)
            % *** 修正：拼写错误 (penalized_stress -> penalized_stress_e) ***
            von_mises(stress_counter, 1) = ...
                sqrt(penalized_stress_e(1, 1)^2 + penalized_stress_e(2, 1)^2 + ...
                    3*penalized_stress_e(3, 1)^2 - penalized_stress_e(1, 1)*penalized_stress_e(2, 1));
        end
    end
    
    % 聚类
    [von_mises_desc, sort_index] = sort(von_mises, 'descend');
    Ni = n/nc; % 每个簇中的单元数
    % 应力水平法
    cluster_vm = reshape(von_mises_desc, [Ni, nc])';
    % 归一化 P-norm 应力约束
    fval = zeros(m, 1); % m=nc
    cluster_p = (cluster_vm / sigmay).^p;
    cluster_sum = sum(cluster_p, 2);
    cluster_mean = cluster_sum ./ Ni;
    sigmapn = (cluster_mean.^(1/p));
    fval = sigmapn - 1; % [nc, 1] 约束函数值 f(x) = sigmapn - 1 <= 0

    % --- 归一化 P-norm 应力约束对单个 von Mises 应力的导数 d(s_i) / d(sigma_a^vM) ---
    dsi_dvm = zeros(nc, n); 
    term_A = (1/p) * (sigmapn.^ (1-p)); % 使用 sigmapn 简化
    term_B_factor = (p / (Ni * (sigmay^p)));
    for i = 1:nc  % 循环 nc 个簇
        for j = 1:Ni  % 循环簇 i 中的每个应力点
            original_index = sort_index((i-1)*Ni + j);
            sigma_a_vM = cluster_vm(i, j); 
            dsi_dvm(i, original_index) = term_A(i) * term_B_factor * (sigma_a_vM^(p-1));
        end       
    end

    % --- von Mises 应力对其应力分量的导数 d(sigma_a^vM) / d(sigma_a) ---
    dvm_dstress = zeros(n, 3);
    vm_safe = max(von_mises, 1e-12); 
    sig_x = penalized_stresses(:, 1);
    sig_y = penalized_stresses(:, 2);
    tau_xy = penalized_stresses(:, 3);
    dvm_dstress(:, 1) = (2*sig_x - sig_y) ./ (2*vm_safe); 
    dvm_dstress(:, 2) = (2*sig_y - sig_x) ./ (2*vm_safe);  
    dvm_dstress(:, 3) = (3*tau_xy)        ./ (vm_safe);

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