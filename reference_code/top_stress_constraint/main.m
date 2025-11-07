clc; clear;

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

% --- CHOOSE PROBLEM TO SOLVE ---
% 1 = (P1) 质量最小化 + 应力约束 
% 2 = (P2) 柔顺度最小化 + 应力约束 + 质量约束 
% 3 = (P3) 柔顺度最小化 + 质量约束 
% 4 = (P4) 柔顺度最小化 + 应力约束 + 体积分数约束 
ProblemID = 1;

% --- 应力约束参数 ---
% 屈服应力限制
sigmay = 350; % [MPa]
% p-norm 聚合参数
p = 8;
% 应力约束数量
nc = 10;

% --- 质量约束参数 ---
% 最大质量
Mbar = 3e-5; % [kg*1e-3];
element_area = unit_size_x * unit_size_y;
element_volume = element_area * t;
% 单个实体单元质量
m_e = density * element_volume; % [ton]

% --- 体积分数约束参数 ---
volfrac = 0.5;

% 优化参数
penal = 3;
% 初始设计
x(1:nely, 1:nelx) = 0.5;

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
% 设计变量数量
n = nelx * nely;
% 设计变量按行展开
count = 1;
for g=1:nely
    for h=1:nelx
        xval(count, 1) = x(g, h);
        count = count + 1;
    end
end

% --- 根据 ProblemID 设置约束数量 m ---
switch ProblemID
    case 1
        % P1: nc 个应力约束
        m = nc;
    case 2
        % P2: nc 个应力约束 + 1 个质量约束
        m = nc + 1;
    case 3
        % P3: 1 个质量约束
        m = 1;
    case 4
        % P4: 柔顺度最小化 + 应力约束 + 体积分数约束 
        m = nc + 1;
end

% --- MMA 初始化 (第 2 部分) ---
% 历史信息
xold1 = zeros(n, 1);
xold2 = zeros(n, 1);
% 移动渐近线
low = zeros(n, 1);
upp = zeros(n, 1);
% 设计变量界限
xmin = 1e-3 * ones(n, 1);
xmax = ones(n, 1);
% MMA 子问题参数
c = 1000 * ones(m, 1); % 松弛变量 yi 的惩罚系数
d = 0;                 % 二次惩罚项
a = zeros(m, 1);       % 约束中 z 的系数
a0 = 1;                % 目标函数中 z 的系数
% 二阶导数
df0dx2 = zeros(n, 1);
dfdx2 = zeros(m, n);
% 迭代计数器
iter = 0;
maxite = 120;
x = reshape(xval, [nelx, nely])';

% ---位移求解 ---
[F, U, K] = FEA(nelx, nely, x, penal, KE);

% --- 根据 ProblemID 计算目标函数和约束函数 ---
switch ProblemID
    case 1
        % P1: 质量最小化 + 应力约束
        % 质量目标函数
        f0val = 0.;
        for ely = 1:nely
            for elx = 1:nelx
                f0val = f0val + x(ely, elx) * m_e;
            end
        end
        % 质量目标函数灵敏度
        df0dx = m_e * ones(n, 1);
        % 计算所有单元的 von Mises 应力
        von_mises = zeros(n, 1);
        solid_stresses = zeros(n, 3);
        penalized_stresses = zeros(n, 3);
        stress_counter = 1;
        for ely = 1:nely
            for elx = 1:nelx
                n1 = (nely+1)*(elx-1)+ely; 
                n2 = (nely+1)* elx   +ely;
                Ue = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1);
                % 单元的实体材料应力
                solid_stress_e = C*B*Ue;
                solid_stresses(stress_counter, :) = solid_stress_e;
                % 罚函数应力
                penalized_stress_e = (x(ely, elx)^0.5) * solid_stress_e;
                penalized_stresses(stress_counter, :) = penalized_stress_e;
                % von Mises 等效应力 (平面应力)
                von_mises(stress_counter, 1) = ...
                    sqrt(penalized_stress(1, 1)^2 + penalized_stress(2, 1)^2 + ...
                        3*penalized_stress(3, 1)^2 - penalized_stress(1, 1)*penalized_stress(2, 1));
                stress_counter = stress_counter + 1;       
            end
        end
        % 聚类
        [von_mises_desc, sort_index] = sort(von_mises, 'descend');
        Ni = n/nc; % 每个簇中的单元数
        % 应力水平法
        cluster_vm = reshape(von_mises_desc, [Ni, nc])';
        % 归一化 P-norm 应力约束
        cluster_p = (cluster_vm / sigmay).^p;
        cluster_sum = sum(cluster_p, 2);
        cluster_mean = cluster_sum ./ Ni;
        sigmapn = (cluster_mean.^(1/p)) - 1; % [nc, 1]

        % --- 归一化 P-norm 应力约束对单个 von Mises 应力的导数 d(s_i) / d(sigma_a^vM) ---
        dsi_dvm = zeros(nc, n); 

        term_A = (1/p) * (cluster_mean.^((1/p) - 1)); 
        term_B_factor = (p / (Ni * (sigmay^p)));

        for i = 1:nc  % 循环 nc 个簇
            
            for j = 1:Ni  % 循环簇 i 中的每个应力点
                
                % 1. 获取该应力点的 *原始* 索引 (在 1..n 范围内)
                original_index = sort_index((i-1)*Ni + j);
            
                % 2. 获取该应力点的 *值*
                sigma_a_vM = cluster_vm(i, j); 
                
                % 3. 计算导数
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

        % 7. 计算伴随载荷 (Adjoint Load) R_i
        
        % 获取总自由度
        ndof = (nelx+1)*(nely+1)*2;
        
        % 每一列 adjoint_RHS(:, i) 就是 R_i
        adjoint_RHS = zeros(ndof, nc);
        
        B_T_C = B' * C;
        
        for i = 1:nc  % 循环 nc 个簇
            
            % R_i 是此簇的全局伴随载荷向量
            R_i = zeros(ndof, 1);
            
            for j = 1:Ni  % 循环簇 i 中的每个单元 'a'
                
                % 1. 获取该单元的 *原始* 索引 (1..n)
                a = sort_index((i-1)*Ni + j);
                
                % 2. 获取该单元的导数值 (标量)
                dsi_dvm_val = dsi_dvm(i, a);
                
                % 3. 获取 d(vm)/d(stress) (3x1 列向量)
                dvm_dstress_vec = dvm_dstress(a, :)';
                
                % 4. 计算单元 'a' 对 R_i 的贡献 (8x1 向量)
                %    r_a = (B' * C) * (dvm/dstress) * (dsi/dvm)
                local_R = (B_T_C * dvm_dstress_vec) * dsi_dvm_val;
                
                % 5. 组装到全局 R_i 向量
                
                % 找到单元 'a' 的 elx, ely
                ely = mod(a-1, nely) + 1;
                elx = floor((a-1) / nely) + 1;
                
                % 找到单元 'a' 的 8 个自由度
                n1 = (nely+1)*(elx-1)+ely; 
                n2 = (nely+1)* elx   +ely;
                dof_indices = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
                
                R_i(dof_indices) = R_i(dof_indices) + local_R;
            end
            
            adjoint_RHS(:, i) = R_i;
        end




        % von Mises 应力及其对密度的导数
        [von_mises, derivative] = stress_func(C, B, U, nelx, nely, x, p);
        % p-norm 聚合函数
        [sigmapn, derivative0] = pnorm(p, von_mises, nc, nelx, nely, sigmay);
        % 应力约束灵敏度
        [dfdx_0] = derivative_stress(derivative, derivative0, nc, n, nelx, nely, penal, rmin, x);
        dfdx = dfdx_0;
        fval = (sigmapn / sigmay) - 1.0;
    case 2
        % P2: nc 个应力约束 + 1 个质量约束
        m = nc + 1;
    case 3
        % P3: 1 个质量约束
        m = 1;
    case 4
        % P4: 柔顺度最小化 + 应力约束 + 体积分数约束 
        f0val = 0.;
        fval_1 = 0;
        for ely = 1:nely
            for elx = 1:nelx
                n1 = (nely+1)*(elx-1)+ely; 
                n2 = (nely+1)* elx   +ely;
                Ue = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2], 1);
                % 柔顺度
                f0val = f0val + x(ely, elx)^penal*Ue'*KE*Ue;
                % 柔顺度灵敏度
                dc(ely, elx) = -penal*x(ely, elx)^(penal-1)*Ue'*KE*Ue;
                % 当前体积
                fval_1 = fval_1 + unit_size_x*unit_size_y*x(ely, elx);
            end
        end

        [dc] = check(nelx, nely, rmin, x, dc);

        count_2 = 1;
        for g = 1:nely
            for h = 1:nelx
                df0dx(count_2, 1) = dc(g, h);
                count_2 = count_2 + 1;
            end
        end
        % 体积分数约束
        fval(1, 1) = (fval_1 / (l*b)) - volfrac;
end


% f0val-目标函数; df0dx-目标函数灵敏度; fval-约束函数
[f0val, df0dx, fval] = Load(F, U, x, M, KE, m, n, nelx, nely, penal, unit_size_x, unit_size_y, rmin, volfrac);




% 体积约束梯度
M = l * b;
dfdx_1 = (unit_size_x * unit_size_y / M) * ones(1, n);







% p-norm 聚合函数
[sigmapn, derivative0] = pnorm(p, von_mises, nc, nelx, nely, sigmay);

% 保留体积约束, 添加应力约束
fval(2:m, 1) = sigmapn(1:nc, 1);
% 只考虑应力约束
% fval = sigmapn;

% 应力约束灵敏度
[dfdx_0] = derivative_stress(derivative, derivative0, nc, n, nelx, nely, penal, rmin, x);

% 保留体力约束灵敏度, 添加应力约束灵敏度
dfdx = [dfdx_1; dfdx_0];
% 只考虑应力约束灵敏度
% dfdx = dfdx_0;

ploty = zeros(m, maxite);  % 记录约束值历史
while iter < maxite
    iter = iter + 1;

    ploty(:, iter) = fval;
    
    [xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp] = mmasub(m, n, iter, ...
                                                xval, xmin, xmax, xold1, xold2, ... 
                                                f0val, df0dx, df0dx2, fval, dfdx, dfdx2, ...
                                                low, upp, a0, a, c, d); 
     
    xold2 = xold1;
    xold1 = xval;
    xval = xmma;
     
    x = reshape(xval, [nelx, nely])';
    
    [F, U] = FEA(nelx, nely, x, penal, KE);
    
    [f0val, df0dx, fval] = Load(F, U, x, M, KE, m, n, nelx, nely, penal, ...
                            unit_size_x, unit_size_y, rmin, volfrac);
    
    [von_mises, derivative] = stress_func(C, B, U, nelx, nely, x, p);
    
    [sigmapn, derivative0] = pnorm(p, von_mises, nc, nelx, nely, sigmay);
    
    fval(2:m, 1) = sigmapn(1:nc, 1);
    % fval = sigmapn;

    [dfdx_0] = derivative_stress(derivative, derivative0, nc, n, nelx, nely, penal, rmin, x);
    dfdx = [dfdx_1; dfdx_0];
    % dfdx  =dfdx_0;

    %Plotting
    colormap(gray); imagesc(-x); axis equal; axis tight; axis off;pause(1e-6);

    fprintf('Iter: %3d  Obj: %10.4f  Vol: %6.3f  MaxStress: %6.3f\n', iter, f0val, fval(1)+volfrac, max(sigmapn)+1);
     
end

ploty = ploty(:, 1:iter)'; 