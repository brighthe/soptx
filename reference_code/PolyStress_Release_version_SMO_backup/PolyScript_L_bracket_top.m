%------------------------------- PolyStress ------------------------------%
% Ref: O Giraldo-Londo駉, GH Paulino, "PolyStress: A Matlab implementation%
% for topology optimization with local stress constraints using the       %
% augmented Lagrangian method", Structural and Multidisciplinary          %
% Optimization, DOI 10.1007/s00158-020-02664-7, 2020                      %
%-------------------------------------------------------------------------%
clear; clc; close all
restoredefaultpath; addpath(genpath('./')); %Use all folders and subfolders
set(0,'defaulttextinterpreter','latex')
%% ------------------------------------------------------------ CREATE Mesh
[Node,Element,Supp,Load] = Mesh_L_bracket(40000); 
NElem = size(Element,1); % Number of elements

%% ---------------------------------------------------------- CREATE 'fem' STRUCT
E0 = 70e3; % E0 in MPa
G = E0/2.5; Et = E0; Ec = E0; % 0<=(Et,Ec)<=3*G; %Material props. (linear)
fem = struct(...
    'NNode',size(Node,1),...        % Number of nodes
    'NElem',size(Element,1),...     % Number of elements
    'Node',Node,...                 % [NNode x 2] array of nodes
    'Element',{Element},...         % [NElem x Var] cell array of elements
    'Supp',Supp,...                 % Array of supports
    'Load',Load,...                 % Array of loads
    'Passive',[],...                % Passive elements  
    'Thickness',1,...               % Element thickness
    'MatModel','Bilinear',...       % Material model ('Bilinear','Polynomial')
    'MatParam',[Et,Ec,G],...        % Material parameters for MatModel
    'SLim',100,...                  % Stress limit
    'TolR', 1e-8, ...               % Tolerance for norm of force residual
    'MaxIter', 15, ...              % Max NR iterations per load step
    'MEX', 'No', ...                % Tag to use MEX functions in NLFEM routine
    'E0', E0,...                    % Young's modulus (PolyTop需要)
    'Nu0', 0.25,...                 % Poisson's ratio (PolyTop需要)
    'Reg', 0);                      % 是否规则网格 (PolyTop需要)
%% ---------------------------------------------------------- CREATE 'opt' STRUCT
R = 0.05; q = 3;                    % Filter radius and exponent
p = 3.5;                            % SIMP参数
B = 1;                              % 投影参数
eta0 = 0.5;                         % 投影阈值
VolFrac = 0.34;                     % 体积分数约束 (PolyTop特有)
m = @(y) MatIntFnc(y, 'SIMP-H1', [p, B, eta0]);
P = PolyFilter(fem,R,q);
zIni = VolFrac*ones(size(P,2),1);   % 初始设计变量
opt = struct(...
    'zMin',0.0,...                  % Lower bound for design variables
    'zMax',1.0,...                  % Upper bound for design variables
    'zIni',zIni,...                 % Initial design variables
    'MatIntFnc',m,...               % Handle to material interpolation fnc.
    'P',P,...                       % Matrix that maps design to element vars.
    'Tol',0.01,...                  % Convergence tolerance
    'MaxIter',300,...               % Max. number of iterations
    'VolFrac',VolFrac,...           % 体积约束 (PolyTop特有)
    'OCMove',0.2,...                % OC移动限 (PolyTop特有)
    'OCEta',0.5);                   % OC阻尼因子 (PolyTop特有)

%% ------------------------------------------------------- RUN 'PolyTop'
fem = preComputations(fem); % Run preComputations before running PolyStress
[z,V,fem] = PolyTop(fem,opt);  
%-------------------------------------------------------------------------%

%% ============ 柔顺度优化结果的应力检查 ============
fprintf('\n========== 柔顺度优化结果分析 ==========\n');

% 1. 计算最终体积分数
if ~isfield(fem,'ElemArea')
    fem.ElemArea = zeros(fem.NElem,1);
    for el=1:fem.NElem
        vx=fem.Node(fem.Element{el},1); 
        vy=fem.Node(fem.Element{el},2);
        fem.ElemArea(el) = 0.5*sum(vx.*vy([2:end 1])-vy.*vx([2:end 1]));
    end
end
actual_volfrac = sum(fem.ElemArea.*V)/sum(fem.ElemArea);
fprintf('目标体积分数: %.4f\n', VolFrac);
fprintf('实际体积分数: %.4f\n', actual_volfrac);

% 2. 运行 FEM 分析获取位移（线性材料，直接求解）
K = sparse(fem.i, fem.j, V(fem.e).*fem.k);
K = (K + K') / 2;
U = zeros(2*fem.NNode, 1);
U(fem.FreeDofs) = K(fem.FreeDofs, fem.FreeDofs) \ fem.F(fem.FreeDofs);

% 3. 计算 von Mises 应力
ElemU = U(fem.eDof);
ee_elem = fem.B0 * ElemU;
ee_elem = reshape(ee_elem, 3, []);
[Cauchy_S, ~] = material_model(fem.MatModel, fem.MatParam, ee_elem);

% 应用材料插值
Cauchy_S = Cauchy_S .* repmat(V', 3, 1);

% von Mises 应力
VM_matrix = [1, -1/2, 0; -1/2, 1, 0; 0, 0, 3];
VM_Stress = max(sqrt(sum(Cauchy_S .* (VM_matrix * Cauchy_S))), eps)';

% 4. 归一化应力
SM = VM_Stress / fem.SLim;

% 5. 统计分析
solid_mask = V > 0.5;
fprintf('\n========== 应力检查结果 ==========\n');
fprintf('应力限制 (σ_lim): %.1f MPa\n', fem.SLim);
fprintf('最大归一化应力: %.4f (%.1f%% 超限)\n', ...
        max(SM), (max(SM) - 1.0) * 100);
fprintf('实体单元最大应力: %.4f\n', max(SM(solid_mask)));
fprintf('实体单元平均应力: %.4f\n', mean(SM(solid_mask)));

violated = sum(SM(solid_mask) > 1.0);
fprintf('超限单元数: %d / %d (%.1f%%)\n', ...
        violated, sum(solid_mask), violated/sum(solid_mask)*100);

if max(SM) > 1.0
    fprintf('\n⚠️  警告：柔顺度优化结果存在应力超限！\n');
    fprintf('   最大超限幅度: %.1f%%\n', (max(SM) - 1.0) * 100);
else
    fprintf('\n✅ 所有应力约束均满足！\n');
end

% 6. 绘制结果对比图
PlotComplianceResults(fem, V, SM, solid_mask);