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
[Node,Element,Supp,Load] = Mesh_L_bracket(10000); 
NElem = size(Element,1); % Number of elements

% % 绘图代码
% figure(101); clf; axis equal; axis off; hold on; 
% title('Generated Quad Mesh');
% 
% % 使用NaN填充法处理不同节点数的单元
% MaxNVer = max(cellfun(@numel, Element));
% PadWNaN = @(E) [E, NaN(1, MaxNVer-numel(E))];
% ElemMat = cellfun(PadWNaN, Element, 'UniformOutput', false);
% ElemMat = vertcat(ElemMat{:});
% 
% patch('Faces', ElemMat, 'Vertices', Node, ...
%       'FaceColor', 'w', 'EdgeColor', 'k');
% 
% % 从Supp和Load中提取节点编号
% fixed_nodes = Supp(:,1);
% load_nodes = Load(:,1);
% 
% plot(Node(fixed_nodes,1), Node(fixed_nodes,2), 'b>', 'MarkerSize', 8);
% plot(Node(load_nodes,1), Node(load_nodes,2), 'm^', 'MarkerSize', 8);
% 
% legend('', 'Support', 'Load', 'Location', 'northoutside', 'Orientation', 'horizontal');
% drawnow;

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
    'E0', E0,...                    % Young's modulus (PolyTop需要)
    'Nu0', 0.25,...                 % Poisson's ratio (PolyTop需要)
    'Reg', 0,...                    % 是否规则网格 (PolyTop需要)
    'Thickness',1);                 % Element thickness

%% ---------------------------------------------------------- CREATE 'opt' STRUCT
R = 0.05; q = 3;                    % Filter radius and exponent
p = 3; eta0 = 0.5;                  % SIMP参数
B = 1;                              % 投影参数（如果不用投影，设为1或很小的值）
eta0 = 0.5;                         % 投影阈值
VolFrac = 0.3;                      % 体积分数约束 (PolyTop特有)
% m = @(y,p,B,eta)MatIntFnc(y,'SIMP-H1',[p,B,eta]);
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
    'MaxIter',200,...               % Max. number of iterations
    'VolFrac',VolFrac,...           % 体积约束 (PolyTop特有)
    'OCMove',0.2,...                % OC移动限 (PolyTop特有)
    'OCEta',0.5);                   % OC阻尼因子 (PolyTop特有)

%% ------------------------------------------------------- RUN 'PolyTop'
% 注意：PolyTop 不需要 preComputations
[z,V,fem] = PolyTop(fem,opt);
%-------------------------------------------------------------------------%