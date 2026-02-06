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
    'MatModel', 'Bilinear',...      % 【新增】应力计算需要
    'MatParam', [Et, Ec, G],...     % 【新增】应力计算需要
    'SLim',100,...                  % Stress limit
    'Reg', 0,...                    % 是否规则网格 (PolyTop需要)
    'Thickness',1);                 % Element thickness

%% ---------------------------------------------------------- CREATE 'opt' STRUCT
R = 0.05; q = 3;                    % Filter radius and exponent
p = 3;                              % SIMP参数
B = 1;                              % 投影参数（如果不用投影，设为1或很小的值）
eta0 = 0.5;                         % 投影阈值
VolFrac = 0.32;                      % 体积分数约束 (PolyTop特有)
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
[z,V,fem,history] = PolyTop(fem,opt);  % 接收 history 输出
%-------------------------------------------------------------------------%