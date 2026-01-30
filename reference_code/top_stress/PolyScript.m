%% PolyStress
% Ref: O. Giraldo-Londono, G.H. Paulino, "PolyStress: A Matlab implementation
% for topology optimization with local stress constraints using the
% augmented Lagrangian method", Structural and Multidisciplinary
 % Optimization, DOI 10.1007/s00158-020-02760-8 [cite: 1112-1115]

clear; clc; close all;

%% 路径设置
restoredefaultpath; addpath(genpath('./'));  % 使用当前文件夹及子文件夹 [cite: 1122]
set(0, 'defaulttextinterpreter', 'latex');   % 设置绘图字体解释器 [cite: 1123]

%% 1. 网格生成 (MESH GENERATION)
% 调用网格生成函数（此处以 L-bracket 为例）
[Node, Element, Supp, Load] = Mesh_L_bracket(40, 40);  % [cite: 1127]
NElem = size(Element, 1);  % 单元数量 [cite: 1129]

%% 2. 定义有限元结构体 'fem' (CREATE 'fem' STRUCT)
E0 = 70e3;               % 弹性模量 (MPa) [cite: 1131]
G = E0/2.5;             % 剪切模量
Et = E0; Ec = E0;        % 拉伸和压缩模量 (线性材料) [cite: 1132]

fem = struct(...
     'NNode', size(Node, 1), ...         % 节点数量 [cite: 1135]
     'NElem', size(Element, 1), ...      % 单元数量 [cite: 1138]
     'Node', Node, ...                   % 节点坐标矩阵 [cite: 1141]
     'Element', {Element}, ...           % 单元连接关系 (Cell数组) [cite: 1143]
     'Supp', Supp, ...                   % 边界条件 [cite: 1147]
     'Load', Load, ...                   % 载荷 [cite: 1150]
     'Passive', [], ...                  % 被动单元 [cite: 1153]
     'Thickness', 1, ...                 % 单元厚度 [cite: 1156]
     'MatModel', 'Bilinear', ...         % 材料模型 ('Bilinear' 或 'Polynomial') [cite: 1159]
     'MatParam', [Et, Ec, G], ...        % 材料参数 [cite: 1161]
     'SLim', 100, ...                    % 应力极限 (Stress Limit) [cite: 1163]
     'TolR', 1e-8, ...                   % 牛顿迭代残差容差 [cite: 1165]
     'MaxIter', 15, ...                  % 每个载荷步的最大 NR 迭代次数 [cite: 1167]
    'MEX', 'No');                        % 是否使用 MEX 加速 [cite: 1169]

%% 3. 过滤与插值参数 (FILTER & INTERPOLATION)
R = 0.05; q = 3;                         % 过滤半径 R 和 过滤指数 q [cite: 1177]
p = 3.5; eta0 = 0.5;                     % SIMP 惩罚因子 p 和 投影阈值 eta0 [cite: 1178]

% 定义材料插值函数句柄 (包含 SIMP 和 Heaviside 投影)
m = @(y, B) MatIntFnc(y, 'SIMP-H1', [p, B, eta0]);  % [cite: 1179]

% 计算过滤矩阵 P
P = PolyFilter(fem, R, q);               % [cite: 1181]
zIni = 0.5 * ones(size(P, 2), 1);        % 初始化设计变量 [cite: 1182]

%% 4. 定义优化结构体 'opt' (CREATE 'opt' STRUCT)
opt = struct(...
     'zMin', 0.0, ...                    % 设计变量下界 [cite: 1188]
     'zMax', 1.0, ...                    % 设计变量上界 [cite: 1190]
     'zIni', zIni, ...                   % 初始设计变量 [cite: 1193]
     'MatIntFnc', m, ...                 % 材料插值函数句柄 [cite: 1196]
     'contB', [5, 1, 1, 10], ...         % 投影参数 beta 的延续策略 [频率, 初值, 增量, 最大值] [cite: 1199]
     'P', P, ...                         % 映射矩阵 (设计变量 -> 单元变量) [cite: 1202]
     'Tol', 0.002, ...                   % 设计变量收敛容差 [cite: 1205]
     'TolS', 0.003, ...                  % 应力约束收敛容差 [cite: 1208]
     'MaxIter', 150, ...                 % 最大 AL 步数 [cite: 1211]
     'MMA_Iter', 5, ...                  % 每个 AL 步内的 MMA 迭代次数 [cite: 1213]
     'lambda0', zeros(NElem, 1), ...     % 初始拉格朗日乘子 [cite: 1226]
     'mu0', 10, ...                      % 初始罚因子 [cite: 1226]
     'mu_max', 10000, ...                % 最大罚因子 [cite: 1216]
     'alpha', 1.1, ...                   % 罚因子更新参数 [cite: 1218]
     'Move', 0.15, ...                   % MMA 移动极限 [cite: 1220]
     'Osc', 0.2, ...                     % MMA 振荡参数 [cite: 1222]
     'AsymInit', 0.2, ...                % MMA 初始渐近线参数 [cite: 1224]
     'AsymInc', 1.2, ...                 % MMA 渐近线增量 [cite: 1235]
     'AsymDecr', 0.7 ...                 % MMA 渐近线减量 [cite: 1238]
    );

%% 5. 运行主程序 (RUN)
fem = PreComputations(fem);              % 运行预计算 (刚度矩阵准备等) [cite: 1244]
[z, V, fem] = PolyStress(fem, opt);      % 运行 PolyStress 主求解器 [cite: 1245]