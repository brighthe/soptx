%--------------------------- ClusterStressOpt ----------------------------%
% 基于 Holmberg 等 (2013) 的聚类应力约束拓扑优化代码框架
% 参考文献: Holmberg E, Torstenfelt B, Klarbring A. 
% "Stress constrained topology optimization", Struct Multidisc Optim, 2013.
%-------------------------------------------------------------------------%
clear; clc; close all
restoredefaultpath; addpath(genpath('./')); % 加载所有子文件夹
set(0,'defaulttextinterpreter','latex')

%% 1. 创建网格 (CREATE Mesh)
% 使用与 PolyStress 相同的 L-bracket 网格生成器
[Node, Element, Supp, Load] = Mesh_L_bracket(6400); % Holmberg 论文中使用 6400 个单元
NElem = size(Element, 1);

%% 2. 定义 FEM 结构体 (CREATE 'fem' STRUCT)
E0 = 71e3;    % 杨氏模量 (MPa), Holmberg 论文数据 [cite: 2077]
nu = 0.33;    % 泊松比, Holmberg 论文数据 [cite: 2077]
SigLim = 350; % 屈服极限 (MPa), Holmberg 论文数据 [cite: 2077]

fem = struct(...
  'NNode', size(Node,1),...
  'NElem', size(Element,1),...
  'Node', Node,...
  'Element', {Element},...
  'Supp', Supp,...
  'Load', Load,...
  'Thickness', 1,...
  'MatModel', 'Linear',...      % Holmberg 论文使用线性材料 [cite: 1722]
  'MatParam', [E0, nu],...      % 材料参数
  'SLim', SigLim,...            % 应力极限
  'P_norm', 8,...               % P-范数因子 p，论文建议取 8 
  'nClusters', 10,...           % 簇的数量 n_c，论文示例中使用 10 
  'ClusterStrategy', 'StressLevel',... % 聚类策略: 'StressLevel' 或 'Distributed' [cite: 1944, 1954]
  'ExcludeLoad', 'Yes',...      % 是否排除加载点附近的单元 (见论文 Section 8.1) 
  'MEX', 'No');

%% 3. 定义优化结构体 (CREATE 'opt' STRUCT)
R = 1.5; % 过滤半径 r_0 (单元尺寸的倍数), 论文取 1.5 [cite: 2079]
% 这里的 R 需要转换为实际物理长度，具体取决于网格尺寸

% 初始设计变量 (0.5) [cite: 2052]
zIni = 0.5 * ones(NElem, 1);

opt = struct(...
  'zMin', 0.001,...             % 下限 (避免刚度矩阵奇异) [cite: 1792]
  'zMax', 1.0,...               % 上限
  'zIni', zIni,...              % 初始设计变量
  'FilterRadius', R,...         % 过滤半径
  'ReclusterFreq', 1,...        % 重新聚类的频率 (1 = 每次迭代更新), 效果最好 
  'MaxIter', 200,...            % 最大迭代步数
  'Tol', 0.01,...               % 收敛容差
  'Move', 0.1,...               % MMA 移动极限 (Holmberg 建议设置保守一些) [cite: 2053]
  'Objective', 'Mass',...       % 目标函数: 'Mass' (P1问题) 或 'Compliance' (P2问题) [cite: 1785]
  'Constraint', 'Stress',...    % 约束类型: 聚类应力约束
  'Solver', 'MMA'...            % 优化求解器 (标准 MMA) [cite: 1764]
);

%% 4. 运行主程序 (RUN Main Solver)
% 注意：这里不再调用 PolyStress (ALM方法)，而是调用基于聚类的求解器
% 该求解器内部将执行:
% 1. 有限元分析
% 2. 计算局部应力
% 3. 执行聚类 (Sorting & Clustering)
% 4. 计算 P-norm 应力约束及其灵敏度
% 5. 调用 MMA 更新变量

fem = preComputations(fem); % 预计算 (刚度矩阵等)
[z_opt, fem_results] = ClusterStressOpt(fem, opt);

%% 5. 结果后处理 (Post-Processing)
% 绘制最终的密度分布和应力云图
PlotResults(z_opt, fem_results);