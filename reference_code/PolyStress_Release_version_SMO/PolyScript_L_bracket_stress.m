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
[Node,Element,Supp,Load] = Mesh_L_bracket(50176); % 10000, 22500, 50176
NElem = size(Element,1); % Number of elements
%% ---------------------------------------------------- CREATE 'fem' STRUCT
E0 = 70e3; % E0 in MPa
G = E0/2.5; Et = E0; Ec = E0;  % 0<=(Et,Ec)<=3*G; %Material props. (linear)
fem = struct(...
  'NNode',size(Node,1),...      % Number of nodes
  'NElem',size(Element,1),...   % Number of elements
  'Node',Node,...               % [NNode x 2] array of nodes
  'Element',{Element},...       % [NElement x Var] cell array of elements
  'Supp',Supp,...               % Array of supports
  'Load',Load,...               % Array of loads
  'Passive',[],...              % Passive elements  
  'Thickness',1,...             % Element thickness
  'MatModel','Bilinear',...     % Material model ('Bilinear','Polynomial')
  'MatParam',[Et,Ec,G],...      % Material parameters for MatModel
  'SLim',100,...                % Stress limit
  'TolR', 1e-8, ...             % Tolerance for norm of force residual
  'MaxIter', 15, ...            % Max NR iterations per load step
  'MEX', 'No');                 % Tag to use MEX functions in NLFEM routine
%% ---------------------------------------------------- CREATE 'opt' STRUCT
R = 0.05; q = 3; % Filter radius and filter exponent
p = 3.5; eta0 = 0.5;
m = @(y,B)MatIntFnc(y,'SIMP-H1',[p,B,eta0]);
P = PolyFilter(fem,R,q);
zIni = 0.5*ones(size(P,2),1);
opt = struct(...               
  'zMin',0.0,...              % Lower bound for design variables
  'zMax',1.0,...              % Upper bound for design variables
  'zIni',zIni,...             % Initial design variables
  'MatIntFnc',m,...           % Handle to material interpolation fnc.
  'contB',[5,1,1,10],...      % Threshold projection continuation params.  
  'P',P,...                   % Matrix that maps design to element vars.
  'Tol',0.002,...             % Convergence tolerance on design vars.
  'TolS',0.003,...            % Convergence tolerance on stress constraints
  'MaxIter',150,...           % Maximum number of AL steps
  'MMA_Iter',5,...            % Number of MMA iterations per AL step
  'lambda0',zeros(NElem,1),...% Initial Lagrange multiplier estimators
  'mu0',10,...                % Initial penalty factor for AL function
  'mu_max',10000,...          % Maximum penalty factor for AL function
  'alpha',1.1,...             % Penalty factor update parameter  
  'Move',0.15,...             % Allowable move step in MMA update scheme
  'Osc',0.2,...               % Osc parameter in MMA update scheme
  'AsymInit',0.2,...          % Initial asymptote in MMA update shecme
  'AsymInc',1.2,...           % Asymptote increment in MMA update scheme  
  'AsymDecr',0.7...           % Asymptote decrement in MMA update scheme     
   );
%% ------------------------------------------------------- RUN 'PolyStress'
fem = preComputations(fem); % Run preComputations before running PolyStress
[z,V,fem] = PolyStress(fem,opt);
%-------------------------------------------------------------------------%
%% ------------------------------------------------------- POST-PROCESSING
% 绘制 Von Mises 屈服面和主应力分布
fprintf('Plotting Yield Surface...\n');
PlotYieldSurface(fem, z, opt);

%% ------------------------------------------------------- AUXILIARY FUNCTION
function PlotYieldSurface(fem, z, opt) % 注意：这里我顺便把输入参数改为了支持 (fem, z, opt) 以匹配你的调用
    % 1. 获取位移和参数
    U = fem.U;                         
    SLim = fem.SLim;                   
    
    % 2. 计算单元中心应变和应力
    ElemU = U(fem.eDof);               
    ee_elem = fem.B0 * ElemU;
    
    % 应变向量重塑为 [3 x NElem] 矩阵
    ee_elem = reshape(ee_elem, 3, []); 
    
    % 调用本构模型
    [Cauchy_S, ~] = material_model(fem.MatModel, fem.MatParam, ee_elem);
    
    % 提取分量 (现在 Cauchy_S 是 3 x NElem 矩阵，提取后为 NElem x 1 向量)
    sig_x = Cauchy_S(1, :)';
    sig_y = Cauchy_S(2, :)';
    tau_xy = Cauchy_S(3, :)';

    % 3. 计算主应力 (Principal Stresses)
    center = (sig_x + sig_y) / 2;
    radius = sqrt(((sig_x - sig_y) / 2).^2 + tau_xy.^2);
    
    sig_1 = center + radius;
    sig_2 = center - radius;

    % 4. 过滤数据
    % 确保 z 和 sig_1 长度一致。通常 z 是 [NElem x 1]。
    if length(z) ~= length(sig_1)
        warning('z 和 sig_1 长度不一致，尝试使用 V 或调整维度');
        % 如果 z 是设计变量，可能需要检查是否被被重塑过，但在 PolyStress 中通常是一致的
    end
    
    mask = z > 0.5;  % 仅绘制实体单元
    
    % 检查是否有满足条件的单元
    if ~any(mask)
        warning('没有单元满足 z > 0.5 的条件，无法绘图。');
        return;
    end

    s1_plot = sig_1(mask) / SLim; 
    s2_plot = sig_2(mask) / SLim; 

    % 5. 绘图
    % 创建新图形窗口，避免覆盖原有图形
    figure('Name', 'Yield Surface Analysis'); 
    hold on; box on; grid on; axis equal;
    
    % 绘制 Von Mises 包络线 (单位圆变换)
    fimplicit(@(x,y) x.^2 - x.*y + y.^2 - 1, [-1.5 1.5 -1.5 1.5], 'r', 'LineWidth', 2);

    % 绘制应力点
    scatter(s1_plot, s2_plot, 15, 'k', 'filled', 'MarkerFaceAlpha', 0.5);

    % 标注
    xlabel('$\sigma_1 / \sigma_{lim}$', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('$\sigma_2 / \sigma_{lim}$', 'Interpreter', 'latex', 'FontSize', 14);
    
    % 计算最大应力比 (基于显示的单元)
    max_stress_ratio = 0;
    if ~isempty(s1_plot)
        % 重新计算这些点的 VM 应力来找最大值，或者直接用 fem.VM_Stress0
        % 这里简单用 fem.VM_Stress0 (如果它被更新了)
        if isfield(fem, 'VM_Stress0') && length(fem.VM_Stress0) == length(mask)
            max_val = max(fem.VM_Stress0(mask)) / SLim;
        else
            max_val = max(sqrt(s1_plot.^2 - s1_plot.*s2_plot + s2_plot.^2)); % 近似反算
        end
        title(['Yield Surface Check (Max Ratio: ' num2str(max_val, '%.2f') ')']);
    else
        title('Yield Surface Check');
    end
    
    xlim([-1.5 1.5]); ylim([-1.5 1.5]);
end