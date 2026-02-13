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
nx = 100; ny = 100;
% nx = 200; ny = 200;
% nx = 300; ny = 300;
[Node, Element, Supp, Load] = Mesh_L_bracket_quad(nx, ny);
% 将数值矩阵转换为元胞数组
% PolyStress 里的函数预期 Element 是 cell array
Element = num2cell(Element, 2);
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
% m = @(y,B)MatIntFnc(y,'SIMP-H1',[p,B,eta0]);
m = @(y,B)MatIntFnc(y,'SIMP',[p,B,eta0]);
P = PolyFilter(fem,R,q);
% P = speye(NElem); 
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
%% ============ 应力约束检查和可视化 ============
% 1. 获取最终的单元密度和材料插值
y = opt.P * z;  % 过滤后的密度
[E, ~, ~, ~] = opt.MatIntFnc(y, opt.contB(4));  % 使用最终的B值

% 2. 计算归一化应力度量 (这是实际的应力约束)
SM = E .* fem.VM_Stress0 / fem.SLim;

% 3. 统计分析
fprintf('\n========== 应力约束检查结果 ==========\n');
fprintf('应力限制 (σ_lim): %.1f MPa\n', fem.SLim);
fprintf('最大归一化应力 (max(SM)): %.4f\n', max(SM));
fprintf('平均归一化应力: %.4f\n', mean(SM));

% 4. 找出违反约束的单元 (SM > 1.0)
tolerance = 0.01;  % 1% 容差
violated_elements = find(SM > 1.0 + tolerance);
fprintf('违反应力约束的单元数量: %d / %d\n', length(violated_elements), fem.NElem);

if ~isempty(violated_elements)
    fprintf('??  警告：存在应力约束违反！\n');
    fprintf('最大违反程度: %.2f%%\n', (max(SM) - 1.0) * 100);
else
    fprintf('? 所有应力约束均满足！\n');
end

% 5. 只对实体单元统计 (密度 > 0.5)
solid_mask = V > 0.5;
fprintf('\n========== 实体单元 (ρ>0.5) 应力统计 ==========\n');
fprintf('实体单元数量: %d\n', sum(solid_mask));
fprintf('实体单元最大归一化应力: %.4f\n', max(SM(solid_mask)));
fprintf('实体单元平均归一化应力: %.4f\n', mean(SM(solid_mask)));

violated_solid = sum(SM(solid_mask) > 1.0 + tolerance);
fprintf('违反约束的实体单元: %d\n', violated_solid);

%% ============ 绘制 von Mises 屈服面 ============
PlotYieldSurface(fem, z, opt, E, V, SM);

%% ============ 屈服面绘制函数 ============
function PlotYieldSurface(fem, z, opt, E, V, SM)
    % 1. 重新计算主应力（需要完整的应力张量）
    y = opt.P * z;
    
    % 2. 从fem中获取最后一次FEM分析的位移
    % 注意：fem.U 在PolyStress的NLFEM中被更新
    if ~isfield(fem, 'U')
        warning('fem.U未定义，无法绘制屈服面');
        return;
    end
    
    U = fem.U;
    ElemU = U(fem.eDof);
    ee_elem = fem.B0 * ElemU;  % 应变 [3*NElem x 1]
    ee_elem = reshape(ee_elem, 3, []);  % [3 x NElem]
    
    % 3. 调用material_model获取Cauchy应力
    [Cauchy_S, ~] = material_model(fem.MatModel, fem.MatParam, ee_elem);
    % Cauchy_S: [3 x NElem], 其中行为 [σ11; σ22; σ12]
    
    % 4. 应力需要乘以材料插值函数
    Cauchy_S = Cauchy_S .* repmat(E', 3, 1);
    
    % 5. 提取应力分量
    sig_11 = Cauchy_S(1, :)';
    sig_22 = Cauchy_S(2, :)';
    tau_12 = Cauchy_S(3, :)';
    
    % 6. 计算主应力
    center = (sig_11 + sig_22) / 2;
    radius = sqrt(((sig_11 - sig_22) / 2).^2 + tau_12.^2);
    sig_1 = center + radius;  % 第一主应力
    sig_2 = center - radius;  % 第二主应力
    
    % 7. 归一化
    sig_1_norm = sig_1 / fem.SLim;
    sig_2_norm = sig_2 / fem.SLim;
    
    % 8. 只绘制实体单元 (ρ > 0.5)
    solid_mask = V > 0.5;
    
    if sum(solid_mask) == 0
        warning('没有实体单元，无法绘制屈服面');
        return;
    end
    
    s1_plot = sig_1_norm(solid_mask);
    s2_plot = sig_2_norm(solid_mask);
    
    % 9. 绘制
    figure('Name', 'Von Mises Yield Surface Analysis', 'Position', [100, 100, 800, 700]);
    hold on; box on; grid on; axis equal;
    
    % 绘制von Mises屈服面 (椭圆)
    theta = linspace(0, 2*pi, 200);
    r = 1.0;
    x_circle = r * cos(theta);
    y_circle = r * sin(theta);
    % von Mises条件: σ1^2 - σ1*σ2 + σ2^2 = σ_lim^2
    plot(x_circle, y_circle, 'r-', 'LineWidth', 2.5, 'DisplayName', 'von Mises屈服面');
    
    % 绘制应力点
    scatter(s1_plot, s2_plot, 20, SM(solid_mask), 'filled', ...
            'MarkerFaceAlpha', 0.6, 'DisplayName', '应力评估点');
    
    % 标注和格式
    xlabel('$\sigma_1 / \sigma_{\rm lim}$', 'Interpreter', 'latex', 'FontSize', 16);
    ylabel('$\sigma_2 / \sigma_{\rm lim}$', 'Interpreter', 'latex', 'FontSize', 16);
    
    % 计算实际的最大应力比
    vm_actual = sqrt(s1_plot.^2 - s1_plot.*s2_plot + s2_plot.^2);
    max_ratio = max(vm_actual);
    
    title(sprintf('屈服面检查 (最大应力比: %.3f)', max_ratio), ...
          'Interpreter', 'latex', 'FontSize', 14);
    
    colorbar;
    colormap('jet');
    caxis([0, max(1.2, max_ratio)]);
    
    xlim([-1.5, 1.5]);
    ylim([-1.5, 1.5]);
    
    legend('Location', 'best', 'Interpreter', 'latex');
    
    % 添加统计信息文本框
    text_str = sprintf('实体单元数: %d\n最大归一化应力: %.3f\n平均归一化应力: %.3f', ...
                       sum(solid_mask), max(SM(solid_mask)), mean(SM(solid_mask)));
    annotation('textbox', [0.15, 0.75, 0.2, 0.15], 'String', text_str, ...
               'FitBoxToText', 'on', 'BackgroundColor', 'white', ...
               'EdgeColor', 'black', 'Interpreter', 'latex');
    
    hold off;
    fprintf('\n? 屈服面图已生成\n');
end