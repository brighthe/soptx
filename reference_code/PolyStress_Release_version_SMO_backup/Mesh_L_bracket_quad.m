function [Node, Element, Supp, Load] = Mesh_L_bracket_quad(nx, ny, plot_mesh)
%-------------------------------------------------------------------------%
% 生成L型梁的规则四边形网格（精确版本，仿照FEALPy实现）
% 输入: nx        - x方向单元数（精确）
%       ny        - y方向单元数（精确）
%       plot_mesh - 是否绘制网格 (可选, 默认true)
% 输出: Node    - 节点坐标 [NNode x 2]
%       Element - 单元连接 [NElem x 4] 四边形单元
%       Supp    - 支撑条件 [NSupp x 3]
%       Load    - 载荷条件 [NLoad x 3]
%
% 用法示例:
%   [Node, Element, Supp, Load] = Mesh_L_bracket_quad(100, 100);
%   实际单元数 = size(Element, 1)  % 精确可预测
%-------------------------------------------------------------------------%

if nargin < 3, plot_mesh = true; end

%% 1. 几何参数
L = 1.0;          % L型梁特征长度
t = 2*L/5;        % 厚度 = 0.4
box = [0, L, 0, L];  % 大矩形边界

%% 2. 生成完整矩形网格的节点
NN = (nx + 1) * (ny + 1);
x = linspace(box(1), box(2), nx + 1);
y = linspace(box(3), box(4), ny + 1);
[X, Y] = meshgrid(x, y);
Node_full = [X(:), Y(:)];  % [NN x 2]

%% 3. 生成完整矩形网格的单元
% 节点索引矩阵 (对应Python的idx)
idx = reshape(1:NN, ny + 1, nx + 1);  % 注意：MATLAB是列优先

% 生成单元 (对应Python的cell)
% 每个单元的4个节点：[左下, 右下, 右上, 左上]
n1 = idx(1:end-1, 1:end-1);  % 左下
n2 = idx(1:end-1, 2:end);    % 右下
n3 = idx(2:end, 2:end);      % 右上
n4 = idx(2:end, 1:end-1);    % 左上
Element_full = [n1(:), n2(:), n3(:), n4(:)];  % [nx*ny x 4]

%% 4. 应用threshold移除右上角单元
% 计算单元中心点 (对应Python的bc)
bc = (Node_full(Element_full(:,1), :) + ...
      Node_full(Element_full(:,2), :) + ...
      Node_full(Element_full(:,3), :) + ...
      Node_full(Element_full(:,4), :)) / 4;

% Threshold函数：判断是否在要移除的区域
% 返回true的单元将被删除
isDelCell = (bc(:,1) >= t) & (bc(:,1) <= L) & ...
            (bc(:,2) >= t) & (bc(:,2) <= L);

% 保留L型区域的单元
Element = Element_full(~isDelCell, :);
NElem = size(Element, 1);

%% 5. 重新编号节点（移除未使用的节点）
isValidNode = false(NN, 1);
isValidNode(Element(:)) = true;  % 标记被使用的节点

% 创建节点映射 (对应Python的idxMap)
Node = Node_full(isValidNode, :);
NNode = size(Node, 1);

idxMap = zeros(NN, 1);
idxMap(isValidNode) = 1:NNode;

% 更新单元的节点编号
Element = idxMap(Element);

%% 6. 边界条件
eps_tol = 0.1 * sqrt((L*L) / NNode);

% 6.1 支撑：固定顶边
TopEdgeNodes = find(abs(Node(:,2) - L) < eps_tol);
Supp = [TopEdgeNodes, ones(length(TopEdgeNodes), 2)];

% 6.2 载荷：右边界上部（y > 0.85*t）
RightNodes = find(abs(Node(:,1) - L) < eps_tol & ...
                  Node(:,2) > 0.85*t);
NLoad = length(RightNodes);
Load = [RightNodes, zeros(NLoad, 1), -2.0/NLoad*ones(NLoad, 1)];

%% 7. 输出信息
fprintf('生成L型梁网格 (精确版本):\n');
fprintf('  网格剖分: nx=%d, ny=%d\n', nx, ny);
fprintf('  节点数: %d\n', NNode);
fprintf('  单元数: %d (精确)\n', NElem);
fprintf('  支撑节点数: %d\n', length(TopEdgeNodes));
fprintf('  载荷节点数: %d\n', NLoad);

%% 8. 可视化
if plot_mesh
    PlotQuadMesh(Node, Element, NElem, Supp, Load);
end

end

%% ============== 可视化函数 ==============
function PlotQuadMesh(Node, Element, NElem, Supp, Load)
figure('Name', 'L型梁网格', 'NumberTitle', 'off');
clf; axis equal; axis off; hold on;

patch('Faces', Element, 'Vertices', Node, ...
      'FaceColor', 'w', 'EdgeColor', 'k', 'LineWidth', 0.5);

idx = 1; h = []; leg = {};

if exist('Supp','var') && ~isempty(Supp)
    h(idx) = plot(Node(Supp(:,1), 1), Node(Supp(:,1), 2), ...
                  'b>', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    leg{idx} = 'Support';
    idx = idx + 1;
end

if exist('Load','var') && ~isempty(Load)
    h(idx) = plot(Node(Load(:,1), 1), Node(Load(:,1), 2), ...
                  'm^', 'MarkerSize', 8, 'MarkerFaceColor', 'm');
    leg{idx} = 'Load';
end

if ~isempty(h)
    legend(h, leg, 'Location', 'NorthOutside', ...
           'Orientation', 'Horizontal');
end

title(sprintf('L型梁网格 (%d 单元, %d 节点)', NElem, size(Node,1)), ...
      'FontSize', 12, 'FontWeight', 'bold');
drawnow; pause(1e-6);

end