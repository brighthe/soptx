function [Node, Element, Supp, Load] = Mesh_L_bracket(nelx, nely)
% MESH_L_BRACKET_QUAD 生成 L 型区域的四边形网格
% 输入:
%   nelx: X 方向总单元数 (建议 40)
%   nely: Y 方向总单元数 (建议 40)
% 输出:
%   Node:    [NNode x 2] 节点坐标矩阵
%   Element: {NElem x 1} 单元连接关系 (Cell 数组，适配 PolyStress)
%   Supp:    [NSupp x 3] 支撑矩阵 [NodeID, DOF, Value]
%   Load:    [NLoad x 3] 载荷矩阵 [NodeID, DOF, Value]

    %% 1. 定义几何与离散化参数
    L = 1.0;          % 总长度
    H = 1.0;          % 总高度
    thick_ratio = 0.4; % L型腿的厚度比例 (例如 0.4 表示 40% 厚度)
    
    dx = L / nelx;
    dy = H / nely;
    
    % 确定 L 型的逻辑区域
    % 我们通过扫描网格点来确定哪些单元是"激活"的
    % 定义：左侧竖腿 + 底部横腿
    limit_x = floor(nelx * thick_ratio);
    limit_y = floor(nely * thick_ratio);
    
    %% 2. 生成节点和单元
    % 这里使用一种简单的方法：先生成全矩形的节点映射，然后剔除空洞
    
    NodeMap = zeros(nely+1, nelx+1); % 用于存储节点的新 ID
    node_counter = 0;
    NodeList = [];
    
    % --- 生成节点 (Node) ---
    for j = 0:nely
        for i = 0:nelx
            % 几何判断：是否在 L 型区域内？
            % 区域定义：x <= limit_x (左腿) 或者 y <= limit_y (底腿)
            if (i <= limit_x) || (j < limit_y) 
                % 注意：这里用 j < limit_y 是为了处理交界处的逻辑，具体视网格对齐而定
                % 简单的逻辑：如果是右上角的空洞区域，则跳过
                if (i > limit_x) && (j >= limit_y)
                    continue;
                end
                
                node_counter = node_counter + 1;
                NodeMap(j+1, i+1) = node_counter;
                % 坐标变换: i -> x, j -> y (注意 y 轴方向，通常 FEM y向上)
                NodeList(node_counter, :) = [i*dx, j*dy];
            end
        end
    end
    Node = NodeList;
    
    % --- 生成单元 (Element) ---
    % 标准 Q4 单元连接顺序：左下 -> 右下 -> 右上 -> 左上 (逆时针)
    elem_counter = 0;
    ElemConnect = {}; % 使用 Cell 数组以直接适配 PolyStress
    
    for j = 1:nely
        for i = 1:nelx
            % 获取该单元四个角点在 NodeMap 中的索引
            % 网格索引 (j,i) 对应 NodeMap 的索引需注意
            % 节点索引：n1(j,i), n2(j,i+1), n3(j+1,i+1), n4(j+1,i)
            
            % 检查这四个节点是否都存在（即 ID > 0）
            n1 = NodeMap(j, i);
            n2 = NodeMap(j, i+1);
            n3 = NodeMap(j+1, i+1);
            n4 = NodeMap(j+1, i);
            
            if (n1 > 0) && (n2 > 0) && (n3 > 0) && (n4 > 0)
                elem_counter = elem_counter + 1;
                % 保存连接关系
                ElemConnect{elem_counter, 1} = [n1, n2, n3, n4];
            end
        end
    end
    Element = ElemConnect;
    
    %% 3. 定义边界条件 (Supp)
    % 标准 L-bracket：上端固定
    % 找到所有 y = H 的节点
    fixed_nodes = find(abs(Node(:,2) - H) < 1e-6);
    
    % 格式: [NodeID, DOF(1=x, 2=y), Value]
    Supp = [];
    for k = 1:length(fixed_nodes)
        node_id = fixed_nodes(k);
        Supp = [Supp; 
                node_id, 1, 0;  % Fix X
                node_id, 2, 0]; % Fix Y
    end
    
    %% 4. 定义载荷 (Load)
    % 标准 L-bracket：右下端点受集中力向下
    % 找到 x = L 且 y = (横腿中点或上沿) 的节点
    % 这里选横腿的最右上角点： x=L, y=limit_y*dy
    load_y = limit_y * dy;
    
    % 寻找距离加载点最近的节点
    target_x = L;
    target_y = load_y; % 或者选中点 0.5*load_y
    
    dist = sum((Node - [target_x, target_y]).^2, 2);
    [~, load_node] = min(dist);
    
    P_val = -1.0; % 向下的力
    Load = [load_node, 2, P_val]; % [NodeID, DOF=2(y), Value]
    
    %% 5. (可选) 绘图验证
    figure(101); clf; axis equal; hold on; title('Generated Quad Mesh');
    patch('Faces', cell2mat(Element), 'Vertices', Node, ...
          'FaceColor', 'w', 'EdgeColor', 'k');
    plot(Node(fixed_nodes,1), Node(fixed_nodes,2), 'r^', 'MarkerFaceColor','r');
    plot(Node(load_node,1), Node(load_node,2), 'bo', 'MarkerFaceColor','b');
    xlabel('X'); ylabel('Y');
    set(gca, 'XLim', [-0.1 1.1], 'YLim', [-0.1 1.1]);
    drawnow;
    
end