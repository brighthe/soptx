%----------------------------- PolyStress --------------------------------%
% 博士论文专用：MBB 梁结构化网格直接生成器
% 直接构建拓扑关系，彻底避开 PolyMesher 的浮点数简并问题
%-------------------------------------------------------------------------%
function [Node, Element, Supp, Load] = Mesh_MBB(Ne_ap)

    % 1. 几何参数
    L = 60; 
    H = 20;
    
    % 2. 计算网格划分 (保证单元接近正方形)
    ny = round(sqrt(Ne_ap/3)); 
    nx = 3 * ny;
    NElem = nx * ny;
    
    dx = L / nx;
    dy = H / ny;
    
    fprintf('正在生成结构化网格: %d x %d (总单元数: %d)...\n', nx, ny, NElem);
    
    % 3. 生成节点 (Node)
    % 节点编号顺序：先y后x (列优先)，这符合 MATLAB 的习惯
    [Y, X] = meshgrid(0:dy:H, 0:dx:L);
    Node = [X(:), Y(:)];
    NNode = size(Node, 1);
    
    % 4. 生成单元连接 (Element) - 关键步骤
    % 只要这里的数学逻辑对，就永远不会报错
    Element = cell(NElem, 1);
    nny = ny + 1; % y方向节点数
    
    cnt = 1;
    for i = 1:nx
        for j = 1:ny
            % 计算当前单元四个角点的索引
            % n1 --- n4
            % |      |
            % n2 --- n3
            
            % 左下角节点索引 (注意：meshgrid生成后展平的索引规律)
            n1 = (i-1)*nny + j;       % Top-Left (因为y是从0到H)
            n2 = n1 + 1;              % Bottom-Left
            n3 = n1 + nny + 1;        % Bottom-Right
            n4 = n1 + nny;            % Top-Right
            
            % PolyStress 里的习惯通常是逆时针
            % 实际上对于矩形，只要顺序是环绕的即可
            % 这里调整为：左下 -> 右下 -> 右上 -> 左上
            Element{cnt} = [n2, n3, n4, n1]; 
            
            cnt = cnt + 1;
        end
    end
    
    % 5. 定义边界条件 (Supp) - 直接查找坐标
    tol = 1e-5;
    
    % 5.1 左侧对称约束 (x=0, fix Ux)
    LeftNodes = find(abs(Node(:,1)) < tol);
    Supp1 = [LeftNodes, ones(length(LeftNodes),1), zeros(length(LeftNodes),1)];
    
    % 5.2 右下角滑移约束 (x=L, y=0, fix Uy)
    RightNode = find(abs(Node(:,1)-L) < tol & abs(Node(:,2)) < tol);
    Supp2 = [RightNode, zeros(length(RightNode),1), ones(length(RightNode),1)];
    
    Supp = [Supp1; Supp2];
    
    % 6. 定义载荷 (Load)
    % 左上角 (x=0, y=H)
    LoadNode = find(abs(Node(:,1)) < tol & abs(Node(:,2)-H) < tol);
    Load = [LoadNode, 0, -400];
    
end
%-------------------------------------------------------------------------%