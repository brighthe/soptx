%----------------------------- PolyStress --------------------------------%
% 终极修正版 MbbDomain
% 1. PFix 返回空，避免干扰结构化网格
% 2. 距离函数增加 -1e-5 的强力容差，防止边界节点被误删
%-------------------------------------------------------------------------%
function [x] = MbbDomain(Demand, Arg)
  
  L = 60; H = 20;

  switch(Demand)
    case('Dist')
        P = Arg;
        % 【关键修正】减去 1e-5 (10微米) 的容差
        % 这让边界点的距离变成负数 (d = -1e-5)，绝对被判定为"内部"
        x = dRectangle(P, 0, L, 0, H) - 1e-5;
        
    case('BdBox')
        x = [0, L, 0, H];
        
    case('PFix')
        % 【关键修正】返回空矩阵。
        % 结构化种子点会自动生成完美的角点，不需要 PFix 强制指定。
        % 返回 [] 可以骗过 PolyMesher 的检查，同时避免节点冲突。
        x = [];
        
    case('BC')
        % 兼容性处理
        if iscell(Arg); Node = Arg{1}; else; Node = Arg; end

        tol = 1e-4;

        % 1. 左侧对称 (x=0)
        LeftNodes = find(abs(Node(:,1)) < tol);
        Supp1 = [LeftNodes, ones(length(LeftNodes),1), zeros(length(LeftNodes),1)];

        % 2. 右下角滑移 (x=L, y=0)
        RightNode = find(abs(Node(:,1)-L) < tol & abs(Node(:,2)) < tol);
        Supp2 = [RightNode, zeros(length(RightNode),1), ones(length(RightNode),1)];

        Supp = [Supp1; Supp2];

        % 3. 左上角载荷 (x=0, y=H)
        LoadNode = find(abs(Node(:,1)) < tol & abs(Node(:,2)-H) < tol);
        Load = [LoadNode, 0, -400];

        x = {Supp, Load};
  end
end

% 距离函数
function d = dRectangle(P, x1, x2, y1, y2)
    d = -min(min(min(-y1+P(:,2), y2-P(:,2)), -x1+P(:,1)), x2-P(:,1));
end
%-------------------------------------------------------------------------%