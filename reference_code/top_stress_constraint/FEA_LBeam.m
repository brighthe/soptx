function [F, U, K, fixeddofs] = FEA_LBeam(nelx, nely, xPhys, penal_K, KE)
    K = sparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1));
    F = sparse(2*(nely+1)*(nelx+1),1); U = zeros(2*(nely+1)*(nelx+1),1);
    
    for elx = 1:nelx
      for ely = 1:nely
        n1 = (nely+1)*(elx-1)+ely; 
        n2 = (nely+1)* elx   +ely;
        edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
        K(edof, edof) = K(edof, edof) + xPhys(ely, elx)^penal_K*KE;
      end
    end

    % --- 定义 L 形梁的载荷和边界条件 ---
    % 载荷：施加在右侧中间位置，向下
    load_x = nelx + 1;              % 右边界
    load_y = round(nely * 2/5);     % 高度的2/5位置（根据图片）
    load_node = (nely+1)*(load_x-1) + load_y;
    F(2*load_node, 1) = -1500.0;    % 载荷1500 N（论文值）
    
    % 边界条件：左上角完全固定（固定支座）
    % 固定左边界顶部的所有自由度
    fixed_x = 1;                    % 左边界
    fixed_y_start = round(nely * 3/5);  % 从3/5高度开始
    fixed_y_end = nely + 1;         % 到顶部
    
    fixeddofs = [];
    for y = fixed_y_start:fixed_y_end
        node = (nely+1)*(fixed_x-1) + y;
        fixeddofs = [fixeddofs, 2*node-1, 2*node];  % X和Y方向都固定
    end

    fixeddofs   = unique(fixeddofs);
    alldofs     = [1:2*(nely+1)*(nelx+1)];
    freedofs    = setdiff(alldofs, fixeddofs);

    % SOLVING
    U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);      
    U(fixeddofs,:)= 0;
end