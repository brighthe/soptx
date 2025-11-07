function [F, U, K] = FEA(nelx, nely, x, penal, KE)
    K = sparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1));
    F = sparse(2*(nely+1)*(nelx+1),1); U = zeros(2*(nely+1)*(nelx+1),1);
    
    for elx = 1:nelx
      for ely = 1:nely
        n1 = (nely+1)*(elx-1)+ely; 
        n2 = (nely+1)* elx   +ely;
        edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
        K(edof, edof) = K(edof, edof) + x(ely, elx)^penal*KE;
      end
    end

    % --- 定义 MBB 梁的载荷和边界条件 ---
    % 载荷 1500 N
    % 载荷施加在左上角 (节点 nely+1) 的 Y 方向 
    load_node = (nely+1); 
    F(2*load_node, 1) = -1500.0; % [N]
    
    % 边界条件 (BCs) 
    % 左边界 (对称) - X 方向固定
    fixeddofs_x = 1:2:2*(nely+1);
    % 右下角 (铰支) - Y 方向固定
    node_bottom_right = (nelx)*(nely+1) + 1;
    fixeddofs_y = 2*node_bottom_right;
    
    fixeddofs = unique([fixeddofs_x, fixeddofs_y]);

    alldofs     = [1:2*(nely+1)*(nelx+1)];
    freedofs    = setdiff(alldofs, fixeddofs);
    % SOLVING
    U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);      
    U(fixeddofs,:)= 0;
end