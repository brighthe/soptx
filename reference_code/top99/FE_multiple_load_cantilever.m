%%%%%%%%%% FE-Aanlysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U] = FE_multiple_load(nelx, nely, x, penal)
    [KE] = lk;
    K = sparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1));
    % 节点载荷和节点位移都要变成 2
    F = sparse(2*(nely+1)*(nelx+1), 2); U = zeros(2*(nely+1)*(nelx+1), 2);
    for elx = 1:nelx
        for ely = 1:nely
            n1 = (nely+1)*(elx-1)+ely;
            n2 = (nely+1)* elx   +ely;
            edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
            K(edof, edof) = K(edof, edof) + x(ely, elx)^penal*KE;
        end
    end
    % Define Loads And Suppotrs (Multiple Load)
    F(2*(nelx+1)*(nely+1), 1) = -1.;
    F(2*(nelx)*(nely+1)+2, 2) = 1.;
    fixeddofs = [1:2*(nely+1)];
    alldofs = [1:2*(nely+1)*(nelx+1)];
    freedofs = setdiff(alldofs, fixeddofs);
    % SOLVING
    U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);
    U(fixeddofs,:)= 0;
end