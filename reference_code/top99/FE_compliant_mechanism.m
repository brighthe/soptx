function [U, Uout] = FE_compliant_mechanism(nelx, nely, x, penal)
    [KE] = lk;
    K = sparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1));
    F = sparse(2*(nely+1)*(nelx+1), 2); U = zeros(2*(nely+1)*(nelx+1), 2);
    for elx = 1:nelx
        for ely = 1:nely
            n1 = (nely+1)*(elx-1)+ely;
            n2 = (nely+1)* elx   +ely;
            edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
            K(edof, edof) = K(edof, edof) + x(ely, elx)^penal*KE;
        end
    end
    K1_dense = full(K);
    % Define Loads And Suppotrs (Half-Force Inverter)
    din = 1;
    dout = 2*nelx*(nely+1)+1;
    F(din, 1) = 1;
    F(dout, 2) = -1;
    K(din, din) = K(din, din) + 0.1;
    K(dout, dout) = K(dout, dout) + 0.1;
    K2_dense = full(K);
    fixeddofs = union([2:2*(nely+1):2*(nely+1)*(nelx+1)], [2*(nely+1):-1:2*(nely+1)-3]);
    alldofs = [1:2*(nely+1)*(nelx+1)];
    freedofs = setdiff(alldofs, fixeddofs);
    % SOLVING
    U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);
    U(fixeddofs,:)= 0;
    Uout = U(dout, 1);
end