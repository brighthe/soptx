%%%%%%%%%% FE-Aanlysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U] = FE_heat_conduction(nelx, nely, x, penal)
    [KE] = lk_heat_conduction;
    K = sparse((nelx+1)*(nely+1), (nelx+1)*(nely+1));
    F = sparse((nely+1)*(nelx+1), 1); U = zeros((nely+1)*(nelx+1), 1);
    for elx = 1:nelx
        for ely = 1:nely
            n1 = (nely+1)*(elx-1)+ely;
            n2 = (nely+1)* elx   +ely;
            edof = [n1; n2; n2+1; n1+1];
            K(edof, edof) = K(edof, edof) + (0.001+0.999*x(ely, elx)^penal)*KE;
        end
    end
    % Define Loads And Suppotrs (Square Plate With Heat Sink)
    F(:, 1) = 0.01;
    fixeddofs = [nely/2+1-(nely/20):nely/2+1+(nely/20)];
    alldofs = [1:(nely+1)*(nelx+1)];
    freedofs = setdiff(alldofs, fixeddofs);
    % SOLVING
    U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);
    U(fixeddofs,:)= 0;
end