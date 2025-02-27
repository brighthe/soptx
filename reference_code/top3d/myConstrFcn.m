function [cneq, ceq, gradc, gradceq] = myConstrFcn(x, H, Hs, volfrac, nele)
    xPhys(:) = (H*x(:))./Hs;
    % Non-linear Constraints
    cneq  = sum(xPhys(:)) - volfrac*nele;
    gradc = ones(nele,1);
    % Linear Constraints
    ceq     = [];
    gradceq = [];
end % mycon
