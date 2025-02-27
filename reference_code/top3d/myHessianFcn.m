function h = myHessianFcn(x, lambda, H, Hs, penal, E0, Emin, nelx, nely, nelz)
    global ce;  % 使用全局变量 ce
    xPhys = reshape(x,nely,nelx,nelz);
    % Compute Hessian of Obj.
    Hessf = 2*(penal*(E0-Emin)*xPhys.^(penal-1)).^2 ./ (E0 + (E0-Emin)*xPhys.^penal) .* ce;
    Hessf(:) = H*(Hessf(:)./Hs);
    % Compute Hessian of constraints
    Hessc = 0; % Linear constraint
    % Hessian of Lagrange
    h = diag(Hessf(:)) + lambda.ineqnonlin*Hessc;
end % myHessianFcn

