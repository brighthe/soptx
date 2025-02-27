nelx = 60;
nely = 20;
nelz = 4;
volfrac = 0.3;
penal = 3;
rmin = 1.5;
% ft = 1;     % 密度滤波器
ft = 2;   % 灵敏度滤波器

% USER-DEFINED LOOP PARAMETERS
maxloop = 500;    % Maximum number of iterations
tolx = 0.01;      % Terminarion criterion
displayflag = 0;  % Display structure flag

% USER-DEFINED MATERIAL PROPERTIES
E0 = 1;           % Young's modulus of solid material
Emin = 1e-9;      % Young's modulus of void-like material
nu = 0.3;         % Poisson's ratio

% USER-DEFINED LOAD DOFs
[il, jl, kl] = meshgrid(nelx, 0, 0:nelz);                 % Coordinates
loadnid = kl*(nelx+1)*(nely+1)+il*(nely+1)+(nely+1-jl);   % Node IDs
loaddof = 3*loadnid(:) - 1;                               % DOFs

% USER-DEFINED SUPPORT FIXED DOFs
[iif, jf, kf] = meshgrid(0, 0:nely, 0:nelz);                  % Coordinates
fixednid = kf*(nelx+1)*(nely+1)+iif*(nely+1)+(nely+1-jf);     % Node IDs
fixeddof = [3*fixednid(:); 3*fixednid(:)-1; 3*fixednid(:)-2]; % DOFs

% PREPARE FINITE ELEMENT ANALYSIS
nele = nelx*nely*nelz;
ndof = 3*(nelx+1)*(nely+1)*(nelz+1);
F = sparse(loaddof,1,-1,ndof,1);
U = zeros(ndof, 1);
freedofs = setdiff(1:ndof,fixeddof);
KE = lk_H8(nu);
nodegrd = reshape(1:(nely+1)*(nelx+1),nely+1,nelx+1);
nodeids = reshape(nodegrd(1:end-1,1:end-1),nely*nelx,1);
nodeidz = 0:(nely+1)*(nelx+1):(nelz-1)*(nely+1)*(nelx+1);
nodeids = repmat(nodeids,size(nodeidz))+repmat(nodeidz,size(nodeids));
edofVec = 3*nodeids(:)+1;
edofMat = repmat(edofVec,1,24)+ ...
    repmat([0 1 2 3*nely + [3 4 5 0 1 2] -3 -2 -1 ...
    3*(nely+1)*(nelx+1)+[0 1 2 3*nely + [3 4 5 0 1 2] -3 -2 -1]],nele,1);
iK = reshape(kron(edofMat,ones(24,1))',24*24*nele,1);
jK = reshape(kron(edofMat,ones(1,24))',24*24*nele,1);

% PREPARE FILTER
iH = ones(nele*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for k1 = 1:nelz
    for i1 = 1:nelx
        for j1 = 1:nely
            e1 = (k1-1)*nelx*nely + (i1-1)*nely+j1;
            for k2 = max(k1-(ceil(rmin)-1),1):min(k1+(ceil(rmin)-1),nelz)
                for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
                    for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                        e2 = (k2-1)*nelx*nely + (i2-1)*nely+j2;
                        k = k+1;
                        iH(k) = e1;
                        jH(k) = e2;
                        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2));
                    end
                end
            end
        end
    end
end
H = sparse(iH, jH, sH);
Hs = sum(H, 2);

% INITIALIZE ITERATION
x = repmat(volfrac, [nely,nelx,nelz]);
xPhys = x; 
loop = 0; 
change = 1;

% INITIALIZE MMA OPTIMIZER
m     = 1;                   % The number of general constraints.
n     = nele;                % The number of design variables x_j.
xmin  = zeros(n, 1);         % Column vector with the lower bounds for the variables x_j.
xmax  = ones(n, 1);          % Column vector with the upper bounds for the variables x_j.
xold1 = x(:);                % xval, one iteration ago (provided that iter>1).
xold2 = x(:);                % xval, two iterations ago (provided that iter>2).
low   = ones(n, 1);          % Column vector with the lower asymptotes from the previous iteration (provided that iter>1).
upp   = ones(n, 1);          % Column vector with the upper asymptotes from the previous iteration (provided that iter>1).
a0    = 1;                   % The constants a_0 in the term a_0*z.
a     = zeros(m, 1);         % Column vector with the constants a_i in the terms a_i*z.
c_MMA = 10000 * ones(m, 1);  % Column vector with the constants c_i in the terms c_i*y_i.
d     = zeros(m, 1);         % Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.

% START ITERATION
while change > tolx && loop < maxloop
    loop = loop + 1;
   
    % FE-ANALYSIS
    sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),24*24*nele,1);
    K = sparse(iK,jK,sK); 
    K = (K+K')/2;
    U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);

    % fprintf('xPhys: %11.10f\n', mean(xPhys(:)));
    % fprintf('K: %11.10f\n', sum(sum(abs(KFULL))));
    % fprintf('u: %11.10f\n', mean(U(:)));

    % OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    ce = reshape(sum((U(edofMat)*KE).*U(edofMat),2), [nely, nelx, nelz]);
    c = sum(sum(sum((Emin + xPhys .^ penal * (E0 - Emin)) .* ce)));
    dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;
    dv = ones(nely,nelx,nelz);
    
    % FILTERING AND MODIFICATION OF SENSITIVITIES
    if ft == 1
        dc(:) = H*(dc(:)./Hs);
        dv(:) = H*(dv(:)./Hs);
    elseif ft == 2
        dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3,x(:));
    end
    
    % METHOD OF MOVING ASYMPTOTES
    xval  = x(:);
    f0val = c;
    df0dx = dc(:);
    fval  = sum(xPhys(:)) / (volfrac*nele) - 1;
    dfdx  = dv(:)' / (volfrac*nele);
    [xmma, ~, ~, ~, ~, ~, ~, ~, ~, low, upp] = ...
            mmasub(m, n, loop, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c_MMA, d);
    % Update MMA Variables
    xnew     = reshape(xmma, nely, nelx, nelz);
    % if loop == 14 || loop == 11 || loop == 12 || loop == 13 || loop == 36 || loop == 37 || loop == 38
    %     fprintf('xPhys: %11.10f\n', mean(xPhys(:)));
    %     fprintf('u: %11.10f\n', mean(U(:)));
    %     fprintf('fval: %11.10f\n', mean(fval(:)))
    %     fprintf('xnew: %11.10f\n', mean(xnew(:)))
    %     fprintf('------------------')
    % end
    % if loop == 106 || loop == 107 || loop == 108 || loop == 109
    %     fprintf('xPhys: %11.10f\n', mean(xPhys(:)));
    %     fprintf('u: %11.10f\n', mean(U(:)));
    %     fprintf('fval: %11.10f\n', mean(fval(:)))
    %     fprintf('xnew: %11.10f\n', mean(xnew(:)))
    %     fprintf('------------------')
    % end
    if ft == 1
        xPhys(:) = (H*xnew(:))./Hs;
    elseif ft == 2
        xPhys = xnew;
    end
    xold2    = xold1(:);
    xold1    = x(:);

    change = max(abs(xnew(:)-x(:)));
    x = xnew;

    % PRINT RESULTS
    fprintf(' It.:%5i Obj.:%11.6f Vol.:%11.6f ch.:%11.6f\n', loop, c, mean(xPhys(:)), change);

end

clf;
display_3D(xPhys);