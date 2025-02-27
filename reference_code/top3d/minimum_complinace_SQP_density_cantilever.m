global ce;  % 声明全局变量 ce
ce = [];    % 初始化 ce

nelx = 30;
nely = 10;
nelz = 2;
volfrac = 0.3;
penal = 3;
rmin = 1.2;
% USER-DEFINED LOOP PARAMETERS
maxloop = 200;    % Maximum number of iterations
tolx = 0.01;      % Terminarion criterion
displayflag = 0;  % Display structure flag
% USER-DEFINED MATERIAL PROPERTIES
E0 = 1;           % Young's modulus of solid material
Emin = 1e-9;      % Young's modulus of void-like material
nu = 0.3;         % Poisson's ratio
% USER-DEFINED LOAD DOFs
[il, jl, kl] = meshgrid(nelx, 0, 0:nelz);                 % Coordinates
loadnid = kl*(nelx+1)*(nely+1)+il*(nely+1)+(nely+1-jl); % Node IDs
loaddof = 3*loadnid(:) - 1;                             % DOFs
% USER-DEFINED SUPPORT FIXED DOFs
[iif, jf, kf] = meshgrid(0, 0:nely, 0:nelz);                  % Coordinates
fixednid = kf*(nelx+1)*(nely+1)+iif*(nely+1)+(nely+1-jf); % Node IDs
fixeddof = [3*fixednid(:); 3*fixednid(:)-1; 3*fixednid(:)-2]; % DOFs

% PREPARE FINITE ELEMENT ANALYSIS
nele = nelx*nely*nelz;
ndof = 3*(nelx+1)*(nely+1)*(nelz+1);
F = sparse(loaddof,1,-1, ndof,1);
U = zeros(ndof, 1);
freedofs = setdiff(1:ndof, fixeddof);
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
                        sH(k) = max(0, rmin-sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2));
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

% 定义 fmincon 的参数范围和选项
A = [];
B = [];
Aeq = [];
Beq = [];
LB = zeros(size(x));
UB = ones(size(x));

% 设置优化选项
OPTIONS = optimset('TolX', tolx, 'MaxIter', maxloop, 'Algorithm', 'interior-point', ...
    'GradObj', 'on', 'GradConstr', 'on', 'Hessian', 'user-supplied', ...
    'HessFcn', @(x, lambda) myHessianFcn(x, lambda, H, Hs, penal, E0, Emin, nelx, nely, nelz), ...
    'Display', 'none', ...
    'OutputFcn', @(x, optimValues, state) myOutputFcn(x, optimValues, state, displayflag, nelx, nely, nelz), ...
    'PlotFcns', @optimplotfval);

% 调用 fmincon 进行优化
fmincon(@(x) myObjFcn(x, H, Hs, KE, penal, E0, Emin, nelx, nely, nelz, edofMat, freedofs, iK, jK, F), ...
                    x, A, B, Aeq, Beq, LB, UB, ...
                    @(x) myConstrFcn(x, H, Hs, volfrac, nele), ...
                    OPTIONS);


