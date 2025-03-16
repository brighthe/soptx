%% 参数设置
nelx = 40;
nely = 40;
volfrac = 0.4;
penal = 3.0;
rmin = 1.2;
ft = 1;     % 密度滤波器
% ft = 2;   % 灵敏度滤波器

% USER-DEFINED LOOP PARAMETERS
maxloop = 200;    % Maximum number of iterations
tolx = 0.01;      % Termination criterion
displayflag = 1;  % Display structure flag

%% MATERIAL PROPERTIES
k0 = 1;           % Good thermal conductivity
kmin = 1e-3;      % Poor thermal conductivity

%% PREPARE FINITE ELEMENT ANALYSIS
% 构建热传导元素刚度矩阵 (KE)
KE = [2/3 -1/6 -1/3 -1/6
     -1/6  2/3 -1/6 -1/3
     -1/3 -1/6  2/3 -1/6
     -1/6 -1/3 -1/6  2/3];

% 自由度处理 (每个节点只有1个温度自由度)
nodenrs = reshape(1:(1+nelx)*(1+nely), 1+nely, 1+nelx);
edofVec = reshape(nodenrs(1:end-1, 1:end-1), nelx*nely, 1);
edofMat = repmat(edofVec, 1, 4) + repmat([0 1 nely+1 nely+2], nelx*nely, 1);
iK = reshape(kron(edofMat, ones(4,1))', 16*nelx*nely, 1);
jK = reshape(kron(edofMat, ones(1,4))', 16*nelx*nely, 1);

%% DEFINE LOADS AND SUPPORTS (热传导边界条件)
% 定义热源 (整个区域均匀加热)
F = sparse((nely+1)*(nelx+1), 1);
F(:, 1) = 0.01;

% 定义热沉 (中心区域固定温度)
fixeddofs = [nely/2+1-(nely/20):nely/2+1+(nely/20)];
alldofs = [1:(nely+1)*(nelx+1)];
freedofs = setdiff(alldofs, fixeddofs);

% 求解准备
U = zeros((nely+1)*(nelx+1), 1);

%% PREPARE FILTER
iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2, 1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
            for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                e2 = (i2-1)*nely+j2;
                k = k+1;
                iH(k) = e1;
                jH(k) = e2;
                sH(k) = max(0, rmin - sqrt((i1-i2)^2 + (j1-j2)^2));
            end
        end
    end
end
H = sparse(iH, jH, sH);
Hs = sum(H, 2);

%% INITIALIZE ITERATION
x = repmat(volfrac, nely, nelx);
xPhys = x;
loop = 0;
change = 1;

%% START ITERATION
while change > tolx && loop < maxloop
    loop = loop + 1;
    
    %% FE-ANALYSIS
    sK = reshape(KE(:) * (kmin + xPhys(:)'.^penal * (k0-kmin)), 16*nelx*nely, 1);
    K = sparse(iK, jK, sK); K = (K+K') / 2;
    U(freedofs) = K(freedofs, freedofs) \ F(freedofs);
    
    %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    ce = reshape(sum((U(edofMat)*KE).*U(edofMat), 2), nely, nelx);
    c = sum(sum((kmin + xPhys.^penal * (k0-kmin)).*ce));
    dc = -penal * (k0-kmin) * xPhys.^(penal-1) .* ce;
    dv = ones(nely, nelx);
    
    %% FILTERING/MODIFICATION OF SENSITIVITIES
    if ft == 1
        dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3, x(:));
    elseif ft == 2
        dc(:) = H*(dc(:)./Hs);
        dv(:) = H*(dv(:)./Hs);
    end
    
    %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    l1 = 0; l2 = 1e9; move = 0.2;
    while (l2-l1)/(l1+l2) > 1e-3
        lmid = 0.5*(l2+l1);
        xnew = max(0, max(x-move, min(1, min(x+move, x.*sqrt(-dc./dv/lmid)))));
        if ft == 1
            xPhys = xnew;
        elseif ft == 2
            xPhys(:) = (H*xnew(:))./Hs;
        end
        if sum(xPhys(:)) > volfrac*nelx*nely, l1 = lmid; else l2 = lmid; end
    end
    
    change = max(abs(xnew(:) - x(:)));
    x = xnew;
    
    %% PRINT RESULTS
    fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n', loop, c, mean(xPhys(:)),change);
    %% PLOT DENSITIES
    colormap(gray); imagesc(1-xPhys); caxis([0 1]); axis equal; axis off; drawnow;
end