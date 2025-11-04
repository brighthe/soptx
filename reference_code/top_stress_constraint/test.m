%=====================================================================
% --- 2D PARAMETERS ---
%=====================================================================
nelx = 40;
nely = 40;
volfrac = 0.35; 
penal = 3;
rmin = 1.5;
% ft = 0;  % 无滤波器
ft = 1;     % 密度滤波器
% ft = 2;   % 灵敏度滤波器

%=====================================================================
% --- STRESS PARAMETERS ---
%=====================================================================
sigmay = 0.028; % 屈服应力 (Yield stress, from user's main.m)
p_stress = 6;   % p-norm 聚合参数 (Aggregation parameter)
q_stress = 0.5; % 应力松弛因子 (Stress relaxation, from user's stress_func.m)

% USER-DEFINED LOOP PARAMETERS
maxloop = 200;    % Maximum number of iterations
tolx = 0.01;      % Terminarion criterion

% MATERIAL PROPERTIES
E0 = 1;           % Young's modulus of solid material
Emin = 1e-9;      % Young's modulus of void-like material
nu = 0.3;         % Poisson's ratio
    
%=====================================================================
% --- 2D BCs ---
%=====================================================================
nele = nelx*nely;
ndof = 2*(nelx+1)*(nely+1);

% LOAD: (Bottom-right corner, Fy = -1)
loadnid = (nelx)*(nely+1) + 1; % Node ID
loaddof = 2*loadnid;           % Y-DOF
F = sparse(loaddof, 1, -1, ndof, 1);
    
% SUPPORT: (Fixed left edge)
fixednid = 1:(nely+1);                       % Node IDs for left edge
fixeddof = [2*fixednid(:)-1; 2*fixednid(:)]; % X and Y DOFs

% FE PREP
U = zeros(ndof, 1);
freedofs = setdiff(1:ndof,fixeddof);

% Use 2D element stiffness matrix
E = 1;
k = [ 1/2-nu/6   1/8+nu/8 -1/4-nu/12 -1/8+3*nu/8 ... 
    -1/4+nu/12 -1/8-nu/8  nu/6       1/8-3*nu/8];
KE = E/(1-nu^2)*[ k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
                  k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
                  k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
                  k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
                  k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
                  k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
                  k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
                  k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)]; 

%=====================================================================
% --- 2D FEA PREP (edofMat) ---
%=====================================================================
nodegrd = reshape(1:(nely+1)*(nelx+1),nely+1,nelx+1);
nodeids = reshape(nodegrd(1:end-1,1:end-1),nely*nelx,1);
edofVec = 2*nodeids(:)-1;
offsetVec = [0 1 2*(nely+1) 2*(nely+1)+1 2*(nely+1)+2 2*(nely+1)+3 2 3];
edofMat = repmat(edofVec,1,8) + repmat(offsetVec,nele,1);
iK = reshape(kron(edofMat,ones(8,1))', 8*8*nele,1);
jK = reshape(kron(edofMat,ones(1,8))', 8*8*nele,1);

%=====================================================================
% --- 2D FILTER PREP (H, Hs) ---
%=====================================================================
iH = ones(nele*(2*(ceil(rmin)-1)+1)^2,1);
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
                sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
            end
        end
    end
end
H = sparse(iH, jH, sH);
Hs = sum(H, 2);
    
%=====================================================================
% --- STRESS CALCULATION PREP (B, D matrix) ---
%=====================================================================
% B 矩阵 (在单元中心 xi=0, eta=0 处)
B_mat = 0.5 * [-1  0  1  0  1  0 -1  0;
                0 -1  0 -1  0  1  0  1;
               -1 -1 -1  1  1  1  1 -1];
% D 矩阵 (平面应力 Plane Stress)
D_mat = E0/(1-nu^2)*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];

% INITIALIZE ITERATION
x = repmat(volfrac, [nely,nelx]);
xPhys = x; 
loop = 0; 
change = 1;
    
%=====================================================================
% --- MMA PREP (m=2 CONSTRAINTS) ---
%=====================================================================
m     = 2;                   % 约束数量 (1=vol, 2=stress)
n     = nele;                % The number of design variables x_j.
xmin  = repmat(1e-3, n, 1);  % Lower bounds (use 1e-3, not 0)
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
    sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),8*8*nele,1);
    K = sparse(iK,jK,sK); 
    K = (K+K')/2;
    U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);

    % --- OBJECTIVE (Compliance) ---
    ce = reshape(sum((U(edofMat)*KE).*U(edofMat),2), [nely, nelx]);
    c = sum(sum((Emin + xPhys .^ penal * (E0 - Emin)) .* ce));
    
    % --- SENSITIVITY (Compliance) ---
    dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;
    
    % --- CONSTRAINT 1 (Volume) ---
    fval(1) = sum(xPhys(:)) / (volfrac*nele) - 1;
    dv = ones(nely,nelx);
    % Volume sensitivity (w.r.t. xPhys)
    dfdx1_phys = dv(:)' / (volfrac*nele);
    
    %=================================================================
    % --- STRESS CALCULATION (CONSTRAINT 2) ---
    %=================================================================
    sigma_vM = zeros(nele, 1);
    dfdx_stress_explicit = zeros(nele, 1);
    F_adj = sparse(ndof, 1); % Adjoint load
    
    % Small epsilon to prevent division by zero (from your previous code)
    epsilon = 1e-10; 

    for e = 1:nele
        Ue = U(edofMat(e,:));
        % (1) Calculate Element Stress
        sigma_e = D_mat * B_mat * Ue; % [sig_x, sig_y, tau_xy]
        
        % (2) Calculate von Mises Stress
        sigma_vM(e) = sqrt( sigma_e(1)^2 + sigma_e(2)^2 - sigma_e(1)*sigma_e(2) + 3*sigma_e(3)^2 );
        
        % (3) Calculate derivative: d(sigma_vM) / d(sigma_e)
        % (This is 'derivative_2' from your stress_func.m)
        dsigma_vM_dsigma = (1 / (2*sigma_vM(e) + epsilon)) * ...
                                 [ (2*sigma_e(1) - sigma_e(2));
                                   (2*sigma_e(2) - sigma_e(1));
                                   (6*sigma_e(3)) ]; % This is a 3x1 vector
        
        % (4) Calculate common term for sensitivities
        g_e_p_prime = p_stress * ( (xPhys(e)^q_stress * sigma_vM(e) / sigmay) + epsilon )^(p_stress - 1);

        % (5) Adjoint Load (for implicit sensitivity)
        % d(g_stress)/d(Ue) = d(g)/d(vM) * d(vM)/d(sig) * d(sig)/d(Ue)
        lambda_load_e_T = (1/nele) * g_e_p_prime * (xPhys(e)^q_stress / sigmay) * ...
                          dsigma_vM_dsigma' * D_mat * B_mat;
        
        F_adj(edofMat(e,:)) = F_adj(edofMat(e,:)) + lambda_load_e_T';
        
        % (6) Explicit Sensitivity
        % d(g_stress)/d(x_e)
        dfdx_stress_explicit(e) = (1/nele) * g_e_p_prime * ...
              ( q_stress * xPhys(e)^(q_stress-1) * sigma_vM(e) / sigmay );
    end
    
    % (7) Solve Adjoint System
    U_adj = zeros(ndof, 1);
    U_adj(freedofs,:) = K(freedofs,freedofs) \ F_adj(freedofs,:);
    
    % (8) Implicit Sensitivity
    % d(K)/d(x_e) * Ue
    dK_dx_U = (penal*(E0-Emin)*xPhys(:).^(penal-1)) .* ...
              reshape(sum((U(edofMat)*KE).*U_adj(edofMat),2), [nele, 1]);
    
    dfdx_stress_implicit = -dK_dx_U;
    
    % (9) Total Stress Constraint & Sensitivity
    fval(2) = sum( ( (xPhys(:).^q_stress .* sigma_vM) / sigmay ).^p_stress ) / nele - 1.0;
    dfdx2_phys = dfdx_stress_explicit + dfdx_stress_implicit;
    
    %=================================================================
    % --- FILTERING (All sensitivities) ---
    %=================================================================
    if ft == 1
        % Filter compliance sensitivity
        dc(:) = H*(dc(:)./Hs);
        % Filter volume sensitivity
        dfdx1_filtered = H*(dfdx1_phys'./Hs);
        dfdx(1,:) = dfdx1_filtered' / (volfrac*nele); % Re-apply scaling
        % Filter stress sensitivity
        dfdx2_filtered = H*(dfdx2_phys(:)./Hs);
        dfdx(2,:) = dfdx2_filtered';
    elseif ft == 0
        dfdx(1,:) = dfdx1_phys;
        dfdx(2,:) = dfdx2_phys';
    end
    % (ft==2 from base code is removed as it was unstable)
    
    %=================================================================
    % --- MMA OPTIMIZER CALL ---
    %=================================================================
    xval  = x(:);
    f0val = c;
    df0dx = dc(:);
    fval = fval'; % m x 1
    dfdx = dfdx;  % m x n
    
    [xmma, ~, ~, ~, ~, ~, ~, ~, ~, low, upp] = ...
            mmasub(m, n, loop, xval, xmin, xmax, xold1, xold2, ...
                   f0val, df0dx, 0*df0dx, fval, dfdx, 0*dfdx, ...
                   low, upp, a0, a, c_MMA, d);
    
    % Update MMA Variables
    xnew = reshape(xmma, nely, nelx);
    if ft == 1
        xPhys(:) = (H*xnew(:))./Hs;
    elseif ft == 0
        xPhys = xnew;
    end
    xold2 = xold1(:);
    xold1 = x(:);
    
    change = max(abs(xnew(:)-x(:)));
    x = xnew;

    % PRINT RESULTS
    fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f Stress:%7.3f ch.:%7.3f\n', ...
        loop, c, mean(xPhys(:)), fval(2)+1.0, change);

    %=================================================================
    % --- PLOTTING (2D) ---
    %=================================================================
    colormap(gray); 
    imagesc(-xPhys); 
    axis equal; axis tight; axis off;
    pause(1e-6);
end
    

