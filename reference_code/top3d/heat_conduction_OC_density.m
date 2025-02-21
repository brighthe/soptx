nelx = 40;
nely = 40;
nelz = 5;
volfrac = 0.3;
penal = 3;
rmin = 1.4;

% USER-DEFINED LOOP PARAMETERS
maxloop = 200;    % Maximum number of iterations
tolx = 0.01;      % Terminarion criterion
displayflag = 0;  % Display structure flag

% USER-DEFINED MATERIAL PROPERTIES
k0   = 1;    % Good thermal conductivity
kmin = 1e-3; % Poor thermal conductivity

% USER-DEFINED SUPPORT FIXED DOFs
il = nelx/2-nelx/20:nelx/2+nelx/20; jl = nely; kl = 0:nelz;
fixedxy = il*(nely+1)+(nely+1-jl);
fixednid = repmat(fixedxy', size(kl)) + repmat(kl*(nelx+1)*(nely+1), size(fixedxy,2),1);
fixeddof = reshape(fixednid, [], 1);

% PREPARE FINITE ELEMENT ANALYSIS
nele = nelx*nely*nelz;
ndof = (nelx+1)*(nely+1)*(nelz+1);
F = sparse(1:ndof, 1, -0.01, ndof, 1);
U = zeros(ndof, 1);
freedofs = setdiff(1:ndof, fixeddof);
KE = lk_H8_heat(k0);
nodegrd = reshape(1:(nely+1)*(nelx+1),nely+1,nelx+1);
nodeids = reshape(nodegrd(1:end-1,1:end-1),nely*nelx,1);
nodeidz = 0:(nely+1)*(nelx+1):(nelz-1)*(nely+1)*(nelx+1);
nodeids = repmat(nodeids, size(nodeidz)) + repmat(nodeidz, size(nodeids));
edofVec = nodeids(:)+1;
edofMat = repmat(edofVec, 1, 8)+ ...
        repmat([0 nely + [1 0] -1 ...
        (nely+1)*(nelx+1)+[0 nely + [1 0] -1]], nele, 1);
iK = reshape(kron(edofMat,ones(8,1))', 8*8*nele, 1);
jK = reshape(kron(edofMat,ones(1,8))', 8*8*nele, 1);

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
Hs = sum(H,2);

% INITIALIZE ITERATION
x = repmat(volfrac, [nely,nelx,nelz]);
xPhys = x; 
loop = 0; 
change = 1;

fileID = fopen('heat_conduction_OC_density.txt', 'w');
fprintf(fileID, 'Iteration\tObjective\tVolume\tChange\n');
figure('Position', [100, 100, 1130, 784]);
v = VideoWriter('heat_conduction_OC_desnity.avi');
v.FrameRate = 10;
open(v);
firstFrame = true;
frameSize = [0, 0];

% START ITERATION
while change > tolx && loop < maxloop
    loop = loop + 1;

    % FE-ANALYSIS
    sK = reshape(KE(:) * (kmin + (1-kmin) * xPhys(:)' .^ penal), 8*8*nele, 1);
    K = sparse(iK, jK, sK); K = (K+K')/2;
    U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);

    % OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    ce = reshape(sum((U(edofMat)*KE).*U(edofMat),2),[nely,nelx,nelz]);
    c = sum(sum(sum((kmin + (1-kmin) * xPhys .^ penal) .* ce)));
    dc = -penal * (1-kmin) * xPhys .^ (penal-1) .* ce;
    dv = ones(nely, nelx, nelz);

    % FILTERING AND MODIFICATION OF SENSITIVITIES
    dc(:) = H*(dc(:)./Hs);  
    dv(:) = H*(dv(:)./Hs);
    % OPTIMALITY CRITERIA UPDATE
    l1 = 0; l2 = 1e9; move = 0.2;
    while (l2-l1)/(l1+l2) > 1e-3
        lmid = 0.5*(l2+l1);
        xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
        xPhys(:) = (H*xnew(:))./Hs;
        if sum(xPhys(:)) > volfrac*nele, l1 = lmid; else l2 = lmid; end
    end
    change = max(abs(xnew(:)-x(:)));
    x = xnew;

    % PRINT RESULTS
    fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n', loop, c, mean(xPhys(:)), change);
    fprintf(fileID, '%4i\t%10.4f\t%6.3f\t%6.3f\n', loop, c, mean(xPhys(:)), change);

    if mod(loop, 10) == 0 || loop == 1 || change <= tolx || loop == maxloop
        clf;
        display_3D(xPhys);
        title(sprintf('Iteration: %d, Objective: %.4f', loop, c));
        drawnow;
        
        % 获取当前帧
        frame = getframe(gcf);
        
        % 检查和调整帧大小
        if firstFrame
            frameSize = size(frame.cdata);
            firstFrame = false;
        else
            frame.cdata = imresize(frame.cdata, [frameSize(1), frameSize(2)]);
        end
        
        % 将当前帧添加到视频
        writeVideo(v, frame);
    end
end

% 关闭视频对象
close(v);

% 显示最终结果
clf;
display_3D(xPhys);
title('Final Optimized Design');