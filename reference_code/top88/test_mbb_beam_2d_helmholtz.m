nelx = 60;
nely = 20;
rmin = 2.4;

% nelx = 150;
% nely = 50;
% rmin = 6;

% nelx = 300;
% nely = 100;
% rmin = 12;

volfrac = 0.5;
penal = 3;

% ft = 1;   % 灵敏度滤波器
ft = 2;   % 密度滤波器

% MATERIAL PROPERTIES
E0 = 1;
Emin = 1e-9;
nu = 0.3;

% PREPARE FINITE ELEMENT ANALYSIS
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);

% DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
F = sparse(2, 1, -1, 2*(nely+1)*(nelx+1), 1);
U = zeros(2*(nely+1)*(nelx+1), 1);
fixeddofs = union([1:2:2*(nely+1)], [2*(nelx+1)*(nely+1)]);
alldofs = [1:2*(nely+1)*(nelx+1)];
freedofs = setdiff(alldofs, fixeddofs);

% PREPARE FILTER
Rmin = rmin / 2 / sqrt(3);
KEF = Rmin^2*[4 -1 -2 -1; -1  4 -1 -2; -2 -1  4 -1; -1 -2 -1  4]/6 + ...
             [4  2  1  2;  2  4  2  1;  1  2  4  2;  2  1  2  4]/36;
edofVecF = reshape(nodenrs(1:end-1,1:end-1), nelx*nely,1);
edofMatF = repmat(edofVecF,1,4)+repmat([0 nely+[1:2] 1], nelx*nely,1);
iKF = reshape(kron(edofMatF,ones(4,1))', 16*nelx*nely,1);
jKF = reshape(kron(edofMatF,ones(1,4))', 16*nelx*nely,1);
sKF = reshape(KEF(:)*ones(1,nelx*nely), 16*nelx*nely,1);
KF = sparse(iKF, jKF, sKF);
LF = chol(KF, 'lower');
iTF = reshape(edofMatF, 4*nelx*nely, 1);
jTF = reshape(repmat([1:nelx*nely], 4, 1)', 4*nelx*nely, 1);
sTF = repmat(1/4, 4*nelx*nely,1);
TF = sparse(iTF, jTF, sTF);

%%-------------------- INITIALIZE ITERATION --------------------%%
x = repmat(volfrac, nely, nelx);
xPhys = x;

loop = 0;
change = 1;

% START ITERATION
while change > 0.01
	tic;  % Start timing the iteration
	loop = loop + 1;

	% FE-ANALYSIS
	sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)), 64*nelx*nely,1);
	K = sparse(iK, jK, sK); K = (K + K') / 2;
	U(freedofs) = K(freedofs,freedofs) \ F(freedofs);

	% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
	Ue = U(edofMat);
	ce = reshape(sum((U(edofMat)*KE).*U(edofMat),2), nely, nelx);
	c = sum(sum((Emin + xPhys.^penal * (E0 - Emin)).*ce));
	dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;
	dv = ones(nely, nelx);
	
	% FILTERING/MODIFICATION OF SENSITIVITIES
	if ft == 1
        dc(:) = (TF' * (LF' \ (LF \ (TF * (dc(:) .* xPhys(:)))))) ...
                    ./ max(1e-3, xPhys(:));
	elseif ft == 2
        dc(:) = TF' * (LF' \ (LF \ (TF * dc(:))));
        dv(:) = TF' * (LF' \ (LF \ (TF * dv(:))));
	end
	
	% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
	l1 = 0; l2 = 1e9; move = 0.2;
    % fprintf(' x.:%16.12f\n', mean(x(:)));
	while (l2-l1)/(l1+l2) > 1e-3
		lmid = 0.5 * (l2 + l1);
		xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
		if ft == 1
			xPhys = xnew;
		elseif ft == 2
			xPhys(:) = (TF' * (LF' \ (LF \ (TF * xnew(:)))));
		end
		if sum(xPhys(:)) > volfrac*nelx*nely, l1 = lmid; else l2 = lmid; end
	end
	change = max(abs(xnew(:)-x(:)));
	x = xnew;

	% PRINT RESULTS
	iter_time = toc;  % Stop timing and get iteration time
	fprintf(' It.:%5i Obj.:%11.4f Vol.:%16.12f ch.:%7.3f Time:%7.3f sec\n', loop, c, mean(xPhys(:)), change, iter_time);

	% PLOT DENSITIES
	colormap(gray); imagesc(1-xPhys); caxis([0 1]); axis equal; axis off; drawnow;
end
