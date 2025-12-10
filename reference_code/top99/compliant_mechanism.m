nelx = 40;
nely = 20;
volfrac = 0.3;
penal = 3.0;
rmin = 1.2;
% 初始化
x(1:nely, 1:nelx) = volfrac;
loop = 0;
change = 1.;
% Start Iteration
while change > 0.01
    loop = loop + 1;
    xold = x;
    % FE-Analysis
    [U, c] = FE_compliant_mechanism(nelx, nely, x, penal);
    % OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    [KE] = lk;
    for ely = 1:nely
        for elx = 1:nelx 
            n1 = (nely+1)*(elx-1)+ely;
            n2 = (nely+1)* elx   +ely;
            Ue1 = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2], 1);
            Ue2 = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2], 2);
            dc(ely, elx) = penal*x(ely, elx)^(penal-1)*Ue1'*KE*Ue2;
        end
    end
    % Filtering Of Sensitivity
    dc_sum = sum(dc(:));
    fprintf('dc_sum:%5f\n', dc_sum);
    dc_mean = mean(dc(:));
    fprintf('dc_mean:%5f\n', dc_mean);

    [dc] = check(nelx, nely, rmin, x, dc);
    
    dc_sum = sum(dc(:));
    fprintf('dc_sum:%5f\n', dc_sum);
    dc_mean = mean(dc(:));
    fprintf('dc_mean:%5f\n', dc_mean);

    % Design Update By The Optimality Criteria Method
    x_sum = sum(x(:));
    fprintf('x_sum:%5f\n', x_sum);
    x_mean = mean(x(:));
    fprintf('x_mean:%5f\n', x_mean);
    [x] = OC_compliant_mechanism(nelx, nely, x, volfrac, dc);
    x_sum = sum(x(:));
    fprintf('x_sum:%5f\n', x_sum);
    x_mean = mean(x(:));
    fprintf('x_mean:%5f\n', x_mean);

    % Print Results
    change = max(max(abs(x-xold)));
    disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.5f',c) ...
        ' Vol.: ' sprintf('%6.4f',sum(sum(x))/(nelx*nely)) ...
        ' ch.: ' sprintf('%6.4f',change )])
    
    % Plot Densities
    colormap(gray); imagesc(-x); axis equal; axis tight; axis off;pause(1e-6);

end