nelx = 30;
nely = 30;
% nelx = 3;
% nely = 2;
volfrac = 0.4;
penal = 3.0;
rmin = 1.2;

x(1:nely, 1:nelx) = volfrac;
loop = 0;
change = 1.;

% 打开一个文件用于写入
fileID = fopen('results.txt', 'w');
% 写入标题
fprintf(fileID, 'Iteration\tObjective\tVolume\tChange\n');

% 创建一个视频写入对象
v = VideoWriter('topology_optimization.avi');
v.FrameRate = 10; % 设置帧率
open(v);

% Start Iteration
while change > 0.01
    loop = loop + 1;
    xold = x;
    % FE-Analysis
    [U] = FE_multiple_load_cantilever(nelx, nely, x, penal);

    % OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    [KE] = lk;
    c = 0.;
    for ely = 1:nely
        for elx = 1:nelx
            n1 = (nely+1)*(elx-1)+ely;
            n2 = (nely+1)* elx   +ely;
            dc(ely, elx) = 0.;
            for i = 1:2
              Ue = U([2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2], i); % 单元位移矢量
              c = c + x(ely, elx)^penal*Ue'*KE*Ue; % 目标函数
              dc(ely, elx) = dc(ely, elx) - penal*x(ely, elx)^(penal-1)*Ue'*KE*Ue;
            end
        end
    end
    F = sparse(2*(nely+1)*(nelx+1), 1);
    F(2*(nelx+1)*(nely+1), 1) = -1;
    % c1 = F' * U;
    % Filtering Of Sensitivity
    [dc] = check(nelx, nely, rmin, x, dc);
    % Design Update By The Optimality Criteria Method
    [x] = OC(nelx, nely, x, volfrac, dc);
    % Print Results
    change = max(max(abs(x-xold)));
    disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.4f',c) ...
        ' Vol.: ' sprintf('%6.3f',sum(sum(x))/(nelx*nely)) ...
        ' ch.: ' sprintf('%6.3f',change )])

    % 保存结果到文件
    fprintf(fileID, '%4i\t%10.4f\t%6.3f\t%6.3f\n', loop, c, sum(sum(x))/(nelx*nely), change);
    
    % Plot Densities
    colormap(gray); imagesc(-x); axis equal; axis tight; axis off;pause(1e-6);

    % 捕捉当前帧并写入视频
    frame = getframe(gcf);
    writeVideo(v, frame);
end

% 关闭文件
fclose(fileID);
% 关闭视频写入对象
close(v);