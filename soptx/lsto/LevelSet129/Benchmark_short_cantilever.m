nely = 20;
nelx = 32;
volReq = 0.4;
stepLength = 2;
topWeight = 2;
numReinit = 3;

% Initialization
struc = ones(nely, nelx);
[lsf] = reinit(struc);
shapeSens = zeros(nely, nelx); topSens = zeros(nely, nelx);
[KE, KTr, lambda, mu] = materialInfo();

% 打开一个文件用于写入
fileID = fopen('results.txt', 'w');
% 写入标题
fprintf(fileID, 'It\tCompl\tVol\n');

% 创建一个视频写入对象
v = VideoWriter('topology_optimization.avi');
v.FrameRate = 10; % 设置帧率
open(v);

% Main loop
num = 200;
for iterNum = 1:num
    % FE-analysis, calculate sensitivities
    [U] = FE(struc, KE);
    for ely = 1:nely
        for elx = 1:nelx
                n1 = (nely+1)*(elx-1)+ely;
                n2 = (nely+1) * elx  +ely;
                Ue = U([2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2], 1);
                shapeSens(ely, elx) = -max(struc(ely, elx), 0.0001)*Ue'*KE*Ue;
                coef = struc(ely, elx) * pi/2*(lambda+2*mu)/mu/(lambda+mu);
                topSens(ely, elx) = struc(ely, elx) * pi/2*(lambda+2*mu)/mu/(lambda+mu) ...,
                *(4*mu*Ue'*KE*Ue+(lambda-mu)*Ue'*KTr*Ue);
        end
    end
    % Store data, print & plot information
    objective(iterNum) = -sum(shapeSens(:));
    volCurr = sum(struc(:))/(nelx*nely);
    disp([' It.: ' num2str(iterNum) 'Compl.: ' sprintf('%10.4f', objective(iterNum)) ...,
                                       'Vol.:' sprintf('%6.3f', volCurr)]);
    % 保存结果到文件
    % 保存结果到文件
    fprintf(fileID, '%4i\t%10.4f\t%6.3f\n', iterNum, objective(iterNum), volCurr);


    colormap("gray");imagesc(-struc, [-1,0]); axis equal; axis tight; axis off; drawnow;
    % 捕捉当前帧并写入视频
    frame = getframe(gcf);
    writeVideo(v, frame);

    % Check for convergence
    if iterNum > 5 && (abs(volCurr-volReq)<0.005) && ...,
            all( abs(objective(end)-objective(end-5:end-1)) < 0.01*abs(objective(end)) );
        return;
    end
    % Set augmented Lagrangian parameters
    if iterNum == 1
        la = -0.01; La = 1000; alpha = 0.9;
    else
        la = la - 1/La * (volCurr - volReq); La = alpha * La;
    end
    % Include volume sensitivites
    shapeSens = shapeSens - la + 1/La*(volCurr-volReq);
    topSens = topSens + pi*(la - 1/La*(volCurr-volReq));

    % Design update
    [struc, lsf] = updateStep(lsf, shapeSens, topSens, stepLength, topWeight);

    if ~mod(iterNum, numReinit)
        [lsf] = reinit(struc);
    end
end

% 关闭文件
fclose(fileID);

% 关闭视频写入对象
close(v);