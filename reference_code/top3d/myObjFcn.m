function [f, gradf] = myObjFcn(x, H, Hs, KE, penal, E0, Emin, nelx, nely, nelz, edofMat, freedofs, iK, jK, F)
    global ce;  % 使用全局变量 ce

    % 计算物理密度
    xPhys = (H * x(:)) ./ Hs;
    xPhys = reshape(xPhys, [nely, nelx, nelz]);
    
    % 材料刚度插值
    E = Emin + xPhys(:)'.^penal * (E0 - Emin);
    
    % 构建全局刚度矩阵
    sK = reshape(KE(:) * E, 24*24*nelx*nely*nelz, 1);
    K = sparse(iK, jK, sK);
    K = (K + K') / 2;  % 确保对称性
    
    % 解线性方程组
    U = zeros(size(F));
    U(freedofs) = K(freedofs, freedofs) \ F(freedofs);
    
    % 计算目标函数（柔度）
    ce = reshape(sum((U(edofMat) * KE) .* U(edofMat), 2), [nely, nelx, nelz]);
    c = sum(sum(sum((Emin+xPhys.^penal*(E0-Emin)).*ce)));
    
    % 计算梯度
    dc = -penal * (E0 - Emin) * xPhys(:).^(penal - 1) .* ce(:);
    
    % 应用密度滤波
    dc = (H * (dc ./ Hs))';
    
    % 返回目标函数值和梯度
    f = c;
    gradf = dc(:);
end


 