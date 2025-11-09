function lambda = FEA_adjoint(K, rhs, fixeddofs)

    [ndof, nc] = size(rhs);
    lambda = zeros(ndof, nc);
    
    alldofs = 1:ndof;
    freedofs = setdiff(alldofs, fixeddofs);
    
    % 对每个聚类求解
    for i = 1:nc
        lambda(freedofs, i) = K(freedofs, freedofs) \ rhs(freedofs, i);
    end
end