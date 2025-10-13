function [xnew] = OC_compliant_mechanism(nelx, nely, x, volfrac, dc)
    l1 = 0; l2 = 100000; move = 0.1;
    while (l2-l1) / (l2+l1) > 1e-4 & l2 > 1e-40
        lmid = 0.5*(l2+l1);
        xnew = max(0.001, max(x-move, min(1., min(x+move, x.*(max(1e-10, -dc./lmid)).^0.3))));
        if sum(sum(xnew)) - volfrac*nelx*nely > 0
            l1 = lmid;
        else
            l2 = lmid;
        end
    end
end