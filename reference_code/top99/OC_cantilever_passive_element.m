function [xnew] = OC_cantilever_passive_element(nelx, nely, x, volfrac, dc, passive)
    l1 = 0; l2 = 100000; move = 0.2;
    while (l2-l1 > 1e-4)
        lmid = 0.5*(l2+l1);
        xnew = max(0.001, max(x-move, min(1., min(x+move, x.*sqrt(-dc./lmid)))));
        xnew(find(passive)) = 0.001;
        if sum(sum(xnew)) - volfrac*nelx*nely > 0
            l1 = lmid;
        else
            l2 = lmid;
        end
    end
end