function[dfdx_0] = derivative_stress(derivative, derivative0, nc, n, nelx, nely, penal, rmin, x)
    dfdx_0 = zeros(nc, n);
    for i = 1:nc
        dfdx_0(i, :) = derivative0(i, :).*derivative';
        dfdx_1 = reshape(dfdx_0(i, :), [nelx, nely])';
        [dfdx_1] = check(nelx, nely, rmin, x, dfdx_1);
        count = 1;
        for g = 1:nely
            for h = 1:nelx
                dfdx_2(count, 1) = dfdx_1(g, h);
                count = count + 1;
            end
        end
    dfdx_0(i, :) = dfdx_2;
    end
end