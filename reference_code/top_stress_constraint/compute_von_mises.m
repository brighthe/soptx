function [stress_vm] = compute_von_mises(nelx, nely, x, U, penal_S, B, D)

    n = nelx * nely;
    stress_vm = zeros(n, 1);
    
    for elx = 1:nelx
        for ely = 1:nely
            elem_idx = (elx-1)*nely + ely; 

            n1 = (nely+1)*(elx-1)+ely; 
            n2 = (nely+1)* elx   +ely;
            Ue = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2], 1);

            % 单元应变: ε = B * Ue
            strain_e = B * Ue;
            
            % 实体材料单元应力: σ̂  = D * ε
            stress_solid = D * strain_e;

            % 应力惩罚 σ = ρ^penal_S * σ̂
            stress_penalized = (x(ely, elx)^penal_S) * stress_solid;

            % von Mises 应力 (平面应力)
            % σ_vm = sqrt(σ_x² + σ_y² - σ_x*σ_y + 3*τ_xy²)
            sigma_x = stress_penalized(1);
            sigma_y = stress_penalized(2);
            tau_xy = stress_penalized(3);
            
            stress_vm(elem_idx) = sqrt(sigma_x^2 + sigma_y^2 - ...
                                       sigma_x*sigma_y + 3*tau_xy^2);

        end
    end

end