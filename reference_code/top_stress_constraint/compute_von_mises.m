function [stress_vm, stress_penalized] = compute_von_mises(nelx, nely, xPhys, U, penal_S, B, D)

    n = nelx * nely;
    stress_vm = zeros(n, 1);
    stress_penalized = zeros(n, 3);
    
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
            stress_elem  = (xPhys(ely, elx)^penal_S) * stress_solid;
            stress_penalized(elem_idx, :) = stress_elem';

            % von Mises 应力 (平面应力)
            % σ_vm = sqrt(σ_x² + σ_y² - σ_x*σ_y + 3*τ_xy²)
            sigma_x = stress_elem(1);
            sigma_y = stress_elem(2);
            tau_xy = stress_elem(3);
            
            stress_vm(elem_idx) = sqrt(sigma_x^2 + sigma_y^2 - ...
                                       sigma_x*sigma_y + 3*tau_xy^2);

        end
    end

end