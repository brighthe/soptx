%% Element stiffness matrix
function [K] = stiffnessMatrix(k)
    % Forms stiffness matrix from first now
    K = [k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
         k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
         k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
         k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
         k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
         k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
         k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
         k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)];
end
