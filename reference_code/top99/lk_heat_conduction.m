%%%%%%%%%% Element Stiffness Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [KE] = lk_heat_conduction
    KE = [2/3 -1/6 -1/3 -1/6
         -1/6  2/3 -1/6 -1/3
         -1/3 -1/6  2/3 -1/6
         -1/6 -1/3 -1/6  2/3];
end