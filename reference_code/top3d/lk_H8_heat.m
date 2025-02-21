function [KE] = lk_H8_heat(k)
    A1  = 4*eye(2);   A2 = -eye(2);
    A3  = fliplr(A2); A4 = -ones(2);
    KE1 = [A1 A2; A2 A1];
    KE2 = [A3 A4; A4 A3];
    KE  = 1/12 * k * [KE1 KE2; KE2 KE1];
end
