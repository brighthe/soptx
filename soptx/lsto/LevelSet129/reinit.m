%% Reinitialization of level-set function
function [lsf] = reinit(struc)
    strucFull = zeros(size(struc)+2); strucFull(2:end-1, 2:end-1) = struc;
    % Use "bwdist"
    lsf = (~strucFull).*(bwdist(strucFull)-0.5) - strucFull.*(bwdist(strucFull-1)-0.5);
end