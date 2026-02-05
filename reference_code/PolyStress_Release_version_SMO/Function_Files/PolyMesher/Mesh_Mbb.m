%----------------------------- PolyStress --------------------------------%
% Ref: O Giraldo-Londoño, GH Paulino, "PolyStress: A Matlab implementation%
% for topology optimization with local stress constraints using the       %
% augmented Lagrangian method", Structural and Multidisciplinary          %
% Optimization, DOI 10.1007/s00158-020-02664-7, 2020                      %
%-------------------------------------------------------------------------%
function [Node,Element,Supp,Load] = Mesh_Mbb(Ne_ap)
L = 3; H = 1;  % MBB beam dimensions: Length × Height
nn = 2*floor(round(sqrt(Ne_ap/(L/H)))/2); he = H/nn; 
NElem = round((L/he)*(H/he)); 
[X,Y] = meshgrid(he/2:he:L-he/2,he/2:he:H-he/2);
P = [X(:) Y(:)]; % Mesh seed
[Node,Element,Supp,Load,~] = PolyMesher(@MbbDomain,NElem,0,P);
%-------------------------------------------------------------------------%