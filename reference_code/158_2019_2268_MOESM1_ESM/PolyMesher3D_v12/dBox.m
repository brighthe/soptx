%-------------------------------------------------------------------------%
% PolyMesher3D developed by Anderson Pereira, April 2019                  %
% Contact: anderson@puc-rio.br                                            %
%-------------------------------------------------------------------------%
% Ref1: C Talischi, GH Paulino, A Pereira, IFM Menezes,                   %
%      "PolyMesher: A general-purpose mesh generator for polygonal        %
%      elements written in Matlab", Struct Multidisc Optim, 2012,         %
%      DOI 10.1007/s00158-011-0706-z                                      %
%                                                                         %
% Ref2: RS Thedin, A Pereira, IFM Menezes, GH Paulino,                    %
%      "Polyhedral mesh generation and optimization for finite element    %
%      computations. In: CILAMCE 2014 - XXXV Ibero-Latin American         %
%      Congress on Computational Methods in Engineering, 2014             %
%                                                                         %
% Ref3: A Pereira, C Talischi, GH Paulino, IFM Menezes, MS Carvalho,      %
%      "Implementation of fluid flow topology optimization in PolyTop",   %
%      Struct Multidisc Optim, 2016, DOI 10.1007/s00158-014-1182-z        %
%                                                                         %
% Ref4: H Chi, A Pereira, IFM Menezes, GH Paulino,                        %
%      "Virtual Element Method (VEM)-based topology optimization:         %
%      an integrated framework", Struct Multidisc Optim, 2019,            %
%      DOI 10.1007/s00158-019-02268-w                                     %
%-------------------------------------------------------------------------%
function d = dBox(P,x1,x2,y1,y2,z1,z2)
d = [x1-P(:,1), P(:,1)-x2, y1-P(:,2), P(:,2)-y2, z1-P(:,3), P(:,3)-z2];
d = [d,max(d,[],2)];
%-------------------------------------------------------------------------%


