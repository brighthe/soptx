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
function [x] = Hook3DDomain(Demand,Arg)
  BdBox = [-35.0812, 64.8842, -48.1395, 100.6226  -6 6];
  switch(Demand)
    case('Dist');   x = DistFnc(Arg,BdBox);
    case('BC');     x = BndryCnds(Arg{:},BdBox);
    case('BdBox');  x = BdBox;
    case('PFix');   x = FixedPoints(BdBox);
    case('Normal'); x = Normal(Arg{:});
  end
%----------------------------------------------- COMPUTE DISTANCE FUNCTIONS
function Dist = DistFnc(P,BdBox)
  %addpath('./PolyMesher_v11'); % to use PolyMesher 2D functions
  c1 = dCircle(P,59.9713,78.7683,80);
  c2 = dCircle(P,54.8716,76.8672,35);
  c3 = dCircle(P,0,80.6226,20);
  c4 = dCircle(P,0,80.6226,10);
  c5 = dCircle(P,14.8842,1.8605,50);
  c6 = dCircle(P,0,0,19);
  c7 = dCircle(P,-27.0406,0,8.0406);
  l1 = dLine(P,65.4346,76.9983,-19.9904,81.2407);
  l2 = dLine(P,-25.6060,-27.4746,65.4346,76.9983);
  l3 = dLine(P,1,0,0,0);
  d1 = dDiff(dUnion(dIntersect(dDiff(c1,c2),dIntersect(l1,l2)),c3),c4);
  d2 = dUnion(dIntersect(dDiff(c5,c6),l3),c7);
  d3 = dIntersect(dDiff(c5,c6),-l2);
  Dist = dUnion(dUnion(d1,d2),d3);
  % Extrusion
  d4 = [BdBox(5)-P(:,3), P(:,3)-BdBox(6)];
  d4 = [d4,max(d4,[],2)];
  Dist = dIntersect(d4,Dist);
  %rmpath('./PolyMesher_v11');
  %---------------------------------------------- SPECIFY BOUNDARY CONDITIONS
function [x] = BndryCnds(Node,Element,BdBox)
  eps =1e1*((BdBox(2)-BdBox(1))*...
             (BdBox(4)-BdBox(3))*...
             (BdBox(6)-BdBox(5)))^(1/3)/size(Element,1)/2;
  UpperHalfCircleNodes = find(abs(sqrt(Node(:,1).^2+(Node(:,2)...
    -80.6226).^2)-10)<0.04 & Node(:,2)>+80.6226 & abs(Node(:,3))<eps);
  Supp = ones(size(UpperHalfCircleNodes,1),4);
  Supp(:,1) = UpperHalfCircleNodes;
  LowerHalfCircleNodes = ...
      find(abs(sqrt(Node(:,1).^2+Node(:,2).^2)-19)<0.025 & ...
      Node(:,2)<0 & abs(Node(:,3))<eps);
  Load = -1*ones(size(LowerHalfCircleNodes,1),4);
  Load(:,1) = LowerHalfCircleNodes; Load(:,2) = 0; Load(:,4) = 0;
  x = {Supp,Load};
%----------------------------------------------------- SPECIFY FIXED POINTS
function [PFix] = FixedPoints(BdBox)
  PFix = [];
%--------------------------------------------- SPECIFY ANALYTICAL GRADIENTS
function [x] = Normal(P,n1,n2,n3)
  n1(:,1:2) = 0;
  n2(:,1:2) = 0;
  n3(:,1) = -1; n3(:,2) =  1;
  n3(:,3:end) =  0;
  x = {n1,n2,n3};
%-------------------------------------------------------------------------%
