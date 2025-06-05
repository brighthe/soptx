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
function [x] = Wrench3DDomain(Demand,Arg)
  BdBox = [-0.3 2.5 -0.5 0.5 -0.1 0.1];
  switch(Demand)
    case('Dist');   x = DistFnc(Arg,BdBox);
    case('BC');     x = BndryCnds(Arg{:},BdBox);
    case('BdBox');  x = BdBox;
    case('PFix');   x = [];
    case('Normal'); x = Normal(Arg{:});
  end
%----------------------------------------------- COMPUTE DISTANCE FUNCTIONS
function Dist = DistFnc(P,BdBox)
  %addpath('./PolyMesher_v11'); % to use PolyMesher 2D functions
  d1 = dLine(P,0,0.3,0,-0.3);
  d2 = dLine(P,0,-0.3,2,-0.5);
  d3 = dLine(P,2,-0.5,2,0.5);
  d4 = dLine(P,2,0.5,0,0.3);
  d5 = dCircle(P,0,0,0.3);
  d6 = dCircle(P,2,0,0.5);
  douter = dUnion(d6,dUnion(d5,...
           dIntersect(d4,dIntersect(d3,dIntersect(d2,d1)))));
  d7 = dCircle(P,0,0,0.175);
  d8 = dCircle(P,2,0,0.3);
  din = dUnion(d8,d7);
  Dist = dDiff(douter,din);
  % Extrusion
  d4 = [BdBox(5)-P(:,3), P(:,3)-BdBox(6)];
  d4 = [d4,max(d4,[],2)];
  Dist = dIntersect(d4,Dist);
  %rmpath('./PolyMesher_v11');
%---------------------------------------------- SPECIFY BOUNDARY CONDITIONS
function [x] = BndryCnds(Node,Element,BdBox)
  eps =  1*sqrt((BdBox(2)-BdBox(1))*(BdBox(4)-BdBox(3))/size(Node,1));
  RightCircleNodes = ...
      find(abs(sqrt((Node(:,1)-2).^2+ Node(:,2).^2)-0.3)<eps);   
  Supp = ones(size(RightCircleNodes,1),4);
  Supp(:,1) = RightCircleNodes; Supp(:,4) = 0;
  LeftHalfCircleNodes = ...
      find(abs(max(sqrt(Node(:,1).^2+Node(:,2).^2)-0.175,Node(:,2)))<eps);
  Load = -0.1*ones(size(LeftHalfCircleNodes,1),4);
  Load(:,1) = LeftHalfCircleNodes; Load(:,2) = 0; Load(:,4) = 0;
  x = {Supp,Load};
%--------------------------------------------- SPECIFY ANALYTICAL GRADIENTS
function [x] = Normal(P,n1,n2,n3)
  n1(:,1:2) = 0;
  n2(:,1:2) = 0;
  n3(:,1) = -1; n3(:,2) =  1;
  n3(:,3:end) =  0;
  x = {n1,n2,n3};
%-------------------------------------------------------------------------%