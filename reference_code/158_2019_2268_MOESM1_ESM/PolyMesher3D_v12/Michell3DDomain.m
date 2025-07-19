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
function [x] = Michell3DDomain(Demand,Arg)
  BdBox = [0 5 -2 2 -1 1];
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
  d1 = dRectangle(P,BdBox(1),BdBox(2),BdBox(3),BdBox(4));
  d2 = dCircle(P,0,0,BdBox(4)/2);
  Dist = dDiff(d1,d2);
  d3 = [BdBox(5)-P(:,3), P(:,3)-BdBox(6)];
  d3 = [d3,max(d3,[],2)];
  Dist = dIntersect(Dist,d3);
  %rmpath('./PolyMesher_v11');
%---------------------------------------------- SPECIFY BOUNDARY CONDITIONS
function [x] = BndryCnds(Node,Element,BdBox)
  eps =100*((BdBox(2)-BdBox(1))*...
             (BdBox(4)-BdBox(3))*...
             (BdBox(6)-BdBox(5)))^(1/3)/size(Node,1);
  CircleNodes=[];
  CircleNodes = [CircleNodes; find(max(abs(sqrt(Node(:,1).^2+Node(:,2).^2)-1.0),abs(Node(:,3)-1))<eps )];
  CircleNodes = [CircleNodes; find(max(abs(sqrt(Node(:,1).^2+Node(:,2).^2)-1.0),abs(Node(:,3)+1))<eps )]; 
  Supp = ones(size(CircleNodes,1),4);
  Supp(:,1) = CircleNodes;
  LoadNode= find(abs(Node(:,1)-5)<eps & abs(Node(:,2)-0)<eps & abs(Node(:,3)-0)<eps );
  Load = [LoadNode,0,-1,0];
  x = {Supp,Load};
%----------------------------------------------------- SPECIFY FIXED POINTS
function [PFix] = FixedPoints(BdBox)
  PFix = [5 0 0];
%--------------------------------------------- SPECIFY ANALYTICAL GRADIENTS
function [x] = Normal(P,n1,n2,n3)
  dc1 = n1(:,5); dc2 = n2(:,5);
  n1 = repmat([-1 1  0 0 0  0 0 0],size(n1,1),1);
  n2 = repmat([ 0 0 -1 1 0  0 0 0],size(n2,1),1);
  n3 = repmat([ 0 0  0 0 0 -1 1 0],size(n3,1),1);
  n1(:,5) = dc1; n2(:,5) = dc2;
  x = {n1,n2,n3};
%-------------------------------------------------------------------------%