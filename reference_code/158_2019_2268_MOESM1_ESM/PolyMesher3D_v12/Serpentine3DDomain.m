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
function [x] = Serpentine3DDomain(Demand,Arg)
  r=4; l=3; a=4;
  b=a*l/r; c=a/r*sqrt(r^2-l^2); d=-sqrt(-l^2+r^2);
  BdBox = [0,3*l+2*b,-r-a-d,r+a+d,0,a];
  switch(Demand)
    case('Dist');  x = DistFnc(Arg,BdBox,r,l,a,b,c,d);
    case('BC');    x = BndryCnds(Arg{:},BdBox,r,l,a,b,c,d);
    case('BdBox'); x = BdBox;
    case('PFix');  x = FixedPoints(BdBox);
    case('Normal'); x = Normal(Arg{:});
  end
%----------------------------------------------- COMPUTE DISTANCE FUNCTIONS
function Dist = DistFnc(P,BdBox,r,l,a,b,c,d)
  %addpath('./PolyMesher_v11'); % to use PolyMesher 2D functions
  d1 = dCircle(P,0,d,r+a);
  d2 = dCircle(P,0,d,r);
  d3 = dLine(P,0,d,l,0);
  d4 = dLine(P,0,1,0,d);
  Dist1 = dIntersect(dIntersect(dDiff(d1,d2),d3),d4);
  d5 = dCircle(P,2*l+b,c-d,r+a);
  d6 = dCircle(P,2*l+b,c-d,r);
  d7 = dLine(P,2*l+b,c-d,l+b,c);
  d8 = dLine(P,3*l+b,c,2*l+b,c-d);
  Dist2 = dIntersect(dIntersect(dDiff(d5,d6),d7),d8);
  Dist = dUnion(Dist1,Dist2);
  % Extrusion
  d4 = [BdBox(5)-P(:,3), P(:,3)-BdBox(6)];
  d4 = [d4,max(d4,[],2)];
  Dist = dIntersect(d4,Dist);
  %rmpath('./PolyMesher_v11');
%----------------------------------------------------- SPECIFY FIXED POINTS
function [PFix] = FixedPoints(BdBox)
  PFix = [BdBox(2)*ones(1,1),zeros(1,1),BdBox(6)/2];
  %---------------------------------------------- SPECIFY BOUNDARY CONDITIONS
function [x] = BndryCnds(Node,Element,BdBox,r,l,a,b,c,d)
  eps =1e1*((BdBox(2)-BdBox(1))*...
             (BdBox(4)-BdBox(3))*...
             (BdBox(6)-BdBox(5)))^(1/3)/size(Node,1);
  leftface=find(abs(Node(:,1))<eps );
  Supp = ones(length(leftface),4);
  Supp(:,1) = leftface;
  Loadid=[];
  Loadid=[Loadid;find(abs(Node(:,1)-BdBox(2))<eps & abs(Node(:,2))<eps & abs(Node(:,3)-BdBox(6)/2)<eps)];
  Load = -0.1*ones(size(Loadid,1),4);
  Load(:,1) = Loadid; Load(:,2) = 0; Load(:,4) = 0;
  x = {Supp,Load};
%--------------------------------------------- SPECIFY ANALYTICAL GRADIENTS
function [x] = Normal(P,n1,n2,n3)
  n1(:,1:2) = 0;
  n2(:,1:2) = 0;
  n3(:,1) = -1; n3(:,2) =  1;
  n3(:,3:end) =  0;
  x = {n1,n2,n3};
%-------------------------------------------------------------------------%