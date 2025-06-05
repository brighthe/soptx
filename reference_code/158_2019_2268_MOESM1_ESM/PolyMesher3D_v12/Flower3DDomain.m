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
function [x] = Flower3DDomain(Demand,Arg)
  R = 6;
  BdBox = [-R R -R R -R/6 R/6];
  switch(Demand)
    case('Dist');   x = DistFnc(Arg,BdBox);
    case('BC');     x = BndryCnds(Arg{:},BdBox);
    case('BdBox');  x = BdBox;
    case('PFix');   x = FixedPoints(BdBox);
    case('Normal'); x = Normal(Arg{:});
  end
%----------------------------------------------- COMPUTE DISTANCE FUNCTIONS
function Dist = DistFnc(P,BdBox)
%  addpath('./PolyMesher_v11'); % to use PolyMesher 2D functions
  d1 = dCircle(P,0,0,BdBox(4));
  d2 = dCircle(P,0,0,1/6*BdBox(4));
  Dist = dDiff(d1,d2);
  d3 = [BdBox(5)-P(:,3), P(:,3)-BdBox(6)];
  d3 = [d3,max(d3,[],2)];
  Dist = dIntersect(Dist,d3);
%  rmpath('./PolyMesher_v11');
%---------------------------------------------- SPECIFY BOUNDARY CONDITIONS
function [x] = BndryCnds(Node,Element,BdBox)
  Nload = 8;
  dist = sqrt(Node(:,1).^2+Node(:,2).^2);
  Inner = find(dist-1/6*BdBox(4)<1e-2*BdBox(4));
  Supp = ones(length(Inner),4);
  Supp(:,1) = Inner;
  theta = linspace(0,2*pi,Nload+1);
  theta(end)=[]; Petal=zeros(Nload,1);
  P = BdBox(4) * [cos(theta)' sin(theta)'];
  for i=1:Nload
      aux = sqrt((Node(:,1)-P(i,1)).^2+(Node(:,2)-P(i,2)).^2);
      [~,aux] = sort(aux); Petal(i)=aux(1);
  end
  Load = [Petal -sin(theta)' cos(theta)' zeros(Nload,1)];
  x = {Supp,Load};
%----------------------------------------------------- SPECIFY FIXED POINTS
function [PFix] = FixedPoints(BdBox)
  Nload = 8;
  theta = linspace(0,2*pi,Nload+1);
  theta(end)=[]; Z=zeros(Nload,1);
  PFix = BdBox(4) * [cos(theta)' sin(theta)' Z];
%--------------------------------------------- SPECIFY ANALYTICAL GRADIENTS
function [x] = Normal(P,n1,n2,n3)
  n1(:,3:4) = 0;
  n2(:,3:4) = 0;
  n3(:,3) = -1; n3(:,4) =  1;
  n3(:,1:2) =  0;
  x = {n1,n2,n3};
%-------------------------------------------------------------------------%