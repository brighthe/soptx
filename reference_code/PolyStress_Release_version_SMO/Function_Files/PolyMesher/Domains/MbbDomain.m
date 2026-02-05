%------------------------------ PolyMesher -------------------------------%
% Ref: C Talischi, GH Paulino, A Pereira, IFM Menezes, "PolyMesher: A     %
%      general-purpose mesh generator for polygonal elements written in   %
%      Matlab," Struct Multidisc Optim, DOI 10.1007/s00158-011-0706-z     %
%-------------------------------------------------------------------------%
function [x] = MbbDomain(Demand,Arg)
  BdBox = [0 3 0 1];
  switch(Demand)
    case('Dist');  x = DistFnc(Arg,BdBox);
    case('BC');    x = BndryCnds(Arg{:},BdBox);  % ← 修改：添加 {:} 解包
    case('BdBox'); x = BdBox;
    case('PFix');  x = FixedPoints(BdBox);       % ← 添加：补充接口
  end
%----------------------------------------------- COMPUTE DISTANCE FUNCTIONS
function Dist = DistFnc(P,BdBox)
  Dist = dRectangle(P,BdBox(1),BdBox(2),BdBox(3),BdBox(4));
%---------------------------------------------- SPECIFY BOUNDARY CONDITIONS
function [x] = BndryCnds(Node,Element,BdBox)  % ← 修改：添加 Element 参数
  eps = 0.1*sqrt((BdBox(2)-BdBox(1))*(BdBox(4)-BdBox(3))/size(Node,1));
  LeftEdgeNodes = find(abs(Node(:,1)-BdBox(1))<eps);
  RigthBottomNode = find(abs(Node(:,1)-BdBox(2))<eps & ...
                         abs(Node(:,2)-BdBox(3))<eps);
  
  % 左边缘上部区域的分布载荷（与 L_bracketDomain 一致）
  LoadNodes = find(abs(Node(:,1)-BdBox(1))<eps & ...
                   Node(:,2) > 0.85*BdBox(4));
  
  FixedNodes = [LeftEdgeNodes; RigthBottomNode];
  Supp = zeros(length(FixedNodes),3);
  Supp(:,1) = FixedNodes; 
  Supp(1:end-1,2) = 1;  % 左边缘：x方向约束
  Supp(end,3) = 1;       % 右下角：y方向约束
  
  n = length(LoadNodes);
  Load = [LoadNodes, zeros(n,1), -ones(n,1)/n];  % 总载荷 -1，均分到各节点
  
  x = {Supp,Load};
%----------------------------------------------------- SPECIFY FIXED POINTS
function [PFix] = FixedPoints(BdBox)
  PFix = [];
%-------------------------------------------------------------------------%