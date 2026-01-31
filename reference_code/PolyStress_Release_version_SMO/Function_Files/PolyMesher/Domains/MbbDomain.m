%------------------------------ PolyMesher -------------------------------%
% MBB梁（右半部分）的域定义文件                                             %
%-------------------------------------------------------------------------%
function [x] = MbbDomain(Demand,Arg)
  BdBox = [0 60 0 20];  % [xmin xmax ymin ymax]
  switch(Demand)
    case('Dist');  x = DistFnc(Arg,BdBox);
    case('BC');    x = BndryCnds(Arg{:},BdBox);
    case('BdBox'); x = BdBox;
    case('PFix');  x = FixedPoints(BdBox);
  end
%----------------------------------------------- COMPUTE DISTANCE FUNCTIONS
function Dist = DistFnc(P,BdBox)
  Dist = dRectangle(P,BdBox(1),BdBox(2),BdBox(3),BdBox(4));
%---------------------------------------------- SPECIFY BOUNDARY CONDITIONS
function [x] = BndryCnds(Node,Element,BdBox)
  eps = 0.1*sqrt((BdBox(2)-BdBox(1))*(BdBox(4)-BdBox(3))/size(Node,1));
  
  % 左侧边节点：对称约束 (ux = 0)
  LeftEdgeNodes = find(abs(Node(:,1)-BdBox(1))<eps);
  
  % 右下角节点：滚动支座 (uy = 0)
  RightBottomNode = find(abs(Node(:,1)-BdBox(2))<eps & ...
                         abs(Node(:,2)-BdBox(3))<eps);
  
  % 构建支撑数组
  nLeft = length(LeftEdgeNodes);
  Supp = zeros(nLeft + 1, 3);
  Supp(1:nLeft, 1) = LeftEdgeNodes;
  Supp(1:nLeft, 2) = 1;  % 约束ux
  Supp(1:nLeft, 3) = 0;  % 不约束uy
  Supp(nLeft+1, 1) = RightBottomNode(1);
  Supp(nLeft+1, 2) = 0;  % 不约束ux
  Supp(nLeft+1, 3) = 1;  % 约束uy

  % %% 集中载荷的方式
  % % 左上角节点：载荷施加点
  % LeftTopNode = find(abs(Node(:,1)-BdBox(1))<eps & ...
  %                    abs(Node(:,2)-BdBox(4))<eps);
  % 
  % % 构建载荷数组
  % n = length(LeftTopNode);
  % Load = [LeftTopNode, zeros(n,1), -400*ones(n,1)/n];

  %% 分布载荷的方式
  % 上边缘左侧区域节点：分布载荷
  % 载荷分布在上边缘 x <= 0.1*H 的范围内
  H = BdBox(4) - BdBox(3);  % 梁高度
  d = 0.1 * H;  % 载荷分布长度
  TopLeftNodes = find(abs(Node(:,2)-BdBox(4))<eps & ...
                      Node(:,1) <= BdBox(1) + d);

  % 构建载荷数组：总力 P=-400 均分到各节点
  P_total = -400;
  n = length(TopLeftNodes);
  Load = [TopLeftNodes, zeros(n,1), P_total/n * ones(n,1)];
  
  x = {Supp,Load};
%----------------------------------------------------- SPECIFY FIXED POINTS
function [PFix] = FixedPoints(BdBox)
  PFix = [BdBox(1), BdBox(4);   % 左上角
          BdBox(2), BdBox(3)];  % 右下角
%-------------------------------------------------------------------------%