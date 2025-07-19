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
function [Node,Face,Element,Supp,Load,P] = PolyMesher3D(Domain,NElem,MaxIter,P,type,FixedSeeds)
if(nargin < 5), type = 'all'; end % Default type
if ~exist('P','var') || isempty(P)
  P=PolyMshr_RndPtSet(NElem,Domain);
end
NElem = size(P,1);
Tol=5e-3; It=0; Err=1; c=1;
BdBox = Domain('BdBox');  PFix = Domain('PFix');
Volume = (BdBox(2)-BdBox(1))*(BdBox(4)-BdBox(3))*(BdBox(6)-BdBox(5));
Pc = P; % figure;
while(It<=MaxIter && Err>Tol)
  Alpha = c*(Volume/NElem)^(1/3);
  switch(lower(type))
    case {'alpha'}
      d = Domain('Dist',P);
      ID = abs(d(:,end))<Alpha;
    case {'fixed'}            
      ID = setdiff(1:size(P,1),FixedSeeds);
    otherwise % all
      ID = 1:size(P,1);
  end
  P(ID,:) = Pc(ID,:); %Lloyd's update
  R_P = PolyMshr_Rflct(P,NElem,Domain,Alpha);   %Generate the reflections
  [P,R_P] = PolyMshr_FixedPoints(P,R_P,PFix); % Fixed Points 
  [Node,Element] = voronoin([P;R_P]);           %Construct Voronoi diagram
  [Pc,V] = PolyMshr_CntrdPly(Element,Node,NElem);
  Volume = sum(abs(V));
  Err = sqrt(sum((V.^2).*sum((Pc-P).*(Pc-P),2)))*NElem/Volume^1.5;
  fprintf('It: %3d  Error: %1.3e  Volume: %1.3e\n',It,Err,Volume); It=It+1;
end
[Face,Element] = PolyMshr_PolyRep(NElem,Node,Element);  % Compute face list
[Node,Face] = PolyMshr_ExtrNds(size(Face,1),Node,Face); % Extract node list
[Node,Face] = PolyMshr_RsqsNds(Node,Element,Face);
BC=Domain('BC',{Node,Element}); Supp=BC{1}; Load=BC{2}; % Recover BC arrays
PolyMshr_PlotMsh(Node,Face,Element,Supp,Load);          % Plot mesh and BCs
%------------------------------------------------- GENERATE RANDOM POINTSET
function P = PolyMshr_RndPtSet(NElem,Domain)
P=zeros(NElem,3); BdBox=Domain('BdBox'); Ctr=0;
while Ctr<NElem  
  Y(:,1) = (BdBox(2)-BdBox(1))*rand(NElem,1)+BdBox(1);
  Y(:,2) = (BdBox(4)-BdBox(3))*rand(NElem,1)+BdBox(3);
  Y(:,3) = (BdBox(6)-BdBox(5))*rand(NElem,1)+BdBox(5);
  d = Domain('Dist',Y);
  I = find(d(:,end)<0);                 %Index of seeds inside the domain
  NumAdded = min(NElem-Ctr,length(I));  %Number of seeds that can be added
  P(Ctr+1:Ctr+NumAdded,:) = Y(I(1:NumAdded),:);
  Ctr = Ctr+NumAdded;
end
%------------------------------------------------------------- FIXED POINTS
function [P,R_P] = PolyMshr_FixedPoints(P,R_P,PFix)
PP = [P;R_P];
for i = 1:size(PFix,1)
  [B,I] = sort(sqrt((PP(:,1)-PFix(i,1)).^2+...
                    (PP(:,2)-PFix(i,2)).^2+...
                    (PP(:,3)-PFix(i,3)).^2));
  for j = 1:6
    n = PP(I(j),:) - PFix(i,:); n = n/norm(n);
    PP(I(j),:) = PP(I(j),:)-n*(B(j)-0.8*B(1));
  end
end
P = PP(1:size(P,1),:); R_P = PP(1+size(P,1):end,:);
%--------------------------------------------- NORMALS TO THE BDRY SEGMENTS
function [n1,n2,n3] = PolyMshr_Normals(P,d,NElem,Domain)
eps=1e-8; % perturbation for the numerical differentiation
n1 = (Domain('Dist',P+repmat([eps,0,0],NElem,1))-d)/eps;
n2 = (Domain('Dist',P+repmat([0,eps,0],NElem,1))-d)/eps;
n3 = (Domain('Dist',P+repmat([0,0,eps],NElem,1))-d)/eps;
% overwrite the normals with analytical gradients
n = Domain('Normal',{P,n1,n2,n3}); n1 = n{1}; n2 = n{2}; n3 = n{3};
%--------------------------------------------------------- REFLECT POINTSET
function R_P = PolyMshr_Rflct(P,NElem,Domain,Alpha)
eta=0.9; d = Domain('Dist',P);  
NBdrySegs = size(d,2)-1;          %Number of constituent bdry segments
[n1,n2,n3] = PolyMshr_Normals(P,d,NElem,Domain);
I = abs(d(:,1:NBdrySegs))<Alpha;  %Logical index of seeds near the bdry
P1 = repmat(P(:,1),1,NBdrySegs);  %[NElem x NBdrySegs] extension of P(:,1)
P2 = repmat(P(:,2),1,NBdrySegs);  %[NElem x NBdrySegs] extension of P(:,2)
P3 = repmat(P(:,3),1,NBdrySegs);  %[NElem x NBdrySegs] extension of P(:,3)
R_P(:,1) = P1(I)-2*n1(I).*d(I);  
R_P(:,2) = P2(I)-2*n2(I).*d(I);
R_P(:,3) = P3(I)-2*n3(I).*d(I);
d_R_P = Domain('Dist',R_P);
J = abs(d_R_P(:,end))>=eta*abs(d(I)) & d_R_P(:,end)>0;
R_P=R_P(J,:); R_P=unique(R_P,'rows');
%---------------------------------------------- COMPUTE CENTROID OF POLYGON
function [Pc,V] = PolyMshr_CntrdPly(Element,Node,NElem)
Pc=zeros(NElem,3); V=zeros(NElem,1);
for el = 1:NElem
  P = Node(Element{el},:);
  if any(isinf(P(:)) | isnan(P(:)))
    disp('Error::PolyMshr_CntrdPly: Inf of NaN node');
  end
  DT =delaunayTriangulation(P);
  T=DT.ConnectivityList;
  nT  = size(T,1);  C = zeros(1,3); Ve = 0;
  for i = 1:nT
    tetra = P(T(i,:),:);
    centi = mean(tetra);
    vol = abs(det(tetra(1:3,:) - tetra([4 4 4],:)) / 6);
    C = C + centi * vol;
    Ve = Ve + vol;
  end
  Pc(el,:)=C./Ve;
  V(el) = Ve;
end
%------------------------------------------------------- EXTRACT MESH NODES
function [Node,Element] = PolyMshr_ExtrNds(NElem,Node0,Element0)
map = unique([Element0{1:NElem}]);
cNode = 1:size(Node0,1);
cNode(setdiff(cNode,map)) = max(map);
[Node,Element] = PolyMshr_RbldLists(Node0,Element0(1:NElem),cNode);
%--------------------------------------------------------- RESEQUENSE NODES
function [Node,Face] = PolyMshr_RsqsNds(Node0,Element0,Face0)
NNode0=size(Node0,1); NElem0=size(Element0,1);
ElementND=cell(NElem0,1);
for el=1:NElem0
   FG = Face0(Element0{el}); 
   ElementND{el}=unique([FG{:}]);
end
ElemLnght=cellfun(@length,ElementND); nn=sum(ElemLnght.^2); 
i=zeros(nn,1); j=zeros(nn,1); s=zeros(nn,1); index=0;
for el = 1:NElem0
  eNode=ElementND{el}; ElemSet=index+1:index+ElemLnght(el)^2;
  i(ElemSet) = kron(eNode,ones(ElemLnght(el),1))';
  j(ElemSet) = kron(eNode,ones(1,ElemLnght(el)))';
  s(ElemSet) = 1;
  index = index+ElemLnght(el)^2;
end
K = sparse(i,j,s,NNode0, NNode0);
p = symrcm(K);
cNode(p(1:NNode0))=1:NNode0;
[Node,Face] = PolyMshr_RbldLists(Node0,Face0,cNode);
%------------------------------------------------------------ REBUILD LISTS
function [Node,Element] = PolyMshr_RbldLists(Node0,Element0,cNode)
% This function uses the package geom3d by David Legland (2019).
% geom3d (https://www.mathworks.com/matlabcentral/fileexchange/24484-geom3d)
% MATLAB Central File Exchange. Retrieved April 18, 2019.
addpath(genpath('./geom3d')); % to use angleSort3d
Element = cell(size(Element0,1),1);
[~,ix,jx] = unique(cNode);
if ~isequal(size(jx),size(cNode)), jx=jx'; end % +R2013a compatibility fix
if size(Node0,1)>length(ix), ix(end)=max(cNode); end
Node = Node0(ix,:); 
for el=1:size(Element0,1)
  Element{el} = unique(jx(Element0{el}));
  [~,iix]  = angleSort3d(Node(Element{el},:));
  SortedElement = Element{el}(iix);
  Element{el} = SortedElement(:)';
end
rmpath(genpath('./geom3d')); % to use angleSort3d
%---------------------------------------------- POLYHEDRON REPRESENTATITION
function [Face,Elem2Face] = PolyMshr_PolyRep(InElem,Node,Element)
NElem = size(Element,1); NNode = size(Node,1);
Node2Elem = cell(NNode,1);
for el=1:NElem
  tmp = Element{el};
  for i=1:length(tmp)
    Node2Elem{tmp(i)} = [Node2Elem{tmp(i)} el];
  end
end
NNode2Elem = cellfun(@length,Node2Elem);
Elem2Face = cell(NElem,1);
Face = cell(20*NElem,1); NFace = 0;
for el=1:InElem
  ElementNeighbor = zeros(sum(NNode2Elem(Element{el})),1);
  index = 0;
  for nd=Element{el}
    ElementNeighbor(index+1:index+NNode2Elem(nd)) = Node2Elem{nd};
    index = index + NNode2Elem(nd);
  end
  ElementNeighbor = setdiff(unique(ElementNeighbor),el);
  for i=1:length(ElementNeighbor)
    if (el>ElementNeighbor(i)), continue; end
    f = intersect(Element{el},Element{ElementNeighbor(i)});
    if ~isempty(f) && length(f)>2
      NFace = NFace + 1;
      Face{NFace} = f;
      Elem2Face{el} = [Elem2Face{el} NFace];
      Elem2Face{ElementNeighbor(i)} = ...
        [Elem2Face{ElementNeighbor(i)} NFace];
    end
  end
end
Face = Face(1:NFace);
Elem2Face = Elem2Face(1:InElem);
%---------------------------------------------------------------- PLOT MESH
function PolyMshr_PlotMsh(Node,Face,Element,Supp,Load)
clf; axis equal; axis off; hold on;
MaxNVer = max(cellfun(@numel,Face));      %Max. num. of vertices in mesh
PadWNaN = @(E) [E NaN(1,MaxNVer-numel(E))];  %Pad cells with NaN
ElemMat = cellfun(PadWNaN,Face,'UniformOutput',false);
ElemMat = vertcat(ElemMat{:});               %Create padded element matrix
patch('Faces',ElemMat,'Vertices',Node,'FaceColor','w'); pause(1e-6)
if exist('Supp','var')&&~isempty(Supp) % Plot Supp BC if specified
  plot3(Node(Supp(:,1),1),Node(Supp(:,1),2),...
        Node(Supp(:,1),3),'b>','MarkerSize',8);
end
if exist('Load','var')&&~isempty(Load) % Plot Load BC if specified
  plot3(Node(Load(:,1),1),Node(Load(:,1),2),...
        Node(Load(:,1),3),'m^','MarkerSize',8);
end
hold off;
%-------------------------------------------------------------------------%