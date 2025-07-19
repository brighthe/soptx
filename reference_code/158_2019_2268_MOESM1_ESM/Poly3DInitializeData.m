%--------------------------------- PolyTop3D -----------------------------%
% Ref: H Chi,A Pereira, IFM Menezes, GH Paulino , "Virtual Element Method %
% (VEM)-based topology optimization: an integrated framework",            %
% Struct Multidisc Optim, DOI 10.1007/s00158-019-02268-w                  %
%-------------------------------------------------------------------------%
function [vem,NodeD,PV] = Poly3DInitializeData(vem)
[NodeD,FaceD] = Poly3DGenEdgeNodes(vem.Node,vem.Face);
if ~isfield(vem,'ShapeFnc')
  for el=1:vem.NElem
    [wQ,xQ,N,dNdX,Proj,PhiD] = Poly3DVEMShpFun(vem,el,NodeD,FaceD);
    vem.ShapeFnc{el}.W=wQ;
    vem.ShapeFnc{el}.X=xQ;
    vem.ShapeFnc{el}.PhiD=PhiD;
    vem.ShapeFnc{el}.dNdX=dNdX;
    vem.ShapeFnc{el}.P=Proj;
  end
end
PV = NodeElementMatrix(NodeD,FaceD,vem);
if ~isfield(vem,'ElemVolume')
  vem.ElemVolume = zeros(vem.NElem,1);
  for el=1:vem.NElem
    PT = vem.Node(unique([vem.Face{vem.Element{el}}]),:);
    DT =delaunayTriangulation(PT);
    T=DT.ConnectivityList;
    nT  = size(T,1); Ve = 0;
    for i = 1:nT
      tetra = PT(T(i,:),:);
      vol = abs(det(tetra(1:3,:) - tetra([4 4 4],:)) / 6);
      Ve = Ve + vol;
    end
    vem.ElemVolume(el) = Ve;
  end
end
% --------------------------------------GENERATE MID-EDGE NODES FOR DENSITY
function [Node,Element] = Poly3DGenEdgeNodes(Node0,Face0)
Face = Face0; NFace = size(Face,1);
NNode = size(Node0,1);
Edge = GetEdges(NFace,Face); NEdge = size(Edge,1);           % List of Edges
% Node to Edge
Node2Edge=sparse(Edge(:,[1,2]),Edge(:,[2,1]),[1:NEdge,1:NEdge],NNode,NNode);
Face2Edge = cell(NFace,1);                                   % Face to Edge
for i=1:NFace
  nn = size(Face{i},2);
  e = [Face{i}',Face{i}([2:nn 1])'];
  Face2Edge{i} = zeros(1,nn);
  for j=1:nn
    Face2Edge{i}(j) = Node2Edge(e(j,1),e(j,2));
  end
end
Edge2Face = cell(NEdge,1);                                   % Edge to Face
for i=1:NFace
  for j=1:size(Face{i},2)
    Edge2Face{Face2Edge{i}(j)} = [Edge2Face{Face2Edge{i}(j)} i];
  end
end
Node = zeros(NNode+NEdge,3); Node(1:NNode,:) = Node0;
Element = Face0;
for ed=1:NEdge
  Node(NNode+ed,:) = (Node(Edge(ed,1),:)+Node(Edge(ed,2),:))/2;
  for i=1:length(Edge2Face{ed})
    temp=Element{Edge2Face{ed}(i)};
    ind1=find(temp==Edge(ed,1));
    ind2=find(temp==Edge(ed,2));
    if abs(ind2-ind1)>1
        Element{Edge2Face{ed}(i)} = [temp,NNode+ed];       
    elseif ind1>ind2
        Element{Edge2Face{ed}(i)} = [temp(1:ind2),NNode+ed,temp(ind1:end)];
    elseif ind2>ind1
        Element{Edge2Face{ed}(i)} = [temp(1:ind1),NNode+ed,temp(ind2:end)];
    end
  end
end
%------------------------------------------Obtain list of edges in the mesh
function Edge = GetEdges(NFace,Face)
Edge = zeros(sum(cellfun(@length,Face(1:NFace))),2); 
index = 0;
for fc=1:NFace
  NNodeFace = size(Face{fc},2);
  Edge(index+1:index+NNodeFace,:) = [Face{fc}',Face{fc}([2:end 1])'];
  index = index+NNodeFace;
end
Edge =unique(sort(Edge,2),'rows');
% --------------------------------------- VEM SHAPEFUNCTION AND PROJECTIONS
function [wQ,xQ,Phi,GradPhi,Proj,PhiD] = Poly3DVEMShpFun(vem,el,NodeD,FaceD)
FG = vem.Face(vem.Element{el}); % Faces: global node IDs
N = unique([FG{:}]); % Element nodes IDs
V = vem.Node(N,:); % Element nodes coordinates
F = arrayfun(@(x) find(N == x,1,'first'), [FG{:}]); % Faces: local node IDs
F = mat2cell(F,1,cellfun(@length,FG))';
FGD = FaceD(vem.Element{el}); % Faces: global node IDs
ND = unique([FGD{:}]); % Element nodes IDs
VD = NodeD(ND,:); % Element nodes coordinates
FD = arrayfun(@(x) find(ND == x,1,'first'), [FGD{:}]); % Faces: local node IDs
FD = mat2cell(FD,1,cellfun(@length,FGD))';
[~,~,TetVol,TetCent] = PolyTrnglt(V,F); % triangulate polyhedron
V_e = sum(TetVol);
C_e = sum([TetVol TetVol TetVol].*TetCent)./V_e;
m = size(V,1); mD = size(VD,1); nf = size(F,1); 
GradPhi = zeros(m,3); PhiD = zeros(mD,1);
for fc=1:nf
%---------------- COMPUTE DISP PROJECTION: SHAPE FUNCTION GRADIENT (Eq. 19)
  Face = F{fc}; mf = length(Face); VF = V(Face,:);
  [xC,~,Normal] = PolygonCntrd(VF);
  % Check normal signs
  NormalSign=1;if dot(xC-C_e,Normal)<0;NormalSign=-1; end
  ind1=[2:mf,1]; ind2=[mf,1:(mf-1)];
  GradPhi(Face,:)=GradPhi(Face,:)+1/4/V_e*NormalSign*cross(VF-repmat(xC,mf,1),VF(ind1,:)-VF(ind2,:),2);
%------------------------ COMPUTE DENS. PROJECTION: SHAPE FUNCTION (Eq. 26)   
  FaceDD = FD{fc}; mfD = length(FaceDD); VFD = VD(FaceDD,:);
  ind1D=[2:mfD,1]; ind2D=[mfD,1:(mfD-1)];   
  PhiD(FaceDD,1)=PhiD(FaceDD,1)+...
    diag(1/12/V_e*NormalSign*cross(VFD(ind1D,:)-...
          VFD(ind2D,:),(VFD-repmat(C_e,mfD,1)),2)*(VFD-repmat(xC,mfD,1))'); 
end
xQ=C_e; wQ=V_e;
%----------------- COMPUTE DISP PROJECTION: SHAPE FUNCTION (Eqs. 21 and 25)
Phi_Proj= @(x) GradPhi*([x(:,1)';x(:,2)';x(:,3)']-[sum(V(:,1))/m;sum(V(:,2))/m;sum(V(:,3))/m])+1/m;
Phi=zeros(m,1); Phi(:,1)=Phi_Proj(C_e);
S=[GradPhi,1/m*ones(m,1)];
Pi=[V(:,1)-sum(V(:,1))/m,V(:,2)-sum(V(:,2))/m,V(:,3)-sum(V(:,3))/m,ones(m,1)];
Proj=Pi*S';
%--------------------------------------------------------- POLYGON CENTROID
function [Ce,Ae,Normal] = PolygonCntrd(Node)
n=size(Node,1);  Ae = 0; Amax = 0; Ce=zeros(1,3);
for i=1:n-2
  tri = Node([1 i+1 i+2],:);
  centi = mean(tri);
  tmp=cross(tri(2,:)-tri(1,:),tri(3,:)-tri(1,:));
  area = norm(tmp/2);
  Ce = Ce + centi * area;
  Ae = Ae + area;
  if (area > Amax)
    Amax = area;
    Normal = tmp/norm(tmp);
  end
end
Ce=Ce./Ae;
%-------------------------------------------- POLYHEDRON TETRAHEDRALIZATION
function [N,Tet,Vol,Cent] = PolyTrnglt(V,F)
m = length(V); N = zeros(m+1,3);
Cent=zeros(m+1,3); Vol = zeros(m+1,1);
N(1:m,:) = V; N(m+1,:) = mean(V);
modeFace = cellfun(@length,F);
Tet = zeros(sum(modeFace-2),4); index = 0;
for i=1:size(F,1)
  for j=1:size(F{i},2)-2
    index = index + 1;
    Tet(index,:) = [F{i}(1) F{i}(j+1) F{i}(j+2) m+1];
    X = N(Tet(index,:),:);
    xyz = [1 X(1,1) X(1,2) X(1,3); 1 X(2,1) X(2,2) X(2,3);...
           1 X(3,1) X(3,2) X(3,3); 1 X(4,1) X(4,2) X(4,3)];
    Vol(index) = det(xyz)/6;
    Cent(index,:) = mean(X);
    if Vol(index)<0
      tmp = Tet(index,1);
      Tet(index,1) = Tet(index,2);
      Tet(index,2) = tmp;
      Vol(index) = -Vol(index);
    end
  end
end
%------------------------------------------------ COMPUTE PROJECTION MATRIX
function [P] = NodeElementMatrix(NodeD,FaceD,vem)
NNode = size(NodeD,1);
NElem = size(vem.Element,1);
i=zeros(NNode,1);
j=i; p=i;
index = 0;
for el = 1:NElem 
   eNode = unique([FaceD{vem.Element{el}}]);
   nn = length(eNode);
   i(index+1:index+nn)= el;
   j(index+1:index+nn)= eNode;
   p(index+1:index+nn)= vem.ShapeFnc{el}.PhiD;
   index = index + nn;
end   
P = sparse(i,j,p);