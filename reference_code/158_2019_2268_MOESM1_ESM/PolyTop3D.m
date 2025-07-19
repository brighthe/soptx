%------------------------------ PolyTop3D --------------------------------%
% Ref: H Chi,A Pereira, IFM Menezes, GH Paulino , "Virtual Element Method %
% (VEM)-based topology optimization: an integrated framework",            %
% Struct Multidisc Optim, DOI 10.1007/s00158-019-02268-                   %
%-------------------------------------------------------------------------%
function [z,V,fem] = PolyTop3D(fem,opt)
Iter=0; Tol=opt.Tol*(opt.zMax-opt.zMin); Change=2*Tol; z=opt.zIni; P=opt.P;
[E,dEdy,V,dVdy] = opt.MatIntFnc(P*z);
[Fig] = InitialPlot(fem,opt,opt.PF*z);
while (Iter<opt.MaxIter) && (Change>Tol)  
  Iter = Iter + 1;
  %Compute cost functionals and analysis sensitivities
  [f,dfdE,dfdV,fem] = ObjectiveFnc(fem,E,V);
  [g,dgdE,dgdV,fem] = ConstraintFnc(fem,E,V,opt.VolFrac); 
  %Compute design sensitivities
  dfdz = P'*(dEdy.*dfdE + dVdy.*dfdV);
  dgdz = P'*(dEdy.*dgdE + dVdy.*dgdV);
  %Update design variable and analysis parameters
  [z,Change] = UpdateScheme(dfdz,g,dgdz,z,opt);
  [E,dEdy,V,dVdy] = opt.MatIntFnc(P*z);
  %Output results
  fprintf('It: %i \t Objective: %1.3f\tChange: %1.3f\n',Iter,f,Change);
  ZPlot=reshape(Fig.PPlot*opt.PF*z,size(Fig.X));
  [faces,verts] = isosurface(Fig.X,Fig.Y,Fig.Z,ZPlot,0.5);
  set(Fig.Handle,'Faces', faces, 'Vertices', verts); drawnow
end
%------------------------------------------------------- OBJECTIVE FUNCTION
function [f,dfdE,dfdV,vem] = ObjectiveFnc(vem,E,V)
[U,vem] = VEMAnalysis(vem,E);
f = dot(vem.F,U);
temp = cumsum(-U(vem.i).*vem.k.*U(vem.j));
temp = temp(cumsum(vem.ElemNDof.^2));
dfdE = [temp(1);temp(2:end)-temp(1:end-1)];
dfdV = zeros(size(V));
%------------------------------------------------------ CONSTRAINT FUNCTION
function [g,dgdE,dgdV,vem] = ConstraintFnc(vem,E,V,VolFrac)
g = sum(vem.ElemVolume.*V)/sum(vem.ElemVolume)-VolFrac;
dgdE = zeros(size(E));
dgdV = vem.ElemVolume/sum(vem.ElemVolume);
%----------------------------------------------- OPTIMALITY CRITERIA UPDATE
function [zNew,Change] = UpdateScheme(dfdz,g,dgdz,z0,opt)  
zMin=opt.zMin; zMax=opt.zMax;  
move=opt.OCMove*(zMax-zMin); eta=opt.OCEta;
l1=0; l2=1e6;  
while l2-l1 > 1e-4
  lmid = 0.5*(l1+l2);
  B = -(dfdz./dgdz)/lmid;
  zCnd = zMin+(z0-zMin).*B.^eta;
  zNew = max(max(min(min(zCnd,z0+move),zMax),z0-move),zMin);
  if (g+dgdz'*(zNew-z0)>0),  l1=lmid;
  else                       l2=lmid;  end
end
Change = max(abs(zNew-z0))/(zMax-zMin);
%-------------------------------------------------------------- FE-ANALYSIS
function [U,vem] = VEMAnalysis(vem,E)
if ~isfield(vem,'k')
  V = cell(vem.NElem,1);
  for el = 1:vem.NElem
    V{el} = unique([vem.Face{vem.Element{el}}]);
  end
  vem.ElemNDof = 3*cellfun(@length,V); % # of DOFs per element
  vem.i = zeros(sum(vem.ElemNDof.^2),1); 
  vem.j=vem.i; vem.k=vem.i; vem.e=vem.i;
  index = 0;
  for el = 1:vem.NElem
    Ke=LocalK(vem,vem.ShapeFnc{el},el);
    NDof = vem.ElemNDof(el); eNode = V{el};
    eDof = reshape([3*eNode-2;3*eNode-1;3*eNode],NDof,1);
    I=repmat(eDof ,1,NDof); J=I';
    vem.i(index+1:index+NDof^2) = I(:);
    vem.j(index+1:index+NDof^2) = J(:); 
    vem.k(index+1:index+NDof^2) = Ke(:);
    vem.e(index+1:index+NDof^2) = el;
    index = index + NDof^2;
  end
  NLoad = size(vem.Load,1);
  vem.F = zeros(3*vem.NNode,1);  %external load vector
  vem.F(3*vem.Load(1:NLoad,1)-2) = vem.Load(1:NLoad,2);  %x-crdnt
  vem.F(3*vem.Load(1:NLoad,1)-1) = vem.Load(1:NLoad,3);  %y-crdnt
  vem.F(3*vem.Load(1:NLoad,1))   = vem.Load(1:NLoad,4);  %z-crdnt
  NSupp = size(vem.Supp,1);
  FixedDofs = [vem.Supp(1:NSupp,2).*(3*vem.Supp(1:NSupp,1)-2);
               vem.Supp(1:NSupp,3).*(3*vem.Supp(1:NSupp,1)-1)
               vem.Supp(1:NSupp,4).*(3*vem.Supp(1:NSupp,1))];
  FixedDofs = FixedDofs(FixedDofs>0);
  AllDofs   = 1:3*vem.NNode;
  vem.FreeDofs = setdiff(AllDofs,FixedDofs);
end
K = sparse(vem.i,vem.j,E(vem.e).*vem.k);
K = (K+K')/2;
U = zeros(3*vem.NNode,1);
U(vem.FreeDofs,:) = K(vem.FreeDofs,vem.FreeDofs)\vem.F(vem.FreeDofs,:);
%------------------------------------------------- ELEMENT STIFFNESS MATRIX
function [Ke] = LocalK(vem,ShapeFnc,el)
eNode = unique([vem.Face{vem.Element{el}}]); nn = length(eNode);
D=vem.E0/((1+vem.Nu0)*(1-2*vem.Nu0))*[1-vem.Nu0 vem.Nu0 vem.Nu0 0 0 0;
                                      vem.Nu0 1-vem.Nu0 vem.Nu0 0 0 0;
                                      vem.Nu0 vem.Nu0 1-vem.Nu0 0 0 0;
                                                0 0 0 1/2-vem.Nu0 0 0;
                                                0 0 0 0 1/2-vem.Nu0 0;
                                                0 0 0 0 0 1/2-vem.Nu0];
Ke=zeros(3*nn,3*nn);
W=ShapeFnc.W;
Proj=zeros(3*nn);
Proj(1:3:end,1:3:end)=ShapeFnc.P;
Proj(2:3:end,2:3:end)=ShapeFnc.P;
Proj(3:3:end,3:3:end)=ShapeFnc.P;   
B=zeros(6,3*nn); 
B(1,1:3:3*nn) = ShapeFnc.dNdX(:,1)';
B(2,2:3:3*nn) = ShapeFnc.dNdX(:,2)';
B(3,3:3:3*nn) = ShapeFnc.dNdX(:,3)';
B(4,1:3:3*nn) = ShapeFnc.dNdX(:,2)';
B(4,2:3:3*nn) = ShapeFnc.dNdX(:,1)';
B(5,2:3:3*nn) = ShapeFnc.dNdX(:,3)';
B(5,3:3:3*nn) = ShapeFnc.dNdX(:,2)';
B(6,1:3:3*nn) = ShapeFnc.dNdX(:,3)';
B(6,3:3:3*nn) = ShapeFnc.dNdX(:,1)';
Ke = Ke+B'*D*B*W;
alpha=vem.E0*(6-9*vem.Nu0)/9/(1-2*vem.Nu0)/(1+vem.Nu0);% VEM scaling factor
Ke=Ke+alpha*sum(W)^(1/3)*(eye(3*nn)-Proj')*(eye(3*nn)-Proj);
Ke=1/2*(Ke+Ke');
%------------------------------------------------------------- INITIAL PLOT
function [Fig] = InitialPlot(vem,opt,z0)
BdBox = vem.Domain('BdBox');
dmin = 0.5*(sum(vem.ElemVolume)/vem.NElem)^(1/3);
[Fig.X,Fig.Y,Fig.Z]=...
  meshgrid(BdBox(1)-5*dmin/2:dmin:BdBox(2)+5*dmin/2,...
           BdBox(3)-5*dmin/2:dmin:BdBox(4)+5*dmin/2,...
           BdBox(5)-5*dmin/2:dmin:BdBox(6)+5*dmin/2);
h_E=(sum(vem.ElemVolume)/length(vem.Element))^(1/3);
PS1=[Fig.X(:),Fig.Y(:),Fig.Z(:)];
% Dist = vem.Domain('Dist',PS1); I = find(Dist(:,end)<0); PS1=PS1(I,:);
PS2=[opt.NodeD(:,1),opt.NodeD(:,2),opt.NodeD(:,3)];
Fig.PPlot = PolyFilter3D(PS1,PS2,h_E,2);
ZPlot=reshape(Fig.PPlot*opt.PF*z0,size(Fig.X));
[faces,verts] = isosurface(Fig.X,Fig.Y,Fig.Z,ZPlot,0.5);
cla, hold on, view(30,30), rotate3d on, axis equal
box
axis off
Fig.Handle=patch('Faces', faces, 'Vertices', verts,...
     'FaceColor', 'r', 'EdgeColor', 'none',...
     'FaceLighting','gouraud','AmbientStrength',0.5);
hold off;
camlight
drawnow