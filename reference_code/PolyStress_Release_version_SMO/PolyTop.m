%------------------------------ PolyTop ----------------------------------%
% Modified to include Stress Visualization for Comparison
%-------------------------------------------------------------------------%
function [z,V,fem] = PolyTop(fem,opt)
Iter=0; Tol=opt.Tol*(opt.zMax-opt.zMin); Change=2*Tol; z=opt.zIni; P=opt.P;
[E,dEdy,V,dVdy] = opt.MatIntFnc(P*z);

% ==【修改 1】初始化双窗口绘图 (拓扑 + 应力) ==
figure; 
[hV, hS] = InitialPlot(fem, V, 0*V); 
% ==========================================

while (Iter<opt.MaxIter) && (Change>Tol)  
  Iter = Iter + 1;
  % Analysis
  [f,dfdE,dfdV,fem,U] = ObjectiveFnc(fem,E,V); % 让 ObjectiveFnc 返回 U
  [g,dgdE,dgdV,fem] = ConstraintFnc(fem,E,V,opt.VolFrac); 
  
  % Sensitivity
  dfdz = P'*(dEdy.*dfdE + dVdy.*dfdV);
  dgdz = P'*(dEdy.*dgdE + dVdy.*dgdV);
  
  % Update
  [z,Change] = UpdateScheme(dfdz,g,dgdz,z,opt);
  [E,dEdy,V,dVdy] = opt.MatIntFnc(P*z);
  
  % ==【修改 2】计算应力并更新绘图 ==
  % 计算当前应力 (仅用于显示，不参与优化)
  [VM_Stress, ~] = von_Mises_Stress(fem, U); 
  % 归一化应力 (Stress / Limit)
  SM = E .* VM_Stress ./ fem.SLim; 
  
  fprintf('It: %i \t Obj: %1.3f \t Ch: %1.3f \t MaxStressRatio: %1.3f\n', ...
          Iter, f, Change, max(SM));
      
  set(hV,'FaceColor','flat','CData',1-V); drawnow
  set(hS,'FaceColor','flat','CData',SM); drawnow % 更新右侧应力图
  % ==========================================
end

%------------------------------------------------------- OBJECTIVE FUNCTION
function [f,dfdE,dfdV,fem,U] = ObjectiveFnc(fem,E,V)
[U,fem] = FEAnalysis(fem,E); % 这里 U 是节点位移
f = dot(fem.F,U);
temp = cumsum(-U(fem.i).*fem.k.*U(fem.j));
temp = temp(cumsum(fem.ElemNDof.^2));
dfdE = [temp(1);temp(2:end)-temp(1:end-1)];
dfdV = zeros(size(V));

%------------------------------------------------------ CONSTRAINT FUNCTION
function [g,dgdE,dgdV,fem] = ConstraintFnc(fem,E,V,VolFrac)
if ~isfield(fem,'ElemArea')
  fem.ElemArea = zeros(fem.NElem,1);
  for el=1:fem.NElem
    vx=fem.Node(fem.Element{el},1); vy=fem.Node(fem.Element{el},2);
    fem.ElemArea(el) = 0.5*sum(vx.*vy([2:end 1])-vy.*vx([2:end 1]));
  end
end
g = sum(fem.ElemArea.*V)/sum(fem.ElemArea)-VolFrac;
dgdE = zeros(size(E));
dgdV = fem.ElemArea/sum(fem.ElemArea);

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
function [U,fem] = FEAnalysis(fem,E)
if ~isfield(fem,'k')
  fem.ElemNDof = 2*cellfun(@length,fem.Element); 
  fem.i = zeros(sum(fem.ElemNDof.^2),1); 
  fem.j=fem.i; fem.k=fem.i; fem.e=fem.i;
  index = 0;
  if ~isfield(fem,'ShapeFnc'), fem=TabShapeFnc(fem); end
  if fem.Reg, Ke=LocalK(fem,fem.Element{1}); end
  for el = 1:fem.NElem  
    if ~fem.Reg,  Ke=LocalK(fem,fem.Element{el}); end
    NDof = fem.ElemNDof(el);
    eDof = reshape([2*fem.Element{el}-1;2*fem.Element{el}],NDof,1);
    I=repmat(eDof ,1,NDof); J=I';
    fem.i(index+1:index+NDof^2) = I(:);
    fem.j(index+1:index+NDof^2) = J(:); 
    fem.k(index+1:index+NDof^2) = Ke(:);
    fem.e(index+1:index+NDof^2) = el;
    index = index + NDof^2;
  end
  NLoad = size(fem.Load,1);
  fem.F = zeros(2*fem.NNode,1);  
  fem.F(2*fem.Load(1:NLoad,1)-1) = fem.Load(1:NLoad,2);  
  fem.F(2*fem.Load(1:NLoad,1))   = fem.Load(1:NLoad,3);  
  NSupp = size(fem.Supp,1);
  FixedDofs = [fem.Supp(1:NSupp,2).*(2*fem.Supp(1:NSupp,1)-1);
               fem.Supp(1:NSupp,3).*(2*fem.Supp(1:NSupp,1))];
  FixedDofs = FixedDofs(FixedDofs>0);
  AllDofs   = [1:2*fem.NNode];
  fem.FreeDofs = setdiff(AllDofs,FixedDofs);
end
K = sparse(fem.i,fem.j,E(fem.e).*fem.k);
K = (K+K')/2;
U = zeros(2*fem.NNode,1);
U(fem.FreeDofs,:) = K(fem.FreeDofs,fem.FreeDofs)\fem.F(fem.FreeDofs,:);

%------------------------------------------------- ELEMENT STIFFNESS MATRIX
function [Ke] = LocalK(fem,eNode)
D=fem.E0/(1-fem.Nu0^2)*[1 fem.Nu0 0;fem.Nu0 1 0;0 0 (1-fem.Nu0)/2];
nn=length(eNode); Ke=zeros(2*nn,2*nn); 
W = fem.ShapeFnc{nn}.W;
for q = 1:length(W) 
  dNdxi = fem.ShapeFnc{nn}.dNdxi(:,:,q);
  J0 = fem.Node(eNode,:)'*dNdxi; 
  dNdx = dNdxi/J0;
  B = zeros(3,2*nn);
  B(1,1:2:2*nn) = dNdx(:,1)'; 
  B(2,2:2:2*nn) = dNdx(:,2)'; 
  B(3,1:2:2*nn) = dNdx(:,2)'; 
  B(3,2:2:2*nn) = dNdx(:,1)';
  Ke = Ke+B'*D*B*W(q)*det(J0); 
end

%------------------------------------------------- TABULATE SHAPE FUNCTIONS
function fem = TabShapeFnc(fem)
ElemNNode = cellfun(@length,fem.Element); 
fem.ShapeFnc = cell(max(ElemNNode),1);
for nn = min(ElemNNode):max(ElemNNode)
  [W,Q] = PolyQuad(nn);
  fem.ShapeFnc{nn}.W = W;
  fem.ShapeFnc{nn}.N = zeros(nn,1,size(W,1));
  fem.ShapeFnc{nn}.dNdxi = zeros(nn,2,size(W,1));
  for q = 1:size(W,1)
    [N,dNdxi] = PolyShapeFnc(nn,Q(q,:));
    fem.ShapeFnc{nn}.N(:,:,q) = N;
    fem.ShapeFnc{nn}.dNdxi(:,:,q) = dNdxi;
  end
end

%------------------------------------------------ POLYGONAL SHAPE FUNCTIONS
function [N,dNdxi] = PolyShapeFnc(nn,xi)
N=zeros(nn,1); alpha=zeros(nn,1); dNdxi=zeros(nn,2); dalpha=zeros(nn,2);
sum_alpha=0.0; sum_dalpha=zeros(1,2); A=zeros(nn,1); dA=zeros(nn,2);
[p,Tri] = PolyTrnglt(nn,xi);
for i=1:nn
  sctr = Tri(i,:); pT = p(sctr,:);
  A(i) = 1/2*det([pT,ones(3,1)]);
  dA(i,1) = 1/2*(pT(3,2)-pT(2,2));
  dA(i,2) = 1/2*(pT(2,1)-pT(3,1));
end
A=[A(nn,:);A]; dA=[dA(nn,:);dA];
for i=1:nn
  alpha(i) = 1/(A(i)*A(i+1));
  dalpha(i,1) = -alpha(i)*(dA(i,1)/A(i)+dA(i+1,1)/A(i+1));
  dalpha(i,2) = -alpha(i)*(dA(i,2)/A(i)+dA(i+1,2)/A(i+1));
  sum_alpha = sum_alpha + alpha(i);
  sum_dalpha(1:2) = sum_dalpha(1:2)+dalpha(i,1:2);
end
for i=1:nn
  N(i) = alpha(i)/sum_alpha;
  dNdxi(i,1:2) = (dalpha(i,1:2)-N(i)*sum_dalpha(1:2))/sum_alpha;
end
%---------------------------------------------------- POLYGON TRIANGULATION
function [p,Tri] = PolyTrnglt(nn,xi)
p = [cos(2*pi*((1:nn))/nn); sin(2*pi*((1:nn))/nn)]';
p = [p; xi];
Tri = zeros(nn,3); Tri(1:nn,1)=nn+1;
Tri(1:nn,2)=1:nn; Tri(1:nn,3)=2:nn+1; Tri(nn,3)=1;
%----------------------------------------------------- POLYGONAL QUADRATURE
function [weight,point] = PolyQuad(nn)
[W,Q]= TriQuad;                  
[p,Tri] = PolyTrnglt(nn,[0 0]);  
point=zeros(nn*length(W),2); weight=zeros(nn*length(W),1);
for k=1:nn
  sctr = Tri(k,:);
  for q=1:length(W)
    [N,dNds] = TriShapeFnc(Q(q,:));  
    J0 = p(sctr,:)'*dNds;
    l = (k-1)*length(W) + q;
    point(l,:) = N'*p(sctr,:);
    weight(l) = det(J0)*W(q);
  end                                 
end
%---------------------------------------------------- TRIANGULAR QUADRATURE
function [weight,point] = TriQuad
point=[1/6,1/6;2/3,1/6;1/6,2/3]; weight=[1/6,1/6,1/6];   
%----------------------------------------------- TRIANGULAR SHAPE FUNCTIONS
function [N,dNds] = TriShapeFnc(s)
N=[1-s(1)-s(2);s(1);s(2)]; dNds=[-1,-1;1,0;0,1];

% ==【修改 3】新增 InitialPlot (支持双子图) ==
function [handle1,handle2] = InitialPlot(fem,z01,z02)
ElemNodes = cellfun(@length,fem.Element); 
Faces = NaN(fem.NElem,max(ElemNodes));    
for el = 1:fem.NElem; Faces(el,1:ElemNodes(el)) = fem.Element{el}(:); end
ax1 = subplot(1,2,1); title('Element Densities');
patch('Faces',Faces,'Vertices',fem.Node,'FaceVertexCData',0.*z01,...
      'FaceColor','flat','EdgeColor','k','linewidth',1.5);
handle1 = patch('Faces',Faces,'Vertices',fem.Node,'FaceVertexCData',...
                1-z01,'FaceColor','flat','EdgeColor','none');
axis equal; axis off; axis tight; colormap(ax1,gray); caxis([0 1]);
hsp1 = get(gca, 'Position'); 
ax2 = subplot(1,2,2); title('Normalized von Mises Stress')
handle2 = patch('Faces',Faces,'Vertices',fem.Node,'FaceVertexCData',...
                z02,'FaceColor','flat','EdgeColor','none');
axis equal; axis off; axis tight; colormap(ax2,'jet'); c = colorbar;
w = get(c,'Position');
hsp2 = get(gca, 'Position'); 
set(ax1, 'Position', [hsp1(1)-w(3), hsp1(2:end)]);
set(ax2, 'Position', [hsp2(1)-2*w(3), hsp2(2),  hsp1(3:4)]);
drawnow;

% ==【修改 4】新增 von_Mises_Stress (为了画图) ==
function [VM_Stress,dVM_dU] = von_Mises_Stress(fem,U)
% 这是一个简化版，只用于线性材料画图，去掉了 sensitivity 部分的输出
V = [1 -1/2 0; -1/2 1 0; 0 0 3]; % von Mises matrix
ElemU = U(fem.eDof);    
ee_elem = fem.B0*ElemU; 
ee_elem = reshape(ee_elem,3,[]); 
% 注意：这里直接调用 material_model，你需要确保路径里有 material_model.m
[Cauchy_S, D0] = material_model(fem.MatModel,fem.MatParam,ee_elem);
% 简单的应力计算
VM_Stress = max(sqrt(sum(Cauchy_S.*(V*Cauchy_S))),eps)'; 
dVM_dU = []; % 不需要算导数，因为不参与优化