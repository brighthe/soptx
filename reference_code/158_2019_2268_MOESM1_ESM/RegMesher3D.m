function [Node,Face,Element,Supp,Load,Seeds] = RegMesher3D(Domain,Nmin)
BdBox = Domain('BdBox');
L = [BdBox(2)-BdBox(1),BdBox(4)-BdBox(3),BdBox(6)-BdBox(5)];
dmin = min(L)/Nmin; N = ceil(L./dmin); d = N*dmin-L;
[x1,y1,z1]=meshgrid(BdBox(1)-d(1)/2:dmin:BdBox(2)+d(1)/2,...
                    BdBox(3)-d(2)/2:dmin:BdBox(4)+d(2)/2,...
                    BdBox(5)-d(3)/2:dmin:BdBox(6)+d(3)/2);  
% -----------------------------------------Regular hexahedra dominated mesh
Seeds=[x1(:),y1(:),z1(:)];
% ---------------------------------------Truncated octahedra dominated mesh         
% Seeds=[x1(:),y1(:),z1(:);x1(:)+dmin/2,y1(:)+dmin/2,z1(:)+dmin/2]; 
% -------------------------------------- Remove seeds outside of the domain
I = Domain('Dist',Seeds); Seeds=Seeds(I(:,end)<0,:);% 
[Node,Face,Element,Supp,Load] = PolyMesher3D(Domain,length(Seeds),3,Seeds,'alpha');