function TOPLSM(DomainWidth, DomainHight, EleNumPerRow, EleNumPerCol, LV, LCur,FEAInterval, PlotInterval, TotalItNum) 
 %=================================================================% 
% TOPLSM, a 199-line Matlab program, is developed and presented here for the 
% mean compliance optimization of structures in 2D, with the classical level set method.  
% 
% Developed by: Michael Yu WANG, Shikui CHEN and Qi XIA 
% First Version : July 10, 2004 
% Second Version: September 27, 2004 
% Last Modification:October 31, 2005, optimize the code. 
% 
% The code can be downloaded from the webpage: 
% http://www2.acae.cuhk.edu.hk/~cmdl/download.htm 
% 
% Department of Automation and Computer-Aided Engineering,  
% The Chinese University of Hong Kong 
% Email: yuwang@acae.cuhk.edu.hk 
% 
%Main references: 
% (1.)M.Y. Wang, X. M. Wang, and D. M. Guo,A level set method for structural topology optimization, 
% Computer Methods in Applied Mechanics and Engineering, 192(1-2), 227-246, January 2003 
% 
%(2.) M. Y. Wang and X. M. Wang, PDE-driven level sets, shape sensitivity, and curvature flow for structural topology optimization, 
% CMES: Computer Modeling in Engineering & Sciences, 6(4), 373-395, October 2004. 
% 
%(3.) G. Allaire, F. Jouve, A.-M. Toader, Structural optimization using sensitivity analysis and a level-set method ,  
% J. Comp. Phys. Vol 194/1, pp.363-393 ,2004.   
 
%Parameters: 
% DomainWidth : the width of the design domain; 
% DomainHight : the hight of the design domain; 
% EleNumPerRow : the number of finite elements in horizontal direction; 
% EleNumPerCol : the number of finite elements in vertical direction; 
% LV : Lagrange multiplier for volume constraint; 
% LCur : Lagrange multiplier for perimeter constraint whose shape sensitivity is curvature; 
% FEAInterval : parameters to specify the frequency of finite element 
% analysis; 
% PlotInterval : parameters to specify the frequency of plotting; 
% TotalItNum : total iteration number. 
%=================================================================% 
 
% Step 1: Data initialization 
EW = DomainWidth / EleNumPerRow;       %  The width of each finite element. 
EH = DomainHight / EleNumPerCol;          % The hight of each finite element. 
M = [ EleNumPerCol + 1 , EleNumPerRow + 1 ]; % the number of nodes in each dimension 
[ x , y ] = meshgrid( EW * [ -0.5 : EleNumPerRow + 0.5 ] , EH * [ -0.5 : EleNumPerCol + 0.5 ]);  
[ FENd.x, FENd.y, FirstNodePerCol ] = MakeNodes(EleNumPerRow,EleNumPerCol,EW,EH); % get the coordinates of the finite element nodes 
Ele.NodesID = MakeElements( EleNumPerRow, EleNumPerCol, FirstNodePerCol );  
LSgrid.x = x(:); LSgrid.y = y(:);                           % The coordinates of each Level Set grid 
 for i = 1 : length(Ele.NodesID(:,1)) 
    Ele.LSgridID(i) = find((LSgrid.x - FENd.x(Ele.NodesID(i,1)) - EW/2).^2 +... % get the ID of the level set grid that lies in the middle of a finite element 
       (LSgrid.y - FENd.y(Ele.NodesID(i,1)) - EH/2).^2 <= 100*eps); 
 end; 
cx = DomainWidth / 200 * [ 33.33  100  166.67  0   66.67  133.33  200  33.33  100  166.67  0   66.67  133.33  200  33.33  100  166.67]; 
cy = DomainHight / 100 * [   0     0     0     25   25      25     25    50    50    50    75    75     75     75    100   100   100];  
 for i = 1 : length( cx ) 
       tmpPhi( :, i ) = - sqrt ( ( LSgrid . x - cx ( i ) ) .^2 + ( LSgrid . y  - cy ( i ) ) .^2 ) + DomainHight/10;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
end; 
LSgrid.Phi = - (max(tmpPhi.')).';                          % define the initial level set function 
LSgrid.Phi((LSgrid.x - min(LSgrid.x))  .* (LSgrid.x - max(LSgrid.x))  .* (LSgrid.y - max(LSgrid.y)) .* (LSgrid.y - min(LSgrid.y)) <= 100*eps) = -1e-6; 
FENd.Phi = griddata( LSgrid.x, LSgrid.y, LSgrid.Phi, FENd.x, FENd.y, 'cubic'); % project Phi values from the level set function to the finite element nodes 
F = sparse( 2 * (EleNumPerRow + 1 ) * (EleNumPerCol + 1), 1 );  
F( 2 * (EleNumPerRow + 1 ) * (EleNumPerCol + 1) - EleNumPerCol ) = -1; % construct force vector 
ItNum = 1; 
while ItNum <= TotalItNum %  loops begin 
 
%Step 2: Finite element analysis  
disp(' '); disp( ['Finite Element Analysis No. ', num2str(ItNum), ' starts ... '] ); 
FEAStartTime = clock; 
[ FENd.u, FENd.v , MeanCompliance] = FEA(1, 1e-3, F, EleNumPerRow, EleNumPerCol, EW, EH, FENd, Ele, 0.3); % Get dispacement field 
LSgrid.Curv = CalcCurvature(reshape(LSgrid.Phi,M + 1), EW, EH); % Calculate geometric quantities 
 
%Step 3: Shape sensitivity analysis and normal velocity field calculation 
LSgrid.Beta = zeros(size(LSgrid.x)); 
for e = 1: EleNumPerRow * EleNumPerCol        
         LSgrid.Beta(Ele.LSgridID(e)) = SensiAnalysis( 1, 1e-3, Ele.NodesID, FENd.u , FENd.v, EW, EH,... 
         0.3, e, LV, LCur, LSgrid.Phi(Ele.LSgridID(e)), LSgrid.Curv(Ele.LSgridID(e)));     
end; 
LSgrid.Vn = LSgrid.Beta/ max(abs(LSgrid.Beta)); % get velocity field 
disp(['FEA costs ',num2str( etime(clock, FEAStartTime)), ' seconds']); 
 
%Step 4: Level set surface update and reinitialization 
LSgrid.Phi = LevelSetEvolve( reshape(LSgrid.Phi,M + 1), reshape(LSgrid.Vn,M + 1), EW, EH, FEAInterval); 
if (ItNum == 1)||mod(ItNum, 5) ==0 
LSgrid.Phi = Reinitialize(reshape(LSgrid.Phi,M + 1), EW, EH, 20); 
end; 
FENd.Phi = griddata( LSgrid.x, LSgrid.y, LSgrid.Phi, FENd.x, FENd.y, 'cubic'); 
 
% Step 5: Results visualization 
[Obj(ItNum), VolRatio(ItNum), Compliance(ItNum)] = ObjFun( MeanCompliance,LSgrid,DomainWidth, DomainHight,EW,EH,LV); 
if (ItNum == 1)||(mod(ItNum,PlotInterval) == 0) 
    subplot(1,2,1); 
    contourf( reshape(FENd.x , M), reshape(FENd.y , M), reshape(FENd.Phi , M), [0 0] ); 
    axis equal;  grid on; 
    subplot(1,2,2);   contourf( x, y, reshape(-LSgrid.Phi , M + 1), [0  0]);  alpha(0.05);  hold on; 
    h3=surface(x, y, reshape(-LSgrid.Phi , M + 1));  view([37.5  30]);  axis equal;  grid on; 
    set(h3,'FaceLighting','phong','FaceColor','interp', 'AmbientStrength',0.6); light('Position',[0 0 1],'Style','infinite'); 
    pause( 0.5 ); 
end; 
 
%Step 6 : Go to step 2. 
ItNum = ItNum + 1; 
end; 
 
function [KK]=AddBoundCondition(K , EleNumPerCol) 
 %=================================================================% 
 % function [KK]=AddBoundCondition(K , EleNumPerCol) is used to add the boundary condition to the stiffness 
 % Gloabal stiffness matrix K calculated by [y]=Assemble( K , ke,elementsNodeID,EleID). 
 %    
 %=================================================================% 
    n = [ 1 : EleNumPerCol + 1 ]; 
    for i = 1 : length(n) 
    K( 2 * n(i)-1 : 2 * n(i),:) = 0; 
    K( : , 2 * n(i)-1: 2 * n(i)) = 0; 
    K(2 * n(i) - 1, 2 * n(i) - 1) = 1; 
    K(2 * n(i) , 2 * n(i)) = 1; 
    end; 
    KK = K; 
% THE END OF THIS FUNCTION 
 
function [K]=Assemble(K,ke,elementsNodeID,EleID) 
%=================================================================% 
% This function assembles the element stiffness matrix ke of the 
% quadrilateral element into the gloabal stiffness matrix K. 
% This function returns the global stiffness matrix K after the element 
% stiffness matrix ke is assembled. 
% K : gloabal stiffness matrix, with the dimension (2 * TotalNodeNum)-by-(2 * TotalNodeNum). 
% ke : the element stiffness matrix. In terms of a quadratic element, it is 
% an 8-by-8 symmetric matrix. 
% elementsNodeID(1:TotalEleNum, 1:4): stores the node ID in a specified 
% element. 
% EleID : the serial number of a specified finite element. 
%    
%=================================================================% 
m = elementsNodeID(EleID,:); 
for i = 1 : length(m) 
    for j = 1 : length(m) 
       K(2*m(i)-1 : 2*m(i), 2*m(j)-1: 2*m(j)) = K(2 * m(i) - 1 : 2 * m(i) , 2 * m(j) - 1: 2*m(j) )+ ke( 2*i-1: 2*i , 2*j-1:2*j); 
    end; 
end; 
% THE END OF THIS FUNCTION 
 
function [KE] = BasicKe(E,nu, a, b) 
%=================================================================% 
% function [KE] = BasicKe(E,nu, a, b) retuerns the element 
% stiffness matrix of a full/empty element  
% E : Young's modulus; 
% nu: Poisson ratio; 
% a : geometric width of a finite element; 
% b : geometric hight of a finite element; 
% KE : a 8-by-8 stiffness matrix 
%  
%=================================================================% 
k = [  -1/6/a/b*(nu*a^2-2*b^2-a^2),   1/8*nu+1/8, -1/12/a/b*(nu*a^2+4*b^2-a^2),     3/8*nu-1/8, ... 
    1/12/a/b*(nu*a^2-2*b^2-a^2),     -1/8*nu-1/8,     1/6/a/b*(nu*a^2+b^2-a^2),      -3/8*nu+1/8]; 
KE = E/(1-nu^2)*... 
                [ k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8) 
                  k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3) 
                  k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2) 
                  k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5) 
                  k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4) 
                  k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7) 
                  k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6) 
                  k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)];         
% THE END OF THIS FUNCTION 
 
function [ Curvature ] = CalcCurvature(Phi, dx, dy) 
%=================================================================% 
% function [Curvature]= CalcCurvature(Phi, dx, dy) calculates 
% curvature of the level set function,where 
% Phi: is an m-by-n matrix; 
% dx : the interval between two adjacent grids in axis X. 
% dy : the interval between two adjacent grids in axis Y. 
% Curvature:is an m-by-1 vector, which approximates the mean curvature. 
%  
%=================================================================% 
Matrix = Matrix4diff( Phi ); 
Phix = (Matrix.i_plus_1 - Matrix.i_minus_1)/(2 * dx); 
Phiy = (Matrix.j_plus_1 - Matrix.j_minus_1)/(2 * dy); 
[Phiyx_Bk, Phiyx_Fw] = UpwindDiff(Phiy , dx, 'x'); 
Phixy = (Phiyx_Bk + Phiyx_Fw)/2; 
Phixx =(Matrix.i_plus_1 - 2*Phi + Matrix.i_minus_1)/dx^2; 
Phiyy =(Matrix.j_plus_1 - 2*Phi + Matrix.j_minus_1)/dy^2; 
Curvature = (Phixx .* Phiy.^2 - 2 * Phix .* Phiy .* Phixy + Phiyy .* Phix.^2) ./ ((Phix.^2 + Phiy.^2).^1.5 + 100*eps); 
Phix = Phix(:); Phiy = Phiy(:); Curvature = Curvature(:); 
% THE END OF THIS FUNCTION 
 
function [ke]= EleStiffMatrix(EW, EH,E1,E0,NU,Phi,EleNodeID,i) 
%=================================================================% 
% function [ke]= EleStiffMatrix(EW, EH,E1,E0,NU,Phi,EleNodeID,i) returns 
% the element stiffness matrix according to their relative position to the 
% boundary, which includes 3 cases, viz. inside the boudary, outside the 
% boundary or on the boundary. It calls the function  ke = BasicKe(E0, NU, 
% EW, EH); 
% EW: geometric width of a finite element; 
% EH : geometric hight of a finite element; 
% E1 : Young's modulus of elastic material; 
% E0 : Young's modulus of void material; 
% NU: Poisson ratio, assume that the two materials have the same NU; 
% EleNodeID : the corresponding serial number of the 4 nodes in a finite 
% element; 
% i : loop number 
%  
%=================================================================% 
if min(Phi(EleNodeID(i,:))) > 0 % the case that the element is inside the boudary 
    ke = BasicKe(E1, NU, EW, EH); 
elseif max(Phi(EleNodeID(i,:))) < 0 
    ke = BasicKe(E0, NU, EW, EH);% the case that the element is outside the boudary 
else% the case that the element is cut by the boudary 
[ s , t ] = meshgrid([-1 : 0.1 : 1],[-1 : 0.1 : 1]); 
tmpPhi = (1 - s(:)).*(1 - t(:))/4 * Phi(EleNodeID(i,1)) + (1 + s(:)).*(1 - t(:))/4 * Phi(EleNodeID(i,2))... 
    + (1 + s(:)).*(1 + t(:))/4 * Phi(EleNodeID(i,3)) + (1-s(:)).*(1 + t(:))/4 * Phi(EleNodeID(i,4)); 
AreaRatio = length(find( tmpPhi >= 0 ))/length(s(:)); 
ke = AreaRatio * BasicKe(E1,NU,EW, EH); 
end; 
% THE END OF THIS FUNCTION 
 
function [ u, v , MeanCompliance] = FEA(E1, E0, F, EleNumPerRow, EleNumPerCol, EW, EH, FENodes, Ele, NU) 
%=================================================================% 
% function [ u, v , MeanCompliance] = FEA(E1, E0, F, EleNumPerRow, EleNumPerCol, EW, EH, FENodes, Ele, NU) 
% returns the 2-dimensional displacement field and the mean compliance; 
% E1 : Young's modulus of elastic material; 
% E0 : Young's modulus of void material; 
% F : the force vector; 
% EW: geometric width of a finite element; 
% EH : geometric hight of a finite element; 
% FENodes : the structure storing finite element nodes; 
% Ele : the structure storing finite element; 
% NU: Poisson ratio, assume that the two materials have the same NU; 
%  
%=================================================================% 
K = sparse(2 * (EleNumPerRow + 1 )*(EleNumPerCol + 1), 2 * (EleNumPerRow + 1 )*(EleNumPerCol + 1) ); 
    for i=1:EleNumPerRow * EleNumPerCol 
        ke = EleStiffMatrix(EW, EH, E1,E0, NU, FENodes.Phi, Ele.NodesID, i); 
        K = Assemble(K, ke, Ele.NodesID, i); 
    end; 
K = AddBoundCondition(K , EleNumPerCol); 
U = K \ F; 
tmp = 2 * (EleNumPerRow + 1 )*(EleNumPerCol + 1) - EleNumPerCol; 
MeanCompliance = F( tmp ) *U( tmp ); 
for i = 1: 0.5 * length(U) 
u(i,1) = U(2 * i - 1); v( i ,1) = U(2 * i ); 
end; 
% THE END OF THIS FUNCTION 
 
function [Phi1] = LevelSetEvolve(Phi0, Vn, dx, dy, LoopNum) 
%=================================================================% 
% function [Phi1] = LevelSetEvolve(Phi0, Vn, dx, dy, LoopNum) 
% updates the level set surface using a first order space convex. 
% Phi0: is an m-by-n matrix. It's the level set surface before evolution. 
% Vn: is an m-by-n matrix.It's the normal velocity field. 
% Phi1: is an m*n-by-1 vector. It's the level set surface after evolution. 
%    
%=================================================================% 
DetT = 0.5 * min(dx,dy)/max(abs(Vn(:))); 
for i = 1 : LoopNum 
[Dx_L, Dx_R] = UpwindDiff(Phi0 , dx , 'x'); 
[Dy_L, Dy_R] = UpwindDiff(Phi0 , dy , 'y'); 
Grad_Plus  = ((max(Dx_L,0)).^2 + (min(Dx_R , 0)).^2 + (max(Dy_L,0)).^2 + (min(Dy_R,0)).^2 ).^0.5; 
Grad_Minus = ((min(Dx_L,0)).^2 + (max(Dx_R , 0)).^2 + (min(Dy_L,0)).^2 + (max(Dy_R,0)).^2 ).^0.5; 
Phi0 = Phi0 - DetT .* (max(Vn, 0) .* Grad_Plus + min(Vn, 0) .* Grad_Minus); 
end; 
Phi1 = Phi0(:); 
% THE END OF THIS FUNCTION 
 
function [EleNodesID]=MakeElements(EleNumPerRow,EleNumPerCol,FirstNodePerCol) 
%=================================================================% 
%  function [EleNodesID]=MakeElements(EleNumPerRow,EleNumPerCol,FirstNodePerCol)  
%  is used to produce finite elements. 
%  EleNodesID(1:NumEle, 1:4): stands for node ID that make up each element; 
%             O----------------------O 
%              |4                        3 | 
%              |                             | 
%              |                             | 
%              |1                        2 | 
%             O----------------------O 
%  EleNumPerRow: number of element per row 
%  EleNumPerCol: number of element per column 
%  FirstNodePerCol: the serial number of the first node in each column 
%   
%=================================================================% 
EleNodesID = zeros(EleNumPerRow * EleNumPerCol , 4); 
for i=1:EleNumPerRow  
    EleNodesID([i * EleNumPerCol : -1:(i-1) * EleNumPerCol + 1] , 4) = [FirstNodePerCol(i): -1:FirstNodePerCol(i) - EleNumPerCol + 1].'; 
end; 
EleNodesID(:,1)=EleNodesID(:,4)- 1; 
EleNodesID(:,2)=EleNodesID(:,1)+ EleNumPerCol + 1; 
EleNodesID(:,3)=EleNodesID(:,2)+ 1; 
% THE END OF THIS FUNCTION 
 
function [NodesX, NodesY, FirstNodePerCol] = MakeNodes(EleNumPerRow,EleNumPerCol,EleWidth,EleHight) 
%=================================================================% 
% function [nodesPosition,FirstNodePerCol]=MakeNodes(EleNumPerRow,EleNumPerCol,EleWidth,EleHight) 
% is used to make nodesPosition 
% NodesX: an n-by-1 vector discribing the x coordinates of FE nodes 
% NodesY: an n-by-1 vector discribing the x coordinates of FE nodes 
% EleNumPerRow: the number of elements in each row; 
% EleNumPerCol: the number of elements in each col 
% EleWidth: is the width of an element. Here EleWidth=1; 
% EleHight: is the hight of an element. Here EleHight=1; 
%    
%=================================================================% 
[ x , y ]= meshgrid( EleWidth * [ 0 : EleNumPerRow ], EleHight * [0 : EleNumPerCol]); 
FirstNodePerCol = find( y(:) == max(y(:))); 
NodesX = x(:); NodesY = y(:); 
% THE END OF THIS FUNCTION 
 
function [Matrix] = Matrix4diff( Phi ) 
%=================================================================% 
% function [Matrix] = Matrix4diff( Phi ) produces a structure used for 
% upwind finite diffence. 
% Phi: is an m-by-n matrix; 
% Matrix: a structure which includes 4 matrixes used to calculate finite 
% difference. 
%=================================================================% 
Matrix.i_minus_1 = zeros(size(Phi)); 
Matrix.i_plus_1 = Matrix.i_minus_1;  
Matrix.j_minus_1 = Matrix.i_minus_1;  
Matrix.j_plus_1 = Matrix.i_minus_1; 
Matrix.i_minus_1(:, 1) = Phi(:, end);  
Matrix.i_minus_1(:, 2:end) = Phi(:,1:end-1); 
Matrix.i_plus_1(:, end) = Phi(:,1);  
Matrix.i_plus_1(:, 1:end-1) = Phi(:,2:end); 
Matrix.j_minus_1(1, :) = Phi(end, :);  
Matrix.j_minus_1(2:end , :) = Phi(1:end-1,:); 
Matrix.j_plus_1(end,:) = Phi(1,:);  
Matrix.j_plus_1(1:end-1, :) = Phi(2:end, :); 
% THE END OF THIS FUNCTION 
 
function [SignDistPhi] = Reinitialize(Phi0, dx, dy, LoopNum) 
%=================================================================% 
%function [SignDistPhi] = Reinitialize(Phi0, dx, dy, LoopNum) is used to 
% regulize the level set function to be a signed distance function. We  
% adopt the PDE-based method proposed by Peng, Merriman, and Osher 
% etc.,where 
% Phi0: is an m-by-n matrix. It's the level set surface before reinitialization. 
% dx : the interval between two adjacent grids in axis X. 
% dy : the interval between two adjacent grids in axis Y. 
% SignDistPhi : is an m*n-by-1 vector. It's the level set surface after reinitialization. 
%    
%=================================================================% 
for i = 1 : LoopNum + 1 
    [Dx_L, Dx_R] = UpwindDiff(Phi0 , dx , 'x'); 
    [Dy_L, Dy_R] = UpwindDiff(Phi0 , dy , 'y'); 
    Dx_C = (Dx_L + Dx_R)/2; 
    Dy_C = (Dy_L + Dy_R)/2; 
    S = Phi0 ./ (sqrt(Phi0.^2 + (Dx_C.^2 + Dy_C.^2) * dx^2) + eps); 
    DetT = 0.5 * min(dx,dy)/max(abs(S(:))); 
    Grad_Plus  = ((max(Dx_L,0)).^2 + (min(Dx_R , 0)).^2 + (max(Dy_L,0)).^2 + (min(Dy_R,0)).^2 ).^0.5; 
    Grad_Minus = ((min(Dx_L,0)).^2 + (max(Dx_R , 0)).^2 + (min(Dy_L,0)).^2 + (max(Dy_R,0)).^2 ).^0.5; 
    Phi0 = Phi0 - DetT .* ((max(S, 0) .* Grad_Plus + min(S, 0) .* Grad_Minus) - S); 
end; 
SignDistPhi = Phi0(:); 
% THE END OF THIS FUNCTION 
 
function [ Beta ] = SensiAnalysis( E1, E0, EleNodesID, u, v, EW , EH,  NU  , EleID , L4Vol , L4Curv , Phi , curvature) 
%************************************************************************** 
% function [Vn] = Beta4NormVelocity(EW,NU,B,ae,LagrangeMulti, Phi,curvature) is used to calculate 
% the velocity field on a certain grid of level set function. 
% ae = [u1 v1 u2 v2 u3 v3 u4 v4].' : a 8*1 column vector  
% matrix B is a 3*8 strain matrix  
%    
%************************************************************************** 
ae = [ u(EleNodesID(EleID,1)); v(EleNodesID(EleID,1)); u(EleNodesID(EleID,2)); v(EleNodesID(EleID,2)); 
         u(EleNodesID(EleID,3)); v(EleNodesID(EleID,3)); u(EleNodesID(EleID,4)); v(EleNodesID(EleID,4))]; 
 
B = 1/2*[ -1/EW,           0,  1/EW,           0,  1/EW,           0, -1/EW,           0; 
                 0, -1/EH,           0, -1/EH,           0,  1/EH,           0,  1/EH; 
          -1/EH, -1/EW, -1/EH,  1/EW,  1/EH,  1/EW,  1/EH, -1/EW]; 
strain = B * ae;% strain isa 3*1 col. vector 
if Phi> 0.75 * EW 
    E = E1; 
elseif Phi< -0.75 * EW 
    E = E0; 
else 
    DENSITY_Min = 1e-3; 
    xd = Phi / (0.75 * EW); 
    E = E1 * 0.75 * ( 1.0 - DENSITY_Min ) * ( xd - xd^3 / 3.0 ) + 0.5 * ( 1 + DENSITY_Min ); 
end; 
D = E / (1 - NU^2) * [ 1  , NU , 0; NU , 1 , 0; 0 , 0, (1 - NU)/2 ]; 
StainEnergy = strain.' * D * strain; 
Beta = L4Vol - StainEnergy - L4Curv * curvature; 
% THE END OF THIS FUNCTION 
 
function [BackDiff, FawdDiff] = UpwindDiff(Phi , dx , strDirection) 
%=================================================================% 
% function [BkDiff, FwDiff] = UpwindDiff(Phi , dx) calculates 
% backward and forward finite difference,where 
% Phi: is an n-by-n matrix; 
% dx : the interval between two adjacent grids in axis X. 
% strDirection : is a character string. It equals to 'x'or 'y'which mean 
% get spacial derivatives in x direction or y direction. 
% BackDiff:is an n-by-n matrix, which stores (Phi(i,j) - Phi(i-1 ,j))/dx; 
% FawdDiff:is an n-by-n matrix, which stores (Phi(i+1,j) - Phi(i ,j))/dx; 
%    
%=================================================================% 
Matrix = Matrix4diff( Phi ); 
if strDirection == 'x' 
    BackDiff = (Phi - Matrix.i_minus_1)/dx; 
    FawdDiff = (Matrix.i_plus_1 - Phi)/dx; 
elseif strDirection == 'y' 
    BackDiff = (Phi - Matrix.j_minus_1)/dx; 
    FawdDiff = (Matrix.j_plus_1 - Phi)/dx;   
end; 
% THE END OF THIS FUNCTION 
 
% ************************************Disclaimer********************************************% 
% The authors reserve all rights for the programs. The programs may be distributed and 
% used for academic and educational purposes. The authors do not guarantee that the 
% code is free from errors, and they shall not be liable in any event caused by the use  
% of the programs. 
%=================================================================% 
 
 
 