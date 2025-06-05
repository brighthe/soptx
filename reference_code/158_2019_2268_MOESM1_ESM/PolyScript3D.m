%----------------------------- PolySript3D -------------------------------%
% Ref: H Chi,A Pereira, IFM Menezes, GH Paulino , "Virtual Element Method %
% (VEM)-based topology optimization: an integrated framework",            %
% Struct Multidisc Optim, DOI 10.1007/s00158-019-02268-w                  %
%-------------------------------------------------------------------------%

%% ---------------------------------------------------- CREATE 'fem' STRUCT
addpath(genpath('./PolyMesher3D_v12')); Domain = @Flower3DDomain;
%[Node,Face,Element,Supp,Load] = PolyMesher3D(Domain,500,10); % CVT
[Node,Face,Element,Supp,Load] = RegMesher3D(Domain,7);
%load('Example_DATA\Ex1_Hex_dominatedMesh.mat');
vem = struct(...
  'Domain',Domain,...          % Domain function
  'NNode',size(Node,1),...     % Number of nodes
  'NFace',size(Face,1),...     % Number of faces
  'NElem',size(Element,1),...  % Number of elements
  'Node',Node,...              % [NNode x 3] array of nodes
  'Face',{Face},...            % [NFace x Var] cell array of faces
  'Element',{Element},...      % [NElement x Var] cell array of elements
  'Supp',Supp,...              % Array of supports
  'Load',Load,...              % Array of loads
  'Nu0',0.3,...                % Poisson's ratio of solid material
  'E0',100.0,...               % Young's modulus of solid material
  'Reg',0 ...                  % Tag for regular meshes
   );
clear Node Face Element Supp Load Domain
[vem,NodeD,PV] = Poly3DInitializeData(vem);
%% ---------------------------------------------------- CREATE 'opt' STRUCT
R = 0.45; q = 2;
VolFrac = 0.1;
m = @(y)MatIntFnc(y,'SIMP',3);
PF = PolyFilter3D(NodeD,NodeD,R,q); P=PV*PF;     
zIni = VolFrac*ones(size(P,2),1);
opt = struct(...               
  'zMin',0.0,...               % Lower bound for design variables
  'zMax',1.0,...               % Upper bound for design variables
  'zIni',zIni,...              % Initial design variables
  'NodeD',NodeD,...            % [NNodeD x 3] array of nodes
  'MatIntFnc',m,...            % Handle to material interpolation fnc.
  'P',P,...                    % Matrix that maps design to element vars.
  'PF',PF,...                  % Matrix that maps design to design vars.
  'VolFrac',VolFrac,...        % Specified volume fraction cosntraint
  'Tol',0.01,...               % Convergence tolerance on design vars.
  'MaxIter',20,...             % Max. number of optimization iterations
  'OCMove',0.3,...             % Allowable move step in OC update scheme
  'OCEta',0.5 ...              % Exponent used in OC update scheme
   );
clear R q VolFrac m P PF PV NodeD zIni
%% ---------------------------------------------------------- RUN 'PolyTop'
figure;
for penal = 1:1:3        %Continuation on the penalty parameter
   disp(['current p: ', num2str(penal)]);
   if penal==3; opt.MaxIter=150; end
   opt.MatIntFnc = @(y)MatIntFnc(y,'SIMP',penal);
   [opt.zIni,V,vem] = PolyTop3D(vem,opt);
end
rmpath(genpath('./PolyMesher3D_v12'));
%% ------------------------------------------------------------------------