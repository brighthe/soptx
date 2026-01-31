%------------------------------- Mesh_Mbb --------------------------------%
% 生成MBB梁（右半部分）的规则四边形网格                                      %
% 使用PolyMesher + 规则种子点的方法                                         %
%-------------------------------------------------------------------------%
function [Node,Element,Supp,Load] = Mesh_MBB(Ne_ap)
  L = 60;   % 长度 (mm)
  H = 20;   % 高度 (mm)
  
  % 计算网格参数
  ratio = L / H;  % 长宽比 = 3
  ny = round(sqrt(Ne_ap / ratio));
  nx = round(ny * ratio);
  he_x = L / nx;
  he_y = H / ny;
  NElem = nx * ny;
  
  % 生成规则网格状的种子点（单元中心）
  [X, Y] = meshgrid(he_x/2:he_x:L-he_x/2, he_y/2:he_y:H-he_y/2);
  P = [X(:), Y(:)];  % Mesh seed
  
  % 调用PolyMesher，MaxIter=0保持种子点规则性
  [Node,Element,Supp,Load,~] = PolyMesher(@MbbDomain, NElem, 0, P);
end
%-------------------------------------------------------------------------%