clear all
clc
% 屈服应力限制
sigmay = 0.028;
% 体积约束
volfrac = 0.35;
% p-norm 聚合参数
p = 6;
% 应力约束数量
nc = 100;
% 网格参数
nelx = 400;
nely = 400;
% 几何参数
l = 400;
b = 400;
M = l*b;
unit_size_x = l / nelx;
unit_size_y = b / nely;
rmin = 1.5 * (unit_size_x + unit_size_y) / 2;
% 材料参数
E = 1;
nu = 0.3;
% 优化参数
penal = 3;
x(1:nely, 1:nelx) = volfrac;

% 单元刚度矩阵
k = [ 1/2-nu/6   1/8+nu/8 -1/4-nu/12 -1/8+3*nu/8 ... 
     -1/4+nu/12 -1/8-nu/8  nu/6       1/8-3*nu/8];
KE = 1/(1-nu^2)*[ k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
                  k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
                  k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
                  k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
                  k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
                  k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
                  k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
                  k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)]; 
% 应变位移矩阵
% B = (1/2/l) * [-1  0  1  0  1  0 -1  0; 
%                 0 -1  0 -1  0  1  0  1; 
%                -1 -1 -1  1  1  1  1 -1];
B = (1/2/unit_size_x) * [-1  0  1  0  1  0 -1  0; 
                          0 -1  0 -1  0  1  0  1; 
                         -1 -1 -1  1  1  1  1 -1];
% 本构矩阵
C = E/(1-nu^2)*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];           
% 设计变量按行展开
count = 1;
for g=1:nely
    for h=1:nelx
        xval(count, 1) = x(g, h);
        count=count+1;
    end
end

% nc 个应力约束 + 1 个体积约束
m = nc + 1;
% nc 个应力约束
% m = nc;
% 设计变量数量
n = nelx*nely;
% MMA 历史信息
xold1 = zeros(n, 1);
xold2 = zeros(n, 1);
% 移动渐近线
low = zeros(n, 1);
upp = zeros(n, 1);
% 设计变量界限
xmin = 10e-3 * ones(n, 1);
xmax = ones(n, 1);
% MMA 子问题参数
c = 1000 * ones(m, 1); % 松弛变量 yi 的惩罚系数
d = 0;                 % 二次惩罚项
a = zeros(m, 1);       % 约束中 z 的系数
a0 = 1;                % 目标函数中 z 的系数
dfdx_1 = (unit_size_x * unit_size_y / M) * ones(1, n);
% 二阶导数
df0dx2 = zeros(n, 1);
dfdx2 = zeros(m, n);
% 迭代计数器
iter = 0;
itte = 0;
maxite = 120;
x = reshape(xval, [nelx, nely])';

[F, U] = FEA(nelx, nely, x, penal, KE);

% f0val-目标函数; df0dx-目标函数灵敏度; fval-约束函数
[f0val, df0dx, fval] = Load(F,U,x,M,KE,m,n,nelx,nely,penal,unit_size_x,unit_size_y,rmin, volfrac);

% von Mises 应力及其对密度的导数
[von_mises, derivative] = stress_func(C, B, U, nelx, nely, x, p);

% p-norm 聚合函数
[sigmapn, derivative0] = pnorm(p, von_mises, nc, nelx, nely, sigmay);

% 保留体积约束, 添加应力约束
fval(2:m, 1) = sigmapn(1:nc, 1);
% 只考虑应力约束
% fval = sigmapn;

% 应力约束灵敏度
[dfdx_0] = derivative_stress(derivative, derivative0, nc, n, nelx, nely, penal, rmin, x);

% 保留体力约束灵敏度, 添加应力约束灵敏度
dfdx = [dfdx_1; dfdx_0];
% 只考虑应力约束灵敏度
% dfdx = dfdx_0;

ploty = zeros(m, maxite);  % 记录约束值历史
while iter < maxite
    iter = iter + 1;

    ploty(:, iter) = fval;
    
    [xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp] = mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2, ... 
                                                        f0val, df0dx, df0dx2, fval, dfdx, dfdx2, low, upp, a0, a, c, d); 
     
    xold2 = xold1;
    xold1 = xval;
    xval = xmma;
     
    x = reshape(xval, [nelx, nely])';
    
    [F, U] = FEA(nelx, nely, x, penal, KE);
    
    [f0val, df0dx, fval] = Load(F, U, x, M, KE, m, n, nelx, nely, penal, unit_size_x, unit_size_y, rmin, volfrac);
    
    [von_mises, derivative] = stress_func(C, B, U, nelx, nely, x, p);
    
    [sigmapn, derivative0] = pnorm(p, von_mises, nc, nelx, nely, sigmay);
    
    fval(2:m, 1) = sigmapn(1:nc, 1);
    % fval = sigmapn;

    [dfdx_0] = derivative_stress(derivative, derivative0, nc, n, nelx, nely, penal, rmin, x);
    dfdx = [dfdx_1; dfdx_0];
    % dfdx  =dfdx_0;

    %Plotting
    colormap(gray); imagesc(-x); axis equal; axis tight; axis off;pause(1e-6);

    fprintf('Iter: %3d  Obj: %10.4f  Vol: %6.3f  MaxStress: %6.3f\n', iter, f0val, fval(1)+volfrac, max(sigmapn)+1);
     
end

ploty = ploty(:, 1:iter)'; 