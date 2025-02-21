%    Version September 2007 (and a small change August 2008)
%
%    Krister Svanberg <krille@math.kth.se>
%    Department of Mathematics, SE-10044 Stockholm, Sweden.
%
%    This function mmasub performs one MMA-iteration, aimed at
%    solving the nonlinear programming problem:
%         
%      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
%    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
%                xmin_j <= x_j <= xmax_j,    j = 1,...,n
%                z >= 0,   y_i >= 0,         i = 1,...,m
function [xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp] = mmasub(...
	m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d)
%*** INPUT:
%
%   m    = 一般约束的数量, 即约束函数 f_i(x) 的个数
%   n    = 变量 x_j 的数量.
%  iter  = 当前的迭代次数 (首次调用 mmasub 时, iter=1).
%  xval  = 当前迭代中设计变量 x_j 的值的列向量.
%  xmin  = 变量 x_j 的下界, 列向量形式.
%  xmax  = 变量 x_j 的上界, 列向量形式.
%  xold1 = 前一次迭代的设计变量 xval（如果 iter > 1）.
%  xold2 = 前两次迭代的设计变量 xval（如果 iter > 2）.
%  f0val = 当前设计变量 xval 下, 目标函数 f_0(x) 的值.
%  df0dx = 目标函数 f_0(x) 对设计变量 x_j 的梯度, 列向量形式, 形状(n, 1).
%  fval  = 当前设计变量 xval 下, 约束函数 f_i(x) 的值, 列向量形式, 形状(m, 1).
%  dfdx  = 约束函数 f_i(x) 对设计变量 x_j 的梯度, 矩阵形式, 形状 (m, n),
% 			其中 dfdx(i, j) = partial f_i / partial x_j.
%  low   = 下渐近线的值, 列向量形式, 形状 (n, 1), 如果 iter > 1, 使用前一次迭代的值.
%  upp   = 上渐近线的值, 列向量形式, 形状 (n, 1), 如果 iter > 1, 使用前一次迭代的值..
%  a0    = 标量, 目标函数的线性项 a_0*z.
%  a     = 列向量, 形状 (m, 1), 约束的线性项 a_i*z.
%  c     = 列向量, 形状 (m, 1), 项 c_i*y_i 中的常数 c_i 的列向量.
%  d     = 项 0.5*d_i*(y_i)^2 中的常数 d_i 的列向量.
%     
%*** OUTPUT:
%
%  xmma  = 当前 MMA 子问题中变量 x_j 的最优值的列向量.
%  ymma  = 当前 MMA 子问题中变量 y_i 的最优值的列向量.
%  zmma  = 当前 MMA 子问题中变量 z 的最优值的标量.
%  lam   = m 个一般 MMA 约束的拉格朗日乘子.
%  xsi   =  n 个约束 alpha_j - x_j <= 0 的拉格朗日乘子.
%  eta   =  n 个约束 x_j - beta_j <= 0 的拉格朗日乘子.
%   mu   = m 个约束 -y_i <= 0 的拉格朗日乘子.
%  zet   = 单个约束 -z <= 0 的拉格朗日乘子.
%   s    = Slack variables for the m general MMA constraints.
%  low   = 当前 MMA 子问题中计算和使用的下渐近线的列向量.
%  upp   = 当前 MMA 子问题中计算和使用的上渐近线的列向量.
%
% epsimin = sqrt(m+n)*10^(-9);

% 常数
epsimin = 10^(-7);
% raa0 = 0.00001;
raa0 = 10^(-5);
% raa0 = 0.01;
% move = 1.0;
% albefa = 0.4;
albefa = 0.1;
% asyinit = 0.1;
asyinit = 0.5;
asyincr = 1.2;
% asyincr = 0.8;
asydecr = 0.7;
eeen = ones(n, 1);
eeem = ones(m, 1);
zeron = zeros(n, 1);
move = 0.2;

% Calculation of the asymptotes low and upp :
if iter < 2.5
	low = xval - asyinit*(xmax - xmin);
	upp = xval + asyinit*(xmax - xmin);
else
	zzz = (xval-xold1).*(xold1-xold2);
	factor = eeen;
    epsilon = 1e-12;
    factor(find(zzz > epsilon)) = asyincr;
	factor(find(zzz < -epsilon)) = asydecr;
	% factor(find(zzz > 0)) = asyincr;
	% factor(find(zzz < 0)) = asydecr;
    % small_vals = zzz(abs(zzz) < 1e-10);
    % fprintf('Near zero values count: %d\n', length(small_vals));
    % fprintf('Min absolute non-zero value: %e\n', min(abs(zzz(zzz ~= 0))));
	low = xval - factor.*(xold1 - low);
	upp = xval + factor.*(upp - xold1);
    % fprintf('xval: %f\n', mean(xval(:)));
    % fprintf('factor: %f\n', mean(factor(:)));
    % fprintf('xold1: %f\n', mean(xold1(:)));
    % fprintf('xold2: %f\n', mean(xold2(:)));
    % fprintf('low: %f\n', mean(low(:)));
    % fprintf('upp: %f\n', mean(upp(:)));
    % fprintf('upp: %f\n', mean(upp(:)));
	lowmin = xval - 10*(xmax-xmin);
	lowmax = xval - 0.01*(xmax-xmin);
	uppmin = xval + 0.01*(xmax-xmin);
	uppmax = xval + 10*(xmax-xmin);
	low = max(low, lowmin);
	low = min(low, lowmax);
	upp = min(upp, uppmax);
	upp = max(upp, uppmin);
    % fprintf('Positive count: %d\n', sum(zzz > epsilon));
    % fprintf('Negative count: %d\n', sum(zzz < -epsilon));
    % fprintf('Zero count: %d\n', sum(abs(zzz) < epsilon));
    % fprintf('xval: %f\n', mean(xval(:)));
    % fprintf('Max diff xval-xold1: %e\n', mean(xval-xold1));
    % fprintf('Max diff xold1-xold2: %e\n', mean(xold1-xold2));
    % fprintf('zzz: %f\n', mean(zzz(:)));
    % fprintf('factor: %f\n', mean(factor(:)));
    % fprintf('low: %f\n', mean(low(:)));   
    % fprintf('upp: %f\n', mean(upp(:)));
    % fprintf('upp: %f\n', mean(upp(:)));
    % fprintf('zzz: %f\n', mean(zzz(:)));
    % fprintf('xold1: %f\n', mean(xold1(:)));
    % fprintf('xold2: %f\n', mean(xold2(:)));
    % fprintf('low: %f\n', mean(low(:)));
    % fprintf('upp: %f\n', mean(upp(:)));
    % fprintf('upp: %f\n', mean(upp(:)));
end

% Calculation of the bounds alfa and beta :
zzz1 = low + albefa*(xval-low);
zzz2 = xval - move*(xmax-xmin);
zzz  = max(zzz1,zzz2);
alfa = max(zzz,xmin);
zzz1 = upp - albefa*(upp-xval);
zzz2 = xval + move*(xmax-xmin);
zzz  = min(zzz1,zzz2);
beta = min(zzz,xmax);
% Calculations of p0, q0, P, Q and b.
xmami = xmax-xmin;
xmamieps = 0.00001*eeen;
xmami = max(xmami,xmamieps);
xmamiinv = eeen./xmami;
ux1 = upp-xval;
ux2 = ux1.*ux1;
xl1 = xval-low;
xl2 = xl1.*xl1;
uxinv = eeen./ux1;
xlinv = eeen./xl1;
%
p0 = zeron;
q0 = zeron;
p0 = max(df0dx,0);
q0 = max(-df0dx,0);
%p0(find(df0dx > 0)) = df0dx(find(df0dx > 0));
%q0(find(df0dx < 0)) = -df0dx(find(df0dx < 0));
pq0 = 0.001*(p0 + q0) + raa0*xmamiinv;
p0 = p0 + pq0;
q0 = q0 + pq0;
p0 = p0.*ux2;
q0 = q0.*xl2;
%
P = sparse(m,n);
Q = sparse(m,n);
P = max(dfdx,0);
Q = max(-dfdx,0);
%P(find(dfdx > 0)) = dfdx(find(dfdx > 0));
%Q(find(dfdx < 0)) = -dfdx(find(dfdx < 0));
PQ = 0.001*(P + Q) + raa0*eeem*xmamiinv.';
P = P + PQ;
Q = Q + PQ;
P = P * spdiags(ux2,0,n,n);
Q = Q * spdiags(xl2,0,n,n);
b = P*uxinv + Q*xlinv - fval(:) ;
%
%%% Solving the subproblem by a primal-dual Newton method
[xmma,ymma,zmma,lam,xsi,eta,mu,zet,s] = ...  
subsolve(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d);
