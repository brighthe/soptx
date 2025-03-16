%% 参数定义
E0 = 1;
Emin = 1e-4;
nu = 0.3;
nvol = 100;
dt = 0.1;
d = -0.02;
p = 4;
phi = ones((nely+1)*(nelx+1), 1);
str = ones(nely, nelx);
volInit = sum(str(:)) / (nelx * nely);
