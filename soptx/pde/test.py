from sympy import symbols, sin, cos, Matrix, lambdify
from sympy import eye, tensorcontraction

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian

class LinearElasticPDE():
    def __init__(self, u, lambda0, lambda1):
        x, y = symbols('x y')
        self.u = u
        self.lambda0 = lambda0
        self.lambda1 = lambda1

        grad_u = Matrix([[0, 0], [0, 0]])
        grad_u[0, 0] = u[0].diff(x)
        grad_u[0, 1] = u[0].diff(y)
        grad_u[1, 0] = u[1].diff(x)
        grad_u[1, 1] = u[1].diff(y)

        epsilon = (grad_u + grad_u.T) / 2    

        trepsilon = tensorcontraction(epsilon, (0, 1))

        c0 = 1/lambda0
        c1 = lambda1/(lambda0 - 2*lambda1)
        sigma = c0*(c1*trepsilon*eye(2) + epsilon) 

        f = [-sigma[0, 0].diff(x) - sigma[0, 1].diff(y), 
             -sigma[1, 0].diff(x) - sigma[1, 1].diff(y)]

        self.sigmaxx = lambdify((x, y), sigma[0, 0], 'numpy')
        self.sigmayy = lambdify((x, y), sigma[1, 1], 'numpy')
        self.sigmaxy = lambdify((x, y), sigma[0, 1], 'numpy')

        self.fx = lambdify((x, y), f[0], 'numpy')
        self.fy = lambdify((x, y), f[1], 'numpy')

        self.ux = lambdify((x, y), u[0], 'numpy')
        self.uy = lambdify((x, y), u[1], 'numpy')

    def stress(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sigma = bm.zeros(p.shape[:-1] + (3, ), dtype=bm.float64)
        sigma[..., 0] = self.sigmaxx(x, y)
        sigma[..., 1] = self.sigmaxy(x, y)
        sigma[..., 2] = self.sigmayy(x, y)
        return sigma

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        f = bm.zeros(p.shape[:-1] + (2, ), dtype=bm.float64)
        f[..., 0] = self.fx(x, y)
        f[..., 1] = self.fy(x, y)
        return f

    def displacement(self, p):
        x = p[..., 0]
        y = p[..., 1]
        u = bm.zeros(p.shape[:-1] + (2, ), dtype=bm.float64)
        u[..., 0] = self.ux(x, y)
        u[..., 1] = self.uy(x, y)
        return u

    def boundary_displacement(self, p):
        return self.displacement(p) 

    def boundary_stress(self, p, n):
        sigma = self.stress(p)
        bs = bm.zeros(p.shape, dtype=bm.float64)
        bs[..., 0] = sigma[..., 0]*n[..., 0] + sigma[..., 1]*n[..., 1]
        bs[..., 1] = sigma[..., 2]*n[..., 0] + sigma[..., 1]*n[..., 1]
        return bs