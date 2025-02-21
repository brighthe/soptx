from fenics import *
from dolfin_adjoint import *
import cyipopt as ipopt

E, nu = 1e5, 0.3 # Structure material properties
L, H, B = 3.0, 1.0, 0.2  # Geometry of the design domain
F = 2000         # Load (T)
p, eps = Constant(3.0), Constant(1.0e-3)      # penalisation and SIMP constants
rho_0, Vol = Constant(0.5), Constant(0.5*L*H) # Top.Opt.constants: Initial guess and Volume constraint

# Mesh constant
nx, ny, nz = 300, 100, 20
mesh = RectangleMesh(Point(0, 0, 0), Point(L, H, B), nx, ny, nz)