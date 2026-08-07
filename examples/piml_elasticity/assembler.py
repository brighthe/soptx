"""
Global Interface Assembly & Full-Scale Solver for SOPTX
=========================================================
Uses SOPTX native QuadrangleMesh & LinearElasticIntegrator.

Author: Liang He (Postdoc at DUT) & Antigravity Assistant
Date: 2026-08-06
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fealpy.mesh import QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm
from soptx.fem.integrators.linear_elastic_integrator import LinearElasticIntegrator
from soptx.materials import IsotropicLinearElasticMaterial


class GlobalAssembler:
    """
    Assembles global interface system and full-scale FEA reference system.
    """
    def __init__(self, Lx, Ly, n_sub_x, n_sub_y, n_fine_x, n_fine_y, E_base=1.0, nu=0.3):
        self.Lx = Lx
        self.Ly = Ly
        self.n_sub_x = n_sub_x
        self.n_sub_y = n_sub_y
        self.n_fine_x = n_fine_x
        self.n_fine_y = n_fine_y
        self.E_base = E_base
        self.nu = nu

        self.total_fine_x = n_sub_x * n_fine_x
        self.total_fine_y = n_sub_y * n_fine_y

        self.full_mesh = QuadrangleMesh.from_box(box=[0, Lx, 0, Ly], nx=self.total_fine_x, ny=self.total_fine_y)
        self.sspace_full = LagrangeFESpace(self.full_mesh, p=1, ctype='C')
        self.space_full = TensorFunctionSpace(self.sspace_full, shape=(-1, 2))

        self.n_full_nodes_x = self.total_fine_x + 1
        self.n_full_nodes_y = self.total_fine_y + 1
        self.total_full_nodes = self.full_mesh.number_of_nodes()
        self.total_full_dofs = self.space_full.number_of_global_dofs()

    def get_substructure_global_dofs(self, sx, sy, sub_mesh):
        """Map local substructure DOFs to global full-mesh DOFs."""
        sub_global_dofs = []
        for ix in range(sub_mesh.n_nodes_x):
            for iy in range(sub_mesh.n_nodes_y):
                gx = sx * self.n_fine_x + ix
                gy = sy * self.n_fine_y + iy
                gnode = gx * self.n_full_nodes_y + gy
                sub_global_dofs.extend([2 * gnode, 2 * gnode + 1])
        return np.array(sub_global_dofs)

    def solve_fullscale_fea(self, densities, load_dof, load_val=-1.0):
        """
        Assemble & solve full fine-scale FEA direct reference solution using SOPTX LinearElasticIntegrator.
        """
        full_density_grid = np.zeros((self.total_fine_x, self.total_fine_y))
        for sx in range(self.n_sub_x):
            for sy in range(self.n_sub_y):
                sub_idx = sx * self.n_sub_y + sy
                rho_field = densities[sub_idx]
                x_start = sx * self.n_fine_x
                y_start = sy * self.n_fine_y
                full_density_grid[x_start:x_start + self.n_fine_x, y_start:y_start + self.n_fine_y] = rho_field

        simp_coef = np.asarray(full_density_grid.flatten()**3.0, dtype=np.float64)

        material = IsotropicLinearElasticMaterial(youngs_modulus=self.E_base, poisson_ratio=self.nu, hypothesis='plane_stress')
        integrator = LinearElasticIntegrator(material=material)
        integrator.coef = simp_coef

        bform = BilinearForm(self.space_full)
        bform.add_integrator(integrator)

        K_tensor = bform.assembly()
        if hasattr(K_tensor, 'to_scipy'):
            K_full = K_tensor.to_scipy()
        elif hasattr(K_tensor, 'toarray'):
            K_full = sp.csr_matrix(K_tensor.toarray())
        else:
            K_full = sp.csr_matrix(np.asarray(K_tensor))

        # Fixed BC on left boundary (x = 0)
        node_coords = self.full_mesh.entity('node')
        eps = 1e-7
        left_nodes = [idx for idx, pt in enumerate(node_coords) if abs(pt[0]) < eps]
        left_dofs = np.sort(np.hstack([[2 * n, 2 * n + 1] for n in left_nodes]))

        F_full = np.zeros(self.total_full_dofs)
        F_full[load_dof] = load_val

        free_dofs = np.setdiff1d(np.arange(self.total_full_dofs), left_dofs)
        K_free = K_full[np.ix_(free_dofs, free_dofs)]
        F_free = F_full[free_dofs]

        U_free = spla.spsolve(K_free, F_free)

        U_full_ref = np.zeros(self.total_full_dofs)
        U_full_ref[free_dofs] = U_free

        return U_full_ref, free_dofs
