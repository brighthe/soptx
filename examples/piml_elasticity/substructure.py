"""
Substructure Partitioning & Local Stiffness Assembly for SOPTX
================================================================
Uses SOPTX native QuadrangleMesh & LinearElasticIntegrator for general FEA assembly.

Author: Liang He (Postdoc at DUT) & Antigravity Assistant
Date: 2026-08-06
"""

import numpy as np
from fealpy.mesh import QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm
from soptx.fem.integrators.linear_elastic_integrator import LinearElasticIntegrator
from soptx.materials import IsotropicLinearElasticMaterial


class SubstructureMesh:
    """
    Substructure 2D rectangular sub-mesh domain using SOPTX native FEALPy 4.0 QuadrangleMesh.
    """
    def __init__(self, sub_id, x_span, y_span, n_fine_x, n_fine_y, E_base=1.0, nu=0.3):
        self.sub_id = sub_id
        self.x_span = x_span
        self.y_span = y_span
        self.n_fine_x = n_fine_x
        self.n_fine_y = n_fine_y
        self.E_base = E_base
        self.nu = nu

        box = [x_span[0], x_span[1], y_span[0], y_span[1]]
        self.mesh = QuadrangleMesh.from_box(box=box, nx=n_fine_x, ny=n_fine_y)
        self.sspace = LagrangeFESpace(self.mesh, p=1, ctype='C')
        self.space = TensorFunctionSpace(self.sspace, shape=(-1, 2))

        self.material = IsotropicLinearElasticMaterial(youngs_modulus=E_base, poisson_ratio=nu, hypothesis='plane_stress')
        self.integrator = LinearElasticIntegrator(material=self.material)

        self.n_nodes_x = n_fine_x + 1
        self.n_nodes_y = n_fine_y + 1
        self.n_total_nodes = self.mesh.number_of_nodes()
        self.n_total_dofs = self.space.number_of_global_dofs()

        node_coords = self.mesh.entity('node')
        self.internal_nodes = []
        self.boundary_nodes = []

        eps = 1e-7
        for idx, pt in enumerate(node_coords):
            x, y = pt[0], pt[1]
            if (abs(x - x_span[0]) < eps or abs(x - x_span[1]) < eps or
                abs(y - y_span[0]) < eps or abs(y - y_span[1]) < eps):
                self.boundary_nodes.append(idx)
            else:
                self.internal_nodes.append(idx)

        self.i_dofs = np.sort(np.hstack([[2 * n, 2 * n + 1] for n in self.internal_nodes])).astype(int)
        self.b_dofs = np.sort(np.hstack([[2 * n, 2 * n + 1] for n in self.boundary_nodes])).astype(int)

        self.n_i = len(self.i_dofs)
        self.n_b = len(self.b_dofs)

    def assemble_local_stiffness(self, density_field):
        """Assemble local stiffness matrix K_local using SIMP penalty and SOPTX LinearElasticIntegrator."""
        simp_coef = np.asarray(density_field.flatten()**3.0, dtype=np.float64)
        self.integrator.coef = simp_coef

        bform = BilinearForm(self.space)
        bform.add_integrator(self.integrator)

        K_tensor = bform.assembly()
        if hasattr(K_tensor, 'toarray'):
            K_local = K_tensor.toarray()
        else:
            K_local = np.asarray(K_tensor)

        return np.asarray(K_local, dtype=np.float64)
