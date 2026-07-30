import unittest

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearElasticityIntegrator
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.material import LinearElasticMaterial as FealpyMaterial
from fealpy.mesh import TetrahedronMesh

from soptx.materials import (
    IsotropicLinearElasticMaterial,
)


class TestLinearElasticMaterialProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bm.set_backend("numpy")

    def assertTensorAllClose(
        self,
        actual,
        expected,
        *,
        rtol=1.0e-12,
        atol=1.0e-12,
    ) -> None:
        np.testing.assert_allclose(
            bm.to_numpy(actual),
            bm.to_numpy(expected),
            rtol=rtol,
            atol=atol,
        )

    def test_constitutive_matrices_match_fealpy_from_youngs_poisson(
        self,
    ) -> None:
        for hypothesis in ("3D", "plane_stress", "plane_strain"):
            with self.subTest(hypothesis=hypothesis):
                material = IsotropicLinearElasticMaterial(
                    youngs_modulus=210.0,
                    poisson_ratio=0.3,
                    hypothesis=hypothesis,
                )
                reference = FealpyMaterial(
                    name="reference",
                    elastic_modulus=210.0,
                    poisson_ratio=0.3,
                    hypo=hypothesis,
                )
                self.assertTensorAllClose(
                    material.elastic_matrix(),
                    reference.elastic_matrix(),
                )

    def test_lame_constants_are_intrinsic_for_plane_stress(self) -> None:
        material = IsotropicLinearElasticMaterial(
            lame_lambda=1.0,
            shear_modulus=1.0,
            hypothesis="plane_stress",
        )
        expected = bm.array(
            [
                [8.0 / 3.0, 2.0 / 3.0, 0.0],
                [2.0 / 3.0, 8.0 / 3.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=bm.float64,
        )

        self.assertAlmostEqual(material.youngs_modulus, 2.5)
        self.assertAlmostEqual(material.poisson_ratio, 0.25)
        self.assertTensorAllClose(
            material.elastic_matrix()[0, 0],
            expected,
        )

    def test_four_consistent_constants_are_accepted(self) -> None:
        material = IsotropicLinearElasticMaterial(
            youngs_modulus=2.5,
            poisson_ratio=0.25,
            lame_lambda=1.0,
            shear_modulus=1.0,
            hypothesis="3D",
        )
        self.assertAlmostEqual(material.bulk_modulus, 5.0 / 3.0)

    def test_invalid_material_parameters_are_rejected(self) -> None:
        invalid_keyword_sets = (
            {"youngs_modulus": 1.0},
            {"lame_lambda": 1.0},
            {
                "youngs_modulus": 2.5,
                "poisson_ratio": 0.25,
                "lame_lambda": 2.0,
                "shear_modulus": 1.0,
            },
            {
                "youngs_modulus": -1.0,
                "poisson_ratio": 0.3,
            },
            {
                "youngs_modulus": 1.0,
                "poisson_ratio": 0.5,
            },
            {
                "lame_lambda": -1.0,
                "shear_modulus": 1.0,
            },
        )
        for keywords in invalid_keyword_sets:
            with self.subTest(keywords=keywords):
                with self.assertRaises(ValueError):
                    IsotropicLinearElasticMaterial(**keywords)

        with self.assertRaises(ValueError):
            IsotropicLinearElasticMaterial(
                lame_lambda=1.0,
                shear_modulus=1.0,
                hypothesis="3d",
            )

    def test_new_protocol_has_no_legacy_material_interface(self) -> None:
        material = IsotropicLinearElasticMaterial(
            lame_lambda=1.0,
            shear_modulus=1.0,
        )
        self.assertFalse(
            hasattr(material, "strain_displacement_matrix")
        )
        self.assertFalse(hasattr(material, "plane_type"))
        self.assertEqual(material.hypothesis, "3D")

    def test_elastic_matrix_accepts_barycentric_coordinates(self) -> None:
        material = IsotropicLinearElasticMaterial(
            lame_lambda=1.0,
            shear_modulus=1.0,
        )
        bcs = bm.array(
            [[0.25, 0.25, 0.25, 0.25]],
            dtype=bm.float64,
        )
        constitutive_matrix = material.elastic_matrix(bcs)
        self.assertEqual(constitutive_matrix.shape, (1, 1, 6, 6))

    def test_standard_strain_matrix_matches_fealpy(self) -> None:
        rng = np.random.default_rng(20260727)
        gphi = bm.array(
            rng.standard_normal((2, 3, 4, 3)),
            dtype=bm.float64,
        )
        material = IsotropicLinearElasticMaterial(
            youngs_modulus=210.0,
            poisson_ratio=0.3,
        )
        reference = FealpyMaterial(
            name="reference",
            elastic_modulus=210.0,
            poisson_ratio=0.3,
            hypo="3D",
        )

        for dof_priority in (False, True):
            with self.subTest(dof_priority=dof_priority):
                self.assertTensorAllClose(
                    material.strain_matrix(
                        dof_priority=dof_priority,
                        gphi=gphi,
                    ),
                    reference.strain_matrix(
                        dof_priority=dof_priority,
                        gphi=gphi,
                    ),
                )

    def test_bbar_strain_matrix_matches_fealpy(self) -> None:
        rng = np.random.default_rng(20260727)
        gphi = bm.array(
            rng.standard_normal((2, 3, 4, 3)),
            dtype=bm.float64,
        )
        ws = bm.array([0.2, 0.3, 0.5], dtype=bm.float64)
        detJ = bm.array(
            [
                [1.0, 1.2, 0.9],
                [0.8, 1.1, 1.3],
            ],
            dtype=bm.float64,
        )
        cm = bm.einsum("q,cq->c", ws, detJ)
        material = IsotropicLinearElasticMaterial(
            youngs_modulus=210.0,
            poisson_ratio=0.3,
        )
        reference = FealpyMaterial(
            name="reference",
            elastic_modulus=210.0,
            poisson_ratio=0.3,
            hypo="3D",
        )

        self.assertTensorAllClose(
            material.strain_matrix(
                dof_priority=False,
                gphi=gphi,
                correction="BBar",
                cm=cm,
                ws=ws,
                detJ=detJ,
            ),
            reference.strain_matrix(
                dof_priority=False,
                gphi=gphi,
                correction="BBar",
                cm=cm,
                ws=ws,
                detJ=detJ,
            ),
        )

        with self.assertRaises(ValueError):
            material.strain_matrix(
                dof_priority=False,
                gphi=gphi[..., :2],
                correction="BBar",
                cm=cm,
                ws=ws,
                detJ=detJ,
            )

    def test_material_assembles_with_fealpy_integrator(self) -> None:
        mesh = TetrahedronMesh.from_box(
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            nx=1,
            ny=1,
            nz=1,
        )
        scalar_space = LagrangeFESpace(mesh, p=1)
        tensor_space = TensorFunctionSpace(
            scalar_space,
            shape=(-1, 3),
        )
        material = IsotropicLinearElasticMaterial(
            lame_lambda=1.0,
            shear_modulus=1.0,
        )
        integrator = LinearElasticityIntegrator(
            material=material,
            q=4,
        )

        cell_matrix = integrator.assembly(tensor_space)
        values = bm.to_numpy(cell_matrix)
        self.assertEqual(values.ndim, 3)
        self.assertTrue(np.all(np.isfinite(values)))

    def test_von_mises_stress_is_preserved(self) -> None:
        material = IsotropicLinearElasticMaterial(
            lame_lambda=1.0,
            shear_modulus=1.0,
        )
        uniaxial_stress = bm.array(
            [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            dtype=bm.float64,
        )
        equivalent_stress = material.calculate_von_mises_stress(
            uniaxial_stress
        )
        self.assertTensorAllClose(
            equivalent_stress,
            bm.ones((1, 1), dtype=bm.float64),
        )


if __name__ == "__main__":
    unittest.main()
