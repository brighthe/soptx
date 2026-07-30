import unittest

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.fem import (
    BilinearForm,
    LinearElasticityIntegrator as FealpyLinearElasticityIntegrator,
)
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import TetrahedronMesh

from soptx.fem.integrators import (
    LinearElasticIntegrator as SoptxLinearElasticIntegrator,
)
from soptx.materials import (
    IsotropicLinearElasticMaterial,
)


class TestLinearElasticIntegratorEquivalence(unittest.TestCase):
    RTOL = 1.0e-12
    ATOL = 1.0e-12
    RELATIVE_ERROR_TOLERANCE = 1.0e-12
    CELL_MATRIX_SHAPE = (6, 12, 12)
    GLOBAL_MATRIX_SHAPE = (24, 24)

    @classmethod
    def setUpClass(cls) -> None:
        bm.set_backend("numpy")
        cls.mesh = TetrahedronMesh.from_box(
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            nx=1,
            ny=1,
            nz=1,
        )
        cls.material = IsotropicLinearElasticMaterial(
            lame_lambda=1.0,
            shear_modulus=1.0,
            hypothesis="3D",
        )
        cls.number_of_cells = cls.mesh.number_of_cells()

    def _make_tensor_space(
        self,
        shape: tuple[int, int],
    ) -> TensorFunctionSpace:
        scalar_space = LagrangeFESpace(self.mesh, p=1)
        return TensorFunctionSpace(scalar_space, shape=shape)

    def _make_fealpy_integrator(
        self,
        method: str,
    ) -> FealpyLinearElasticityIntegrator:
        integrator = FealpyLinearElasticityIntegrator(
            material=self.material,
            q=4,
        )
        if method != "standard":
            integrator.assembly.set(method)
        return integrator

    def _make_soptx_integrator(
        self,
        method: str,
        *,
        coef=None,
    ) -> SoptxLinearElasticIntegrator:
        integrator = SoptxLinearElasticIntegrator(
            material=self.material,
            coef=coef,
            q=4,
            method=method,
        )
        selected_method = integrator.assembly.vm.get_key(integrator)
        self.assertEqual(
            selected_method,
            method,
            msg=(
                "SOPTX constructor did not select the requested assembly "
                f"variant: expected {method!r}, got {selected_method!r}"
            ),
        )
        return integrator

    def _assemble_cell_and_global(
        self,
        integrator,
        space: TensorFunctionSpace,
    ) -> tuple[np.ndarray, np.ndarray]:
        cell_matrix = np.asarray(
            bm.to_numpy(integrator.assembly(space)),
            dtype=np.float64,
        )

        form = BilinearForm(space)
        form.add_integrator(integrator)
        global_matrix = form.assembly(format="csr").to_scipy().toarray()
        global_matrix = np.asarray(global_matrix, dtype=np.float64)
        return cell_matrix, global_matrix

    @staticmethod
    def _relative_frobenius_error(
        actual: np.ndarray,
        expected: np.ndarray,
    ) -> float:
        denominator = max(
            float(np.linalg.norm(expected)),
            np.finfo(np.float64).eps,
        )
        return float(np.linalg.norm(actual - expected) / denominator)

    def _assert_equivalent(
        self,
        label: str,
        actual: np.ndarray,
        expected: np.ndarray,
        *,
        expected_shape: tuple[int, ...],
    ) -> None:
        self.assertEqual(actual.shape, expected_shape)
        self.assertEqual(expected.shape, expected_shape)
        self.assertTrue(
            np.all(np.isfinite(actual)),
            msg=f"{label}: actual matrix contains NaN or inf",
        )
        self.assertTrue(
            np.all(np.isfinite(expected)),
            msg=f"{label}: reference matrix contains NaN or inf",
        )

        relative_error = self._relative_frobenius_error(
            actual,
            expected,
        )
        maximum_absolute_error = float(
            np.max(np.abs(actual - expected))
        )
        print(
            f"{label}: relative_frobenius_error={relative_error:.16e}, "
            f"max_absolute_error={maximum_absolute_error:.16e}"
        )

        self.assertLessEqual(
            relative_error,
            self.RELATIVE_ERROR_TOLERANCE,
            msg=f"{label}: relative Frobenius error is too large",
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=self.RTOL,
            atol=self.ATOL,
            err_msg=label,
        )

    def _assert_symmetric(
        self,
        label: str,
        matrix: np.ndarray,
    ) -> None:
        transpose = np.swapaxes(matrix, -1, -2)
        relative_error = self._relative_frobenius_error(
            matrix,
            transpose,
        )
        print(
            f"{label}: symmetry_relative_error="
            f"{relative_error:.16e}"
        )
        self.assertLessEqual(
            relative_error,
            self.RELATIVE_ERROR_TOLERANCE,
            msg=f"{label}: matrix is not symmetric",
        )
        np.testing.assert_allclose(
            matrix,
            transpose,
            rtol=self.RTOL,
            atol=self.ATOL,
            err_msg=f"{label}: matrix is not symmetric",
        )

    def test_standard_and_voigt_match_fealpy(self) -> None:
        for shape in ((-1, 3), (3, -1)):
            with self.subTest(shape=shape):
                space = self._make_tensor_space(shape)
                fealpy_standard = self._assemble_cell_and_global(
                    self._make_fealpy_integrator("standard"),
                    space,
                )
                soptx_standard = self._assemble_cell_and_global(
                    self._make_soptx_integrator("standard"),
                    space,
                )
                fealpy_voigt = self._assemble_cell_and_global(
                    self._make_fealpy_integrator("voigt"),
                    space,
                )
                soptx_voigt = self._assemble_cell_and_global(
                    self._make_soptx_integrator("voigt"),
                    space,
                )

                comparisons = (
                    (
                        "SOPTX standard vs FEALPy standard",
                        soptx_standard,
                        fealpy_standard,
                    ),
                    (
                        "SOPTX voigt vs FEALPy voigt",
                        soptx_voigt,
                        fealpy_voigt,
                    ),
                    (
                        "FEALPy standard vs FEALPy voigt",
                        fealpy_standard,
                        fealpy_voigt,
                    ),
                    (
                        "SOPTX standard vs SOPTX voigt",
                        soptx_standard,
                        soptx_voigt,
                    ),
                )

                for label, actual, expected in comparisons:
                    cell_label = f"shape={shape}, {label}, cell"
                    global_label = f"shape={shape}, {label}, global"
                    self._assert_equivalent(
                        cell_label,
                        actual[0],
                        expected[0],
                        expected_shape=self.CELL_MATRIX_SHAPE,
                    )
                    self._assert_equivalent(
                        global_label,
                        actual[1],
                        expected[1],
                        expected_shape=self.GLOBAL_MATRIX_SHAPE,
                    )

                matrices = (
                    ("FEALPy standard cell", fealpy_standard[0]),
                    ("FEALPy standard global", fealpy_standard[1]),
                    ("SOPTX standard cell", soptx_standard[0]),
                    ("SOPTX standard global", soptx_standard[1]),
                    ("FEALPy voigt cell", fealpy_voigt[0]),
                    ("FEALPy voigt global", fealpy_voigt[1]),
                    ("SOPTX voigt cell", soptx_voigt[0]),
                    ("SOPTX voigt global", soptx_voigt[1]),
                )
                for label, matrix in matrices:
                    self._assert_symmetric(
                        f"shape={shape}, {label}",
                        matrix,
                    )

    def test_density_coefficient_scaling(self) -> None:
        coefficient_values = np.linspace(
            0.25,
            1.0,
            self.number_of_cells,
            dtype=np.float64,
        )
        unit_coefficient = bm.ones(
            (self.number_of_cells,),
            dtype=bm.float64,
        )
        varying_coefficient = bm.array(
            coefficient_values,
            dtype=bm.float64,
        )

        for shape in ((-1, 3), (3, -1)):
            for method in ("standard", "voigt"):
                with self.subTest(shape=shape, method=method):
                    space = self._make_tensor_space(shape)
                    baseline = self._assemble_cell_and_global(
                        self._make_soptx_integrator(method),
                        space,
                    )
                    with_unit_coefficient = (
                        self._assemble_cell_and_global(
                            self._make_soptx_integrator(
                                method,
                                coef=unit_coefficient,
                            ),
                            space,
                        )
                    )
                    with_varying_coefficient = (
                        self._assemble_cell_and_global(
                            self._make_soptx_integrator(
                                method,
                                coef=varying_coefficient,
                            ),
                            space,
                        )
                    )

                    self._assert_equivalent(
                        (
                            f"shape={shape}, method={method}, "
                            "coef=ones vs coef=None, cell"
                        ),
                        with_unit_coefficient[0],
                        baseline[0],
                        expected_shape=self.CELL_MATRIX_SHAPE,
                    )
                    self._assert_equivalent(
                        (
                            f"shape={shape}, method={method}, "
                            "coef=ones vs coef=None, global"
                        ),
                        with_unit_coefficient[1],
                        baseline[1],
                        expected_shape=self.GLOBAL_MATRIX_SHAPE,
                    )

                    expected_scaled_cells = (
                        coefficient_values[:, None, None] * baseline[0]
                    )
                    self._assert_equivalent(
                        (
                            f"shape={shape}, method={method}, "
                            "nonuniform cell coefficient scaling"
                        ),
                        with_varying_coefficient[0],
                        expected_scaled_cells,
                        expected_shape=self.CELL_MATRIX_SHAPE,
                    )
                    self._assert_symmetric(
                        (
                            f"shape={shape}, method={method}, "
                            "nonuniform coefficient cell"
                        ),
                        with_varying_coefficient[0],
                    )
                    self._assert_symmetric(
                        (
                            f"shape={shape}, method={method}, "
                            "nonuniform coefficient global"
                        ),
                        with_varying_coefficient[1],
                    )


if __name__ == "__main__":
    unittest.main()
