from typing import Optional
from abc import ABC, abstractmethod

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

class LinearElasticMaterial(ABC):
    def __init__(self, 
                density: Optional[float] = None,
                device: Optional[str] = None):
        self.density = density
        self.device = device

    @abstractmethod
    def elastic_matrix(self) -> TensorLike:
        pass

class IsotropicLinearElasticMaterial(LinearElasticMaterial):
    def __init__(self,
                youngs_modulus: Optional[float] = None, 
                poisson_ratio: Optional[float] = None, 
                lame_lambda: Optional[float] = None, 
                shear_modulus: Optional[float] = None,
                density: Optional[float] = None,
                plane_type: str = 'plane_stress',
                device: Optional[str] = None,
            ):
        super().__init__(density=density, device=device)
        
        E, nu = youngs_modulus, poisson_ratio
        lam, mu = lame_lambda, shear_modulus

        provided_params = sum(p is not None for p in [E, nu, lam, mu])
        if provided_params != 2:
            raise ValueError(
                "For Isotropic material, please provide exactly two "
                f"independent elastic constants. You provided {provided_params}."
            )
        
        if E is not None and nu is not None:
            self.youngs_modulus = E
            self.poisson_ratio = nu
            self.lame_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
            self.shear_modulus = E / (2 * (1 + nu))
        elif lam is not None and mu is not None:
            self.lame_lambda = lam
            self.shear_modulus = mu
            self.youngs_modulus = mu * (3 * lam + 2 * mu) / (lam + mu)
            self.poisson_ratio = lam / (2 * (lam + mu))
        else:
            raise ValueError(
                "Unsupported combination of parameters. "
                "Please provide (E, ν) or (λ, G)."
            )
        
        E_val = self.youngs_modulus
        nu_val = self.poisson_ratio
        lam_val = self.lame_lambda
        mu_val = self.shear_modulus
        
        if plane_type == "3D":
            self.D = bm.tensor([[2 * mu_val + lam_val, lam_val, lam_val, 0, 0, 0],
                                [lam_val, 2 * mu_val + lam_val, lam_val, 0, 0, 0],
                                [lam_val, lam_val, 2 * mu_val + lam_val, 0, 0, 0],
                                [0, 0, 0, mu_val, 0, 0],
                                [0, 0, 0, 0, mu_val, 0],
                                [0, 0, 0, 0, 0, mu_val]], 
                                dtype=bm.float64, device=device)
        elif plane_type == "plane_stress":
            self.D = E_val / (1 - nu_val ** 2) * \
                    bm.array([[1, nu_val, 0],
                              [nu_val, 1, 0],
                              [0, 0, (1 - nu_val) / 2]],    
                            dtype=bm.float64, device=device)
        elif plane_type == "plane_strain":
            self.D = bm.tensor([[2 * mu_val + lam_val, lam_val, 0],
                                [lam_val, 2 * mu_val + lam_val, 0],
                                [0, 0, mu_val]], 
                                dtype=bm.float64, device=device)
        else:
            raise NotImplementedError("Only 3D, plane_stress, and plane_strain are supported.")


    def elastic_matrix(self, bcs: Optional[TensorLike] = None) -> TensorLike:
        """
        Calculate the elastic matrix D based on the defined hypothesis (3D, plane stress, or plane strain).

        Returns:
        --------
        TensorLike: The elastic matrix D.
            - For 2D problems (GD=2): (1, 1, 3, 3)
            - For 3D problems (GD=3): (1, 1, 6, 6)
        Here, the first dimension (NC) is the number of cells, and the second dimension (NQ) is the 
        number of quadrature points, both of which are set to 1 for compatibility with other finite 
        element tensor operations.
        """
        kwargs = bm.context(self.D)
        D = bm.tensor(self.D[None, None, ...], **kwargs)

        return D