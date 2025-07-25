from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace.utils import flatten_indices

from ..utils.base_logged import BaseLogged

class LinearElasticMaterial(BaseLogged, ABC):
    def __init__(self, 
                density: Optional[float] = None,
                device: Optional[str] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None,
            ) -> None:
        
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)
        
        self.density = density
        self.device = device

    def strain_displacement_matrix(self, 
                    dof_priority: bool, 
                    gphi: TensorLike, 
                    shear_order: List[str]=['yz', 'xz', 'xy'],
                    # shear_order: List[str]=['xy', 'yz', 'xz'],
                    # shear_order: List[str]=['xy', 'xz', 'yz'],
                ) -> TensorLike:
        '''
        Constructs the strain-displacement matrix B for the material \n
            based on the gradient of the shape functions.
        B = [∂Ni/∂x   0       0    ]
            [0        ∂Ni/∂y  0    ]
            [0        0       ∂Ni/∂z]
            [0        ∂Ni/∂z  ∂Ni/∂y]
            [∂Ni/∂z   0       ∂Ni/∂x]
            [∂Ni/∂y   ∂Ni/∂x  0     ]

        B = [∂Ni/∂x   0       0    ]
            [0        ∂Ni/∂y  0    ]
            [0        0       ∂Ni/∂z]
            [∂Ni/∂y   ∂Ni/∂x  0    ]
            [∂Ni/∂z   0       ∂Ni/∂x]
            [0        ∂Ni/∂z  ∂Ni/∂y]

        B = [∂Ni/∂x   0       0    ]
            [0        ∂Ni/∂y  0    ]
            [0        0       ∂Ni/∂z]
            [∂Ni/∂y   ∂Ni/∂x  0    ]
            [0        ∂Ni/∂z  ∂Ni/∂y]
            [∂Ni/∂z   0       ∂Ni/∂x]

        Parameters:
        -----------
        dof_priority: A flag that determines the ordering of DOFs.
                            If True, the priority is given to the first dimension of degrees of freedom.
        gphi - (NC, NQ, LDOF, GD).
        shear_order: Specifies the order of shear strain components for GD=3.
                                        Valid options are permutations of {'xy', 'yz', 'xz'}.
        
        Returns:
        --------
        B: The strain-displacement matrix `B`, which is a tensor with shape:
            - For 2D problems (GD=2): (NC, NQ, 3, TLDOF)
            - For 3D problems (GD=3): (NC, NQ, 6, TLDOF)
        '''
        ldof, GD = gphi.shape[-2:]
        if dof_priority:
            indices = flatten_indices((ldof, GD), (1, 0))
        else:
            indices = flatten_indices((ldof, GD), (0, 1))
            
        normal_B = self._normal_strain(gphi, indices)
        shear_B = self._shear_strain(gphi, indices, shear_order)

        B = bm.concat([normal_B, shear_B], axis=-2)

        return B
    
    def _normal_strain(self,
                    gphi: TensorLike, 
                    indices: TensorLike, *, 
                    out: Optional[TensorLike]=None
                ) -> TensorLike:
        """Assembly normal strain tensor.

        Parameters:
        -----------
        gphi - (NC, NQ, LDOF, GD).
        indices - (LDOF, GD): Indices of DoF components in the flattened DoF, shaped .
        out - (TensorLike | None, optional): Output tensor. Defaults to None.

        Returns:
        --------
        out - Normal strain shaped (NC, NQ, GD, GD*LDOF): 
        """
        kwargs = bm.context(gphi)
        ldof, GD = gphi.shape[-2:]
        new_shape = gphi.shape[:-2] + (GD, GD*ldof) # (NC, NQ, GD, GD*LDOF)

        if out is None:
            out = bm.zeros(new_shape, **kwargs)
        else:
            if out.shape != new_shape:
                raise ValueError(f'out.shape={out.shape} != {new_shape}')

        for i in range(GD):
            out = bm.set_at(out, (..., i, indices[:, i]), gphi[..., :, i])

        return out

    def _shear_strain(self, 
                    gphi: TensorLike, 
                    indices: TensorLike, 
                    shear_order: List[str], *,
                    out: Optional[TensorLike]=None
                ) -> TensorLike:
        """Assembly shear strain tensor.

        Parameters:
        -----------
        gphi - (NC, NQ, LDOF, GD).
        indices (bool, optional): Indices of DoF components in the flattened DoF, shaped (LDOF, GD).
        shear_order: Specifies the order of shear strain components for GD=3.
                                        Valid options are permutations of {'xy', 'yz', 'xz'}.
        Returns:
        --------
        out - Shear strain shaped (NC, NQ, NNZ, GD*LDOF) where NNZ = (GD + (GD+1))//2: .
        """
        kwargs = bm.context(gphi)
        ldof, GD = gphi.shape[-2:]
        if GD < 2:
            raise ValueError(f"The shear strain requires GD >= 2, but GD = {GD}")
        NNZ = (GD * (GD-1))//2    # 剪切应变分量的数量
        new_shape = gphi.shape[:-2] + (NNZ, GD*ldof) # (NC, NQ, NNZ, GD*LDOF)

        if GD == 2:
            shear_indices = [(0, 1)]  # Corresponds to 'xy'
        elif GD == 3:
            valid_pairs = {'xy', 'yz', 'xz'}
            if not set(shear_order).issubset(valid_pairs):
                raise ValueError(f"Invalid shear_order: {shear_order}. Valid options are {valid_pairs}")

            index_map = {
                'xy': (0, 1),
                'yz': (1, 2),
                'xz': (2, 0),
            }
            shear_indices = [index_map[pair] for pair in shear_order]
        else:
            raise ValueError(f"GD={GD} is not supported")

        if out is None:
            out = bm.zeros(new_shape, **kwargs)
        else:
            if out.shape != new_shape:
                raise ValueError(f'out.shape={out.shape} != {new_shape}')

        for cursor, (i, j) in enumerate(shear_indices):
            out = bm.set_at(out, (..., cursor, indices[:, i]), gphi[..., :, j])
            out = bm.set_at(out, (..., cursor, indices[:, j]), gphi[..., :, i])

        return out
    
    @abstractmethod
    def elastic_matrix(self) -> TensorLike:
        """计算弹性矩阵"""
        pass

class IsotropicLinearElasticMaterial(LinearElasticMaterial):
    def __init__(self,
                youngs_modulus: Optional[float] = None, 
                poisson_ratio: Optional[float] = None, 
                lame_lambda: Optional[float] = None, 
                shear_modulus: Optional[float] = None,
                plane_type: str = 'plane_stress',
                density: Optional[float] = None,
                device: Optional[str] = None,
                enable_logging: bool = True,
                logger_name: Optional[str] = None,
            ):
        super().__init__(
                    density=density, 
                    device=device, 
                    enable_logging=enable_logging, 
                    logger_name=logger_name
                )

        self.plane_type = plane_type

        self._log_info("Initializing isotropic linear elastic material")
        
        self._compute_elastic_constants(
                                youngs_modulus=youngs_modulus, 
                                poisson_ratio=poisson_ratio, 
                                lame_lambda=lame_lambda, 
                                shear_modulus=shear_modulus,
                            )
        
        self._compute_elastic_matrix()

        self._log_info("Isotropic linear elastic material initialized successfully")
        
    def _compute_elastic_constants(self, 
                            youngs_modulus: Optional[float] = None, 
                            poisson_ratio: Optional[float] = None, 
                            lame_lambda: Optional[float] = None, 
                            shear_modulus: Optional[float] = None,
                            bulk_modulus: Optional[float] = None,
                        ) -> None:
        E, nu = youngs_modulus, poisson_ratio
        lam, mu = lame_lambda, shear_modulus

        provided_params = sum(p is not None for p in [E, nu, lam, mu])
        if provided_params != 2:
            error_msg = (f"For Isotropic material, please provide exactly two "
                        f"independent elastic constants. You provided {provided_params}.")
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        if E is not None and nu is not None:
            self.youngs_modulus = E
            self.poisson_ratio = nu
            self.lame_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
            self.shear_modulus = E / (2 * (1 + nu))
            self.bulk_modulus = E / (3 * (1 - 2 * nu))
            
            self._log_info(f"Set elastic constants from E={E:.2e}, ν={nu:.3f}")

        elif lam is not None and mu is not None:
            self.lame_lambda = lam
            self.shear_modulus = mu
            self.youngs_modulus = mu * (3 * lam + 2 * mu) / (lam + mu)
            self.poisson_ratio = lam / (2 * (lam + mu))
            self.bulk_modulus = (3 * lam + 2 * mu) / 3

            
            self._log_info(f"Set elastic constants from λ={lam:.2e}, G={mu:.2e}")

        else:
            error_msg = ("Unsupported combination of parameters. "
                        "Please provide (E, ν) or (λ, G).")
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
    def _compute_elastic_matrix(self):
        E_val = self.youngs_modulus
        nu_val = self.poisson_ratio
        lam_val = self.lame_lambda
        mu_val = self.shear_modulus

        if self.plane_type == "3d":
            self.D = bm.tensor([[2 * mu_val + lam_val, lam_val, lam_val, 0, 0, 0],
                                [lam_val, 2 * mu_val + lam_val, lam_val, 0, 0, 0],
                                [lam_val, lam_val, 2 * mu_val + lam_val, 0, 0, 0],
                                [0, 0, 0, mu_val, 0, 0],
                                [0, 0, 0, 0, mu_val, 0],
                                [0, 0, 0, 0, 0, mu_val]], 
                                dtype=bm.float64, device=self.device)
        elif self.plane_type == "plane_stress":
            self.D = E_val / (1 - nu_val ** 2) * \
                    bm.array([[1, nu_val, 0],
                              [nu_val, 1, 0],
                              [0, 0, (1 - nu_val) / 2]],    
                            dtype=bm.float64, device=self.device)
        elif self.plane_type == "plane_strain":
            self.D = bm.tensor([[2 * mu_val + lam_val, lam_val, 0],
                                [lam_val, 2 * mu_val + lam_val, 0],
                                [0, 0, mu_val]], 
                                dtype=bm.float64, device=self.device)
        else:
            error_msg = "Only 3d, plane_stress, and plane_strain are supported."
            self._log_error(error_msg)
            raise NotImplementedError(error_msg)
        
        self._log_info(f"Elastic matrix computed for {self.plane_type}, shape: {self.D.shape}")

    def elastic_matrix(self) -> TensorLike:
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
        if not hasattr(self, 'D') or self.D is None:
            self._log_warning("Elastic matrix not computed, computing now...")
            self._compute_elastic_matrix()
        
        kwargs = bm.context(self.D)
        D = bm.tensor(self.D[None, None, ...], **kwargs)

        self._log_info(f"[IsotropicLinearElasticMaterial] Elastic matrix computed successfully, "
                   f"shape: {D.shape}")

        return D

    def get_material_params(self) -> Dict[str, Any]:

        params = {
            'youngs_modulus': self.youngs_modulus,
            'poisson_ratio': self.poisson_ratio,
            'lame_lambda': self.lame_lambda,
            'shear_modulus': self.shear_modulus,
            'bulk_modulus': self.bulk_modulus,
            'density': self.density,
        }

        return params

    def display_material_params(self) -> None:

        params = self.get_material_params()
        self._log_info(f"Material parameters: {params}", force_log=True)

    def set_material_parameters(self, 
                            youngs_modulus: Optional[float] = None, 
                            poisson_ratio: Optional[float] = None, 
                            lame_lambda: Optional[float] = None, 
                            shear_modulus: Optional[float] = None,
                            bulk_modulus: Optional[float] = None,
                            density: Optional[float] = None
                        ) -> None:
        self._log_info("Updating material parameters")

        self._compute_elastic_constants(youngs_modulus, poisson_ratio, lame_lambda, shear_modulus, bulk_modulus)

        if density is not None:
            self.density = density
        
        self._compute_elastic_matrix()

        self._log_info(f"[IsotropicLinearElasticMaterial] Material parameters updated successfully, "
                       f"elastic_matrix recalculated")

    def set_plane_type(self, plane_type: str) -> None:
        if plane_type not in ['3d', 'plane_stress', 'plane_strain']:
            error_msg = "Invalid plane type. Choose from '3d', 'plane_stress', or 'plane_strain'."
            self._log_error(error_msg)
            raise ValueError(error_msg)
        
        self.plane_type = plane_type
        self._compute_elastic_matrix()

        self._log_info(f"Plane type set to {self.plane_type}. elastic_matrix updated.")