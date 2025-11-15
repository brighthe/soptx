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
                enable_logging: bool = False,
                logger_name: Optional[str] = None,
            ):
        super().__init__(
                    density=density, 
                    device=device, 
                    enable_logging=enable_logging, 
                    logger_name=logger_name
                )

        self._plane_type = plane_type

        self._log_info("Initializing isotropic linear elastic material")
        
        self._compute_elastic_constants(
                                youngs_modulus=youngs_modulus, 
                                poisson_ratio=poisson_ratio, 
                                lame_lambda=lame_lambda, 
                                shear_modulus=shear_modulus,
                            )
        
        self._compute_elastic_matrix()

        self._log_info("Isotropic linear elastic material initialized successfully")


    #########################################################################################
    # 属性访问器
    #########################################################################################

    @property
    def youngs_modulus(self) -> float:
        """杨氏模量"""
        return self._youngs_modulus
    
    @property
    def poisson_ratio(self) -> float:
        """泊松比"""
        return self._poisson_ratio
    
    @property
    def lame_lambda(self) -> float:
        """拉梅常数 λ"""
        return self._lame_lambda
    
    @property
    def shear_modulus(self) -> float:
        """剪切模量 μ"""
        return self._shear_modulus
    
    @property
    def bulk_modulus(self) -> float:
        """体积模量 K"""
        return self._bulk_modulus
    
    @property
    def plane_type(self) -> str:
        """平面类型"""
        return self._plane_type
    

    #########################################################################################
    # 内部方法
    #########################################################################################
        
    def _compute_elastic_constants(self, 
                            youngs_modulus: Optional[float] = None, 
                            poisson_ratio: Optional[float] = None, 
                            lame_lambda: Optional[float] = None, 
                            shear_modulus: Optional[float] = None,
                        ) -> None:
        E, nu = youngs_modulus, poisson_ratio
        lam, mu = lame_lambda, shear_modulus

        provided_params = sum(p is not None for p in [E, nu, lam, mu])
        if provided_params != 2:
            error_msg = (f"For Isotropic material, please provide exactly two "
                        f"independent elastic constants. You provided {provided_params}.")
            self._log_error(error_msg)

        # ---- Case 1: (E, nu) ----
        if E is not None and nu is not None:
            self._youngs_modulus = E
            self._poisson_ratio = nu

            self._shear_modulus = E / (2 * (1 + nu))

            if self._plane_type in ["3d", "plane_strain"]:
                # 谨防锁死
                if nu >= 0.5 - 1e-6:
                    error_msg = (
                        "Nearly incompressible material (nu ≈ 0.5) under "
                        f"{self._plane_type} is not supported by pure displacement FEM. "
                        "Please use a mixed formulation (e.g. Hu–Zhang, u–p) instead."
                    )
                    self._log_error(error_msg)

                self._lame_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
                self._bulk_modulus = E / (3 * (1 - 2 * nu))

            elif self._plane_type == "plane_stress":
                self._lame_lambda = None
                self._bulk_modulus = None

            else:
                self._log_error(f"Unknown plane_type: {self._plane_type}")

        # ---- Case 2: (λ, μ) ----
        elif lam is not None and mu is not None:
            self._lame_lambda = lam
            self._shear_modulus = mu

            self._youngs_modulus = mu * (3 * lam + 2 * mu) / (lam + mu)
            self._poisson_ratio = lam / (2 * (lam + mu))
            self._bulk_modulus = (3 * lam + 2 * mu) / 3

        else:
            error_msg = ("Unsupported combination of parameters. "
                        "Please provide (E, nu) or (λ, μ).")
            self._log_error(error_msg)
        
    def _compute_elastic_matrix(self):
        E = self._youngs_modulus
        nu = self._poisson_ratio
        lam = self._lame_lambda
        mu = self._shear_modulus

        if self._plane_type == "3d":
            self.D = bm.tensor([[2*mu+lam, lam,      lam,      0,  0,  0],
                                [lam,      2*mu+lam, lam,      0,  0,  0],
                                [lam,      lam,      2*mu+lam, 0,  0,  0],
                                [0,        0,        0,        mu, 0,  0],
                                [0,        0,        0,        0,  mu, 0],
                                [0,        0,        0,        0,  0,  mu]],
                            dtype=bm.float64, device=self.device)
            
        elif self._plane_type == "plane_stress":
            self.D = E / (1 - nu ** 2) * \
                    bm.array([[1,  nu, 0],
                              [nu, 1,  0],
                              [0,  0,  (1-nu)/2]],    
                            dtype=bm.float64, device=self.device)
        
        elif self._plane_type == "plane_strain":
            self.D = bm.tensor([[2*mu+lam, lam,      0],
                                [lam,      2*mu+lam, 0],
                                [0,        0,        mu]], 
                                dtype=bm.float64, device=self.device)
        
        else:
            error_msg = "Only '3d', 'plane_stress', and 'plane_strain' are supported."
            self._log_error(error_msg)
        
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