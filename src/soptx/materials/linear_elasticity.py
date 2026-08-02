from abc import abstractmethod
from math import isclose, isfinite
from typing import Optional, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.functionspace.utils import flatten_indices
from fealpy.typing import TensorLike

from ..core import BaseLogged


class LinearElasticMaterial(BaseLogged):
    """线弹性材料模型的抽象基类."""

    def __init__(
        self,
        density: Optional[float] = None,
        device: Optional[str] = None,
        enable_logging: bool = True,
        logger_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            enable_logging=enable_logging,
            logger_name=logger_name,
        )

        if density is not None:
            density = float(density)
            if not isfinite(density) or density < 0.0:
                raise ValueError("density must be a finite non-negative value")

        self._density = density
        self._device = device

    @property
    def density(self) -> Optional[float]:
        """材料密度; 构造时已校验为有限非负值."""
        return self._density

    @property
    def device(self) -> Optional[str]:
        """本构矩阵所在的计算设备."""
        return self._device

    def strain_matrix(
        self,
        dof_priority: bool,
        gphi: TensorLike,
        shear_order: Sequence[str] = ("yz", "xz", "xy"),
        correction: Optional[str] = None,
        cm: Optional[TensorLike] = None,
        ws: Optional[TensorLike] = None,
        detJ: Optional[TensorLike] = None,
    ) -> TensorLike:
        """构造工程应变矩阵 B.

        Parameters
        ----------
        dof_priority
            张量空间展平后的排序中, 向量分量是否排在标量自由度之前。
        gphi
            标量基函数的物理梯度, 形状为 ``(NC, NQ, LDOF, GD)``。
        shear_order
            三维工程剪应变分量的排列顺序。
        correction
            ``None`` 表示标准矩阵, ``"BBar"`` 表示三维 B-bar 修正。
        cm, ws, detJ
            B-bar 修正所需的单元测度、积分权重和 Jacobi 行列式。
        """
        if gphi.ndim != 4:
            raise ValueError(
                "gphi must have shape (NC, NQ, LDOF, GD), "
                f"but got {gphi.shape}"
            )

        ldof, geometric_dimension = gphi.shape[-2:]
        if geometric_dimension not in (2, 3):
            raise ValueError(
                "strain_matrix supports only geometric dimensions 2 and 3, "
                f"but got {geometric_dimension}"
            )

        if dof_priority:
            indices = flatten_indices(
                (ldof, geometric_dimension),
                (1, 0),
            )
        else:
            indices = flatten_indices(
                (ldof, geometric_dimension),
                (0, 1),
            )

        if correction is None:
            normal_strain = self._normal_strain(gphi, indices)
        elif correction == "BBar":
            normal_strain = self._normal_strain_bbar(
                gphi,
                indices,
                cm=cm,
                ws=ws,
                detJ=detJ,
            )
        else:
            raise ValueError(
                "correction must be None or 'BBar', "
                f"but got {correction!r}"
            )

        shear_strain = self._shear_strain(
            gphi,
            indices,
            shear_order,
        )
        return bm.concat([normal_strain, shear_strain], axis=-2)

    def _normal_strain(
        self,
        gphi: TensorLike,
        indices: TensorLike,
        *,
        out: Optional[TensorLike] = None,
    ) -> TensorLike:
        """组装应变矩阵中的正应变行."""
        kwargs = bm.context(gphi)
        ldof, geometric_dimension = gphi.shape[-2:]
        new_shape = gphi.shape[:-2] + (
            geometric_dimension,
            geometric_dimension * ldof,
        )

        if out is None:
            out = bm.zeros(new_shape, **kwargs)
        elif out.shape != new_shape:
            raise ValueError(f"out.shape={out.shape} != {new_shape}")

        for i in range(geometric_dimension):
            out = bm.set_at(
                out,
                (..., i, indices[:, i]),
                gphi[..., :, i],
            )
        return out

    def _normal_strain_bbar(
        self,
        gphi: TensorLike,
        indices: TensorLike,
        *,
        cm: Optional[TensorLike],
        ws: Optional[TensorLike],
        detJ: Optional[TensorLike],
        out: Optional[TensorLike] = None,
    ) -> TensorLike:
        """组装经三维 B-bar 修正的正应变行."""
        if any(value is None for value in (cm, ws, detJ)):
            raise ValueError("BBar correction requires cm, ws and detJ")

        nc, nq, ldof, geometric_dimension = gphi.shape
        if geometric_dimension != 3:
            raise ValueError(
                "BBar correction is defined here only for 3D materials"
            )
        if cm.shape != (nc,):
            raise ValueError(f"cm must have shape {(nc,)}, got {cm.shape}")
        if ws.shape != (nq,):
            raise ValueError(f"ws must have shape {(nq,)}, got {ws.shape}")
        if detJ.shape != (nc, nq):
            raise ValueError(
                f"detJ must have shape {(nc, nq)}, got {detJ.shape}"
            )
        if bool(bm.any(cm <= 0.0)):
            raise ValueError("cell measures must be positive")

        kwargs = bm.context(gphi)
        new_shape = gphi.shape[:-2] + (
            geometric_dimension,
            geometric_dimension * ldof,
        )
        if out is None:
            out = bm.zeros(new_shape, **kwargs)
        elif out.shape != new_shape:
            raise ValueError(f"out.shape={out.shape} != {new_shape}")

        average_gradient = bm.einsum(
            "cqld,cq,q->cld",
            gphi,
            detJ,
            ws,
        ) / cm[:, None, None]
        average_volumetric_gradient = average_gradient / 3.0

        for strain_component in range(geometric_dimension):
            for displacement_component in range(geometric_dimension):
                gradient = gphi[..., :, displacement_component]
                corrected_gradient = (
                    -gradient / 3.0
                    + average_volumetric_gradient[
                        :, None, :, displacement_component
                    ]
                )
                if strain_component == displacement_component:
                    corrected_gradient = corrected_gradient + gradient

                out = bm.set_at(
                    out,
                    (
                        ...,
                        strain_component,
                        indices[:, displacement_component],
                    ),
                    corrected_gradient,
                )
        return out

    def _shear_strain(
        self,
        gphi: TensorLike,
        indices: TensorLike,
        shear_order: Sequence[str],
        *,
        out: Optional[TensorLike] = None,
    ) -> TensorLike:
        """组装应变矩阵中的工程剪应变行."""
        kwargs = bm.context(gphi)
        ldof, geometric_dimension = gphi.shape[-2:]
        number_of_shear_components = (
            geometric_dimension * (geometric_dimension - 1)
        ) // 2
        new_shape = gphi.shape[:-2] + (
            number_of_shear_components,
            geometric_dimension * ldof,
        )

        if geometric_dimension == 2:
            shear_indices = ((0, 1),)
        else:
            valid_order = {"xy", "yz", "xz"}
            if (
                len(shear_order) != 3
                or set(shear_order) != valid_order
            ):
                raise ValueError(
                    "shear_order must be a permutation of "
                    "('xy', 'yz', 'xz')"
                )
            index_map = {
                "xy": (0, 1),
                "yz": (1, 2),
                "xz": (2, 0),
            }
            shear_indices = tuple(
                index_map[pair] for pair in shear_order
            )

        if out is None:
            out = bm.zeros(new_shape, **kwargs)
        elif out.shape != new_shape:
            raise ValueError(f"out.shape={out.shape} != {new_shape}")

        for cursor, (i, j) in enumerate(shear_indices):
            out = bm.set_at(
                out,
                (..., cursor, indices[:, i]),
                gphi[..., :, j],
            )
            out = bm.set_at(
                out,
                (..., cursor, indices[:, j]),
                gphi[..., :, i],
            )
        return out

    @abstractmethod
    def elastic_matrix(
        self,
        bcs: Optional[TensorLike] = None,
    ) -> TensorLike:
        """返回积分点处的本构矩阵."""


class IsotropicLinearElasticMaterial(LinearElasticMaterial):
    """均匀各向同性线弹性材料.

    ``lame_lambda`` 和 ``shear_modulus`` 始终表示三维本征 Lamé 常数。所选的
    hypothesis 只改变构造本构矩阵时使用的降维方式, 不改变这两个常数的含义。
    """

    _VALID_HYPOTHESES = {
        "3D",
        "plane_stress",
        "plane_strain",
    }

    def __init__(
        self,
        youngs_modulus: Optional[float] = None,
        poisson_ratio: Optional[float] = None,
        lame_lambda: Optional[float] = None,
        shear_modulus: Optional[float] = None,
        hypothesis: str = "3D",
        density: Optional[float] = None,
        device: Optional[str] = None,
        enable_logging: bool = False,
        logger_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            density=density,
            device=device,
            enable_logging=enable_logging,
            logger_name=logger_name,
        )

        if hypothesis not in self._VALID_HYPOTHESES:
            choices = ", ".join(sorted(self._VALID_HYPOTHESES))
            raise ValueError(
                f"hypothesis must be one of {choices}, got {hypothesis!r}"
            )
        self._hypothesis = hypothesis

        (
            self._youngs_modulus,
            self._poisson_ratio,
            self._lame_lambda,
            self._shear_modulus,
            self._bulk_modulus,
        ) = self._resolve_elastic_constants(
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            lame_lambda=lame_lambda,
            shear_modulus=shear_modulus,
        )
        self._D = self._compute_elastic_matrix()

    @property
    def D(self) -> TensorLike:
        """本构矩阵, 构造时算好后不再变化.

        property 只防止整体重新绑定; 返回的张量本身仍可被原地修改, 调用方
        不应这样做。
        """
        return self._D

    @property
    def youngs_modulus(self) -> float:
        """杨氏模量."""
        return self._youngs_modulus

    @property
    def poisson_ratio(self) -> float:
        """泊松比."""
        return self._poisson_ratio

    @property
    def lame_lambda(self) -> float:
        """第一 Lamé 常数."""
        return self._lame_lambda

    @property
    def shear_modulus(self) -> float:
        """剪切模量, 等于第二 Lamé 常数."""
        return self._shear_modulus

    @property
    def bulk_modulus(self) -> float:
        """体积模量."""
        return self._bulk_modulus

    @property
    def hypothesis(self) -> str:
        """本构假设."""
        return self._hypothesis

    @property
    def is_incompressible(self) -> bool:
        """材料在数值上是否接近不可压缩."""
        return self._poisson_ratio >= 0.5 - 1.0e-12

    @staticmethod
    def _validate_youngs_poisson(
        youngs_modulus: float,
        poisson_ratio: float,
    ) -> tuple[float, float, float, float, float]:
        youngs_modulus = float(youngs_modulus)
        poisson_ratio = float(poisson_ratio)
        if not isfinite(youngs_modulus) or youngs_modulus <= 0.0:
            raise ValueError("youngs_modulus must be finite and positive")
        if (
            not isfinite(poisson_ratio)
            or not -1.0 < poisson_ratio < 0.5
        ):
            raise ValueError(
                "poisson_ratio must be finite and satisfy -1 < nu < 0.5"
            )

        shear_modulus = youngs_modulus / (
            2.0 * (1.0 + poisson_ratio)
        )
        lame_lambda = (
            youngs_modulus
            * poisson_ratio
            / (
                (1.0 + poisson_ratio)
                * (1.0 - 2.0 * poisson_ratio)
            )
        )
        bulk_modulus = youngs_modulus / (
            3.0 * (1.0 - 2.0 * poisson_ratio)
        )
        return (
            youngs_modulus,
            poisson_ratio,
            lame_lambda,
            shear_modulus,
            bulk_modulus,
        )

    @staticmethod
    def _validate_lame(
        lame_lambda: float,
        shear_modulus: float,
    ) -> tuple[float, float, float, float, float]:
        lame_lambda = float(lame_lambda)
        shear_modulus = float(shear_modulus)
        if not isfinite(lame_lambda):
            raise ValueError("lame_lambda must be finite")
        if not isfinite(shear_modulus) or shear_modulus <= 0.0:
            raise ValueError("shear_modulus must be finite and positive")

        bulk_modulus = lame_lambda + 2.0 * shear_modulus / 3.0
        if bulk_modulus <= 0.0:
            raise ValueError(
                "lame_lambda and shear_modulus must yield a positive "
                "bulk modulus"
            )

        youngs_modulus = (
            shear_modulus
            * (3.0 * lame_lambda + 2.0 * shear_modulus)
            / (lame_lambda + shear_modulus)
        )
        poisson_ratio = lame_lambda / (
            2.0 * (lame_lambda + shear_modulus)
        )
        return (
            youngs_modulus,
            poisson_ratio,
            lame_lambda,
            shear_modulus,
            bulk_modulus,
        )

    @classmethod
    def _resolve_elastic_constants(
        cls,
        *,
        youngs_modulus: Optional[float],
        poisson_ratio: Optional[float],
        lame_lambda: Optional[float],
        shear_modulus: Optional[float],
    ) -> tuple[float, float, float, float, float]:
        has_youngs_poisson = (
            youngs_modulus is not None
            and poisson_ratio is not None
        )
        has_lame = (
            lame_lambda is not None
            and shear_modulus is not None
        )
        has_partial_youngs_poisson = (
            (youngs_modulus is None)
            != (poisson_ratio is None)
        )
        has_partial_lame = (
            (lame_lambda is None)
            != (shear_modulus is None)
        )

        if has_partial_youngs_poisson or has_partial_lame:
            raise ValueError(
                "provide complete pairs (youngs_modulus, poisson_ratio) "
                "or (lame_lambda, shear_modulus)"
            )
        if not has_youngs_poisson and not has_lame:
            raise ValueError(
                "provide (youngs_modulus, poisson_ratio), "
                "(lame_lambda, shear_modulus), or all four constants"
            )

        from_youngs_poisson = None
        from_lame = None
        if has_youngs_poisson:
            from_youngs_poisson = cls._validate_youngs_poisson(
                youngs_modulus,
                poisson_ratio,
            )
        if has_lame:
            from_lame = cls._validate_lame(
                lame_lambda,
                shear_modulus,
            )

        if from_youngs_poisson is None:
            return from_lame
        if from_lame is None:
            return from_youngs_poisson

        labels = (
            "youngs_modulus",
            "poisson_ratio",
            "lame_lambda",
            "shear_modulus",
            "bulk_modulus",
        )
        for label, value_from_ep, value_from_lame in zip(
            labels,
            from_youngs_poisson,
            from_lame,
        ):
            if not isclose(
                value_from_ep,
                value_from_lame,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "inconsistent elastic constants: "
                    f"{label} differs between the supplied pairs"
                )
        return from_youngs_poisson

    def _compute_elastic_matrix(self) -> TensorLike:
        youngs_modulus = self._youngs_modulus
        poisson_ratio = self._poisson_ratio
        lame_lambda = self._lame_lambda
        shear_modulus = self._shear_modulus

        if self._hypothesis == "3D":
            return bm.tensor(
                [
                    [
                        2.0 * shear_modulus + lame_lambda,
                        lame_lambda,
                        lame_lambda,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        lame_lambda,
                        2.0 * shear_modulus + lame_lambda,
                        lame_lambda,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        lame_lambda,
                        lame_lambda,
                        2.0 * shear_modulus + lame_lambda,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [0.0, 0.0, 0.0, shear_modulus, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, shear_modulus, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, shear_modulus],
                ],
                dtype=bm.float64,
                device=self.device,
            )

        if self._hypothesis == "plane_stress":
            factor = youngs_modulus / (1.0 - poisson_ratio**2)
            return factor * bm.tensor(
                [
                    [1.0, poisson_ratio, 0.0],
                    [poisson_ratio, 1.0, 0.0],
                    [0.0, 0.0, (1.0 - poisson_ratio) / 2.0],
                ],
                dtype=bm.float64,
                device=self.device,
            )

        return bm.tensor(
            [
                [
                    2.0 * shear_modulus + lame_lambda,
                    lame_lambda,
                    0.0,
                ],
                [
                    lame_lambda,
                    2.0 * shear_modulus + lame_lambda,
                    0.0,
                ],
                [0.0, 0.0, shear_modulus],
            ],
            dtype=bm.float64,
            device=self.device,
        )

    def elastic_matrix(
        self,
        bcs: Optional[TensorLike] = None,
    ) -> TensorLike:
        """返回均匀材料的本构矩阵.

        返回形状为 ``(1, 1, NS, NS)``: 前两维是单元与积分点的占位维, 由调用
        方按 ``(NC, NQ, NS, NS)`` 广播。均匀材料在所有单元和积分点上的本构
        相同, 所以这里不实际展开, 避免无谓的内存占用。

        接收 ``bcs`` 是为了兼容 FEALPy 的材料协议; 对均匀材料而言本构矩阵
        与位置无关, 因此这个参数被有意忽略。
        """
        del bcs
        return self._D[None, None, ...]

    def von_mises_matrix(self) -> TensorLike:
        """返回 von Mises 应力对应的二次型矩阵."""
        if self._hypothesis == "3D":
            return bm.tensor(
                [
                    [1.0, -0.5, -0.5, 0.0, 0.0, 0.0],
                    [-0.5, 1.0, -0.5, 0.0, 0.0, 0.0],
                    [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
                ],
                dtype=bm.float64,
                device=self.device,
            )

        if self._hypothesis == "plane_stress":
            return bm.tensor(
                [
                    [1.0, -0.5, 0.0],
                    [-0.5, 1.0, 0.0],
                    [0.0, 0.0, 3.0],
                ],
                dtype=bm.float64,
                device=self.device,
            )

        poisson_ratio = self.poisson_ratio
        diagonal = 1.0 - poisson_ratio + poisson_ratio**2
        off_diagonal = poisson_ratio**2 - poisson_ratio - 0.5
        return bm.tensor(
            [
                [diagonal, off_diagonal, 0.0],
                [off_diagonal, diagonal, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=bm.float64,
            device=self.device,
        )

    def calculate_stress_vector(
        self,
        B: TensorLike,
        u_e: TensorLike,
    ) -> TensorLike:
        """计算实体材料的应力向量."""
        D = self.elastic_matrix()[0, 0]
        return bm.einsum(
            "ij,c...qjk,ck->c...qi",
            D,
            B,
            u_e,
        )

    def calculate_von_mises_stress(
        self,
        stress_vector: TensorLike,
    ) -> TensorLike:
        """计算 von Mises 等效应力."""
        matrix = self.von_mises_matrix()
        squared_stress = bm.einsum(
            "c...qi,ij,c...qj->c...q",
            stress_vector,
            matrix,
            stress_vector,
        )
        floor = bm.tensor(
            1.0e-12,
            dtype=squared_stress.dtype,
            device=bm.get_device(squared_stress),
        )
        return bm.sqrt(bm.maximum(squared_stress, floor))
