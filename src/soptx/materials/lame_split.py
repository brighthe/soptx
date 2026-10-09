"""各向同性本构矩阵按 Lamé 参数的线性分解.

各向同性本构矩阵 (Voigt 记号, 剪切取工程应变) 总可写成

    D = λ* D_λ + μ D_μ,

D_λ、D_μ 为只依赖维数的常数矩阵, 其中

    平面应变 / 3D : λ* = λ = E ν / ((1+ν)(1-2ν))
    平面应力      : λ* = λ̄ = E ν / (1-ν²) = 2λμ / (λ+2μ)
    μ = E / (2(1+ν))

E、ν 逐单元变化 (如泊松比随密度插值) 时, 单元刚度 K_e = λ*_e K_e^λ + μ_e K_e^μ 的两个
基矩阵与设计无关; 对设计变量求导只需对 λ*、μ 做链式法则, 不必重新积分. 本模块只给出
材料侧的分解与导数, 基矩阵的积分由有限元分析器负责.
"""

from typing import Tuple

from soptx.backend import backend_manager as bm
from soptx.typing import TensorLike

_PLANE_HYPOTHESES = ("plane_stress", "plane_strain")


def _check_hypothesis(hypothesis: str) -> None:
    if hypothesis != "3D" and hypothesis not in _PLANE_HYPOTHESES:
        raise ValueError(
            f"hypothesis must be one of 3D, plane_strain, plane_stress, got {hypothesis!r}"
        )


def lame_basis_matrices(hypothesis: str, *, device=None) -> Tuple[TensorLike, TensorLike]:
    """返回常数矩阵 (D_λ, D_μ), 满足 D = λ* D_λ + μ D_μ.

    Parameters
    ----------
    hypothesis : 应力应变假设, 取 '3D', 'plane_strain' 或 'plane_stress'.
    device : 结果所在的设备; 可为 None.

    Returns
    -------
    D_lam, D_mu : '3D' 下各为 (6, 6), 平面假设下各为 (3, 3).
    """
    _check_hypothesis(hypothesis)
    kwargs = dict(dtype=bm.float64, device=device)
    if hypothesis in _PLANE_HYPOTHESES:
        D_lam = bm.tensor([[1.0, 1.0, 0.0],
                           [1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0]], **kwargs)
        D_mu = bm.tensor([[2.0, 0.0, 0.0],
                          [0.0, 2.0, 0.0],
                          [0.0, 0.0, 1.0]], **kwargs)
    else:
        D_lam = bm.zeros((6, 6), **kwargs)
        D_lam = bm.set_at(D_lam, (slice(0, 3), slice(0, 3)), 1.0)
        D_mu = bm.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], **kwargs)

    return D_lam, D_mu


def lame_parameters(E: TensorLike, nu: TensorLike, hypothesis: str) -> Tuple[TensorLike, TensorLike]:
    """由 (E, ν) 计算分解系数 (λ*, μ).

    Parameters
    ----------
    E : 杨氏模量, 任意形状.
    nu : 泊松比, 与 ``E`` 同形状.
    hypothesis : 应力应变假设; λ* 在平面应变、3D 下取 λ, 平面应力下取 λ̄.

    Returns
    -------
    lam, mu : 与 ``E`` 同形状的 λ* 与 μ.
    """
    _check_hypothesis(hypothesis)
    mu = E / (2.0 * (1.0 + nu))
    if hypothesis == "plane_stress":
        lam = E * nu / (1.0 - nu**2)
    else:
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    return lam, mu


def lame_parameter_derivatives(E: TensorLike,
                               nu: TensorLike,
                               hypothesis: str,
                            ) -> Tuple[TensorLike, TensorLike, TensorLike, TensorLike]:
    """计算 (λ*, μ) 对 (E, ν) 的偏导数.

    Parameters
    ----------
    E : 杨氏模量, 任意形状.
    nu : 泊松比, 与 ``E`` 同形状.
    hypothesis : 应力应变假设, 含义同 ``lame_parameters``.

    Returns
    -------
    dlam_dE, dlam_dnu, dmu_dE, dmu_dnu : 与 ``E`` 同形状的 ∂λ*/∂E, ∂λ*/∂ν, ∂μ/∂E, ∂μ/∂ν.
    """
    _check_hypothesis(hypothesis)
    one_plus = 1.0 + nu
    dmu_dE = 1.0 / (2.0 * one_plus)
    dmu_dnu = -E / (2.0 * one_plus**2)
    if hypothesis == "plane_stress":
        denom = 1.0 - nu**2
        dlam_dE = nu / denom
        dlam_dnu = E * (1.0 + nu**2) / denom**2
    else:
        denom = one_plus * (1.0 - 2.0 * nu)
        dlam_dE = nu / denom
        dlam_dnu = E * (1.0 + 2.0 * nu**2) / denom**2

    return dlam_dE, dlam_dnu, dmu_dE, dmu_dnu


def elastic_matrices(E: TensorLike, nu: TensorLike, hypothesis: str, *, device=None) -> TensorLike:
    """由逐单元 (E, ν) 构造逐单元本构矩阵 D_e = λ*_e D_λ + μ_e D_μ.

    Parameters
    ----------
    E : (NC, ) 的杨氏模量.
    nu : (NC, ) 的泊松比.
    hypothesis : 应力应变假设.
    device : 常数矩阵 D_λ、D_μ 所在的设备, 须与 ``E`` 一致; 可为 None.

    Returns
    -------
    D : (NC, NS, NS) 的本构矩阵, NS 在 '3D' 下为 6, 平面假设下为 3.
    """
    lam, mu = lame_parameters(E, nu, hypothesis)
    D_lam, D_mu = lame_basis_matrices(hypothesis, device=device)

    return (bm.einsum('c, kl -> ckl', lam, D_lam)
            + bm.einsum('c, kl -> ckl', mu, D_mu))
