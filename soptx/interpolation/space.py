from typing import Union, Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, Number, _S, Size
from fealpy.functionspace import LagrangeFESpace, Function

class ShepardFESpace(LagrangeFESpace):
    """Shepard 插值函数空间：沿用 Lagrange 的 DOF 布局，只换权重函数"""

    def __init__(self, mesh, p: int = 1, power: float = 2.0):
        super().__init__(mesh, p=p, ctype='C')     # 复用父类初始化
        self._power = power                        # Shepard 权重指数

    # ------ 连续场评估 -------------------------------------------------- #
    def value(self,
            rho_dof: TensorLike,
            points : TensorLike,
            *,
            coordtype: str = 'cartesian',   # <- 新增
            index: Index = _S               # 与父类保持签名兼容
            ) -> TensorLike:
        """ρ(points) using Shepard weights"""
        
        if coordtype != 'cartesian':
            raise NotImplementedError(
                f"ShepardFESpace only supports coordtype='cartesian', "
                f"got '{coordtype}'")
        
        if points.ndim != 2:
            raise ValueError("`points` must be a 2-D tensor of shape (M, GD)")
    
        ip = self.interpolation_points()                       # (GDOF, GD)
        pts = points                            # (M, GD)

        dist2 = bm.sum((pts[:, None, :] - ip[None, :, :])**2, axis=-1)  # (M, GDOF)
        dist2 = bm.maximum(dist2, 1e-12)
        w = dist2 ** (-self._power / 2.0)
        w /= bm.sum(w, axis=1, keepdims=True)

        return bm.dot(w, rho_dof).squeeze()                    # (M, )

    # 如果后端其他模块会调 grad_value，可按需要实现；暂时抛错
    def grad_value(self, *args, **kwargs):
        raise NotImplementedError("grad_value for ShepardFESpace is not implemented")

    # ------ 构造连续函数对象 ------------------------------------------- #
    def function(self,
                 array: Optional[TensorLike] = None,
                 batch: Union[int, Size, None] = None,
                 *,
                 dtype=None,
                 device=None):
        if array is None:
            array = super().array(batch=batch, dtype=dtype, device=device)
        return ShepardFunction(self, array)                    # 返回下文定义的函数

class ShepardFunction(Function):
    """Shepard 连续密度：既当 1-D 张量，又可调用"""

    def __init__(self, space: ShepardFESpace, array: TensorLike):
        super().__init__(space, array, coordtype='cartesian')
        self._power = space._power        # 仅备用

    def __call__(self,
                 points: TensorLike,
                 *,
                 coordtype: str = 'cartesian') -> TensorLike:
        # 注意此处使用 self.array 属性，而非 self.array()
        return self.space.value(self.array, points, coordtype=coordtype)


    # 索引 / 切片 / __array__ 等行为已由父类 Function 提供
