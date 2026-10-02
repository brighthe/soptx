"""统一局部缩聚结果的批量刚度与位移恢复接口."""

from __future__ import annotations

from typing import Any, Callable, Tuple

from fealpy.backend import backend_manager as bm

def normalize_local_reduction(
    condensors: Any,
    n_sub_total: int,
    n_b: int,
) -> Tuple[Any, Callable[[Any], Any]]:
    """把逐个或批量给出的缩聚结果统一成批量形式.

    Parameters
    ----------
    condensors : Any
        缩聚器列表、单个旧式批量缩聚器或无状态
        ``LocalReductionBatchResult``. 刚度批量形状为 ``(B, n_b, n_b)``;
        旧式二维 ``K_s`` 表示全部子结构共用同一结果.
    n_sub_total : int
        子结构总数 ``B``.
    n_b : int
        单个子结构的接口自由度数.

    Returns
    -------
    K_s_batch : Any
        批量缩聚刚度, 形状 ``(B, n_b, n_b)``. 流式结果返回 ``None``.
    recover : Callable
        把形状 ``(B, n_b)`` 的接口位移映射为形状 ``(B, n_i)`` 的函数.

    Raises
    ------
    ValueError
        缩聚器数量或 ``K_s`` 形状与子结构不匹配.
    RuntimeError
        任一旧式缩聚器尚未调用 ``condense``.
    """
    if isinstance(condensors, (list, tuple)):
        if len(condensors) != n_sub_total:
            raise ValueError(
                f"sub_meshes 与 condensors 的数量必须一致; "
                f"当前为 {n_sub_total} 与 {len(condensors)}."
            )
        blocks = []
        for idx, condensor in enumerate(condensors):
            if condensor.K_s is None:
                raise RuntimeError(
                    f"第 {idx} 个 condensor 必须在全局装配前完成 condense()."
                )
            blocks.append(condensor.K_s)
        K_s_batch = bm.stack(blocks, axis=0)

        def recover(u_b_batch: Any) -> Any:
            """逐个子结构恢复内部位移后堆叠."""
            return bm.stack(
                [c.recover(u_b_batch[i]) for i, c in enumerate(condensors)],
                axis=0,
            )
    elif hasattr(condensors, "stiffness") and hasattr(condensors, "recover"):
        # ``LocalReductionBatchResult`` 是新无状态缩聚契约. 直接消费结果
        # 快照, 避免先回填旧 condensor 的 K_s/N 可变属性.
        K_s_batch = condensors.stiffness

        def recover(u_b_batch: Any) -> Any:
            """由无状态批量缩聚结果恢复内部位移."""
            return condensors.recover(u_b_batch)

    else:
        condensor = condensors
        if getattr(condensor, "K_s", None) is None:
            if hasattr(condensor, "get_chunk_stiffness"):
                # 流式容器模式: 不在内存中持有全局全量张量, 按需由 get_chunk_stiffness 提供
                def recover(u_b_batch: Any) -> Any:
                    return condensor.recover(u_b_batch)
                return None, recover
            raise RuntimeError("condensor 必须在全局装配前完成 condense().")
        K_s_batch = condensor.K_s
        if K_s_batch.ndim == 2:
            # 单个缩聚结果广播到全部子结构, 对应各子结构密度完全相同的情形.
            K_s_batch = bm.broadcast_to(
                K_s_batch[None, ...], (n_sub_total,) + tuple(K_s_batch.shape)
            )

        def recover(u_b_batch: Any) -> Any:
            """批量缩聚器的 recover 沿前导维广播, 一次完成全部子结构."""
            return condensor.recover(u_b_batch)

    if K_s_batch is not None:
        if K_s_batch.ndim != 3 or tuple(K_s_batch.shape) != (n_sub_total, n_b, n_b):
            raise ValueError(
                f"K_s 的批量形状必须为 ({n_sub_total}, {n_b}, {n_b}); "
                f"当前为 {tuple(K_s_batch.shape)}."
            )
    return K_s_batch, recover

