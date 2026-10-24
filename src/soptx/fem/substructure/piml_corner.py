"""linear_corner 下由形函数网络预测内部延拓并按变分式构造角点刚度的缩聚器.

与 ``streaming.assemble_corner_from_stiffness_batches`` 同一调用协议, 供
``SubstructureAnalyzer(corner_condensation=...)`` 注入. 每批把网络预测的内部形函数
``N_i`` 与角点迹矩阵 ``L`` 拼成 ``basis = [L; N_i]``, 角点刚度取 ``basis^T K_local basis``;
恢复批次直接给出 ``N_i``, 与分析器的角点恢复路径一致. 三项预测质量诊断量只记录不替换.
推理、补全、变分式与诊断都在接口迹矩阵所在的设备上进行, 只把诊断比值取回主机汇总.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Optional

import numpy as np

from soptx.backend import backend_manager as bm

from .streaming import TraceStiffnessBatch

DIAGNOSTIC_KEYS = ("rigid_residual", "excess_ratio", "reduced_rcond")


def _summary(values: np.ndarray) -> dict:
    """返回一维数组的最小、中位与最大值."""
    return dict(min=float(values.min()), median=float(np.median(values)), max=float(values.max()))


class PIMLShapeCornerCondensation:
    """形函数预测路线的角点缩聚器.

    Parameters
    ----------
    decoder : IndependentPredictionDecoder
        由分析器的参考子结构构造的 linear_corner 解码器, 提供独立条目补全与刚体基.
    network : torch.nn.Module
        参数为 float64 的形函数网络, 输入归一化模量 ``E(rho)/E0``; 须与分析器的接口迹矩阵位于
        同一设备 (numpy 后端为 CPU), 调用时核对.
    provider_metadata : dict
        权重保存的提供器元数据, 用于核对解码器契约.
    uniform_threshold : float, optional
        子结构内 ``max - mean`` 模量差小于该值时改用实体形函数; None 不判别.
    solid_shape : ndarray, optional
        实体子结构的精确内部形函数 ``(n_i, n_q)``; 启用均匀判别时必须给出.

    Attributes
    ----------
    report : dict
        最近一次调用的诊断: 预测数、均匀替换数与三项比值的分布.

    Notes
    -----
    三项比值的定义与 ``ShapeFunctionCondensation`` 的门禁一致: 刚体零空间残量
    ``|K_c R_q|_F / |K_c|_F``, 相对内部固支刚度 ``L^T K_bb L`` 的刚化比
    ``lambda_max(K_c - L^T K_bb L) / lambda_max(L^T K_bb L)``, 变形子空间上的条件数倒数
    ``lambda_min / lambda_max``. 这里只记录分布, 不触发回退.
    """

    def __init__(
        self,
        decoder: Any,
        network: Any,
        *,
        provider_metadata: dict,
        uniform_threshold: Optional[float] = None,
        solid_shape: Optional[Any] = None,
    ) -> None:
        import torch
        from soptx.ml.substructure.independent_checkpoints import decoder_metadata_matches

        if decoder.trace_kind != "linear_corner":
            raise ValueError(f"解码器的接口须为 linear_corner, 得到 {decoder.trace_kind!r}")
        if not decoder_metadata_matches(provider_metadata, decoder.metadata()):
            raise ValueError("解码器的子结构配置或独立条目编号与权重不一致")
        parameters = tuple(network.parameters())
        if any(p.dtype != torch.float64 for p in parameters):
            raise ValueError("形函数网络须为 torch.float64")
        if uniform_threshold is not None:
            if not (float(uniform_threshold) > 0.0):
                raise ValueError("uniform_threshold 须为正数")
            if solid_shape is None:
                raise ValueError("启用均匀判别须提供 solid_shape")
        self.decoder = decoder
        self.network = network
        self.network.eval()
        self.uniform_threshold = None if uniform_threshold is None else float(uniform_threshold)
        self.solid_shape = None if solid_shape is None else np.asarray(solid_shape, dtype=np.float64)
        prototype = decoder.prototype
        self._i = np.asarray(bm.to_numpy(prototype.i_dofs), dtype=np.int64)
        self._b = np.asarray(bm.to_numpy(prototype.b_dofs), dtype=np.int64)
        # 边界行在前、内部行在后拼接后, 按 _order 重排即回到参考子结构的自由度顺序
        stacked = np.concatenate((self._b, self._i))
        if not np.array_equal(np.sort(stacked), np.arange(len(stacked))):
            raise ValueError("内部与边界自由度须恰好划分参考子结构的全部自由度")
        self._order = np.argsort(stacked)
        self._n_cells = int(prototype.n_cells)
        rigid, deformation, _ = prototype.trace_interface_bases(decoder.trace)
        self._rigid = np.asarray(bm.to_numpy(rigid), dtype=np.float64)
        self._deformation = np.asarray(bm.to_numpy(deformation), dtype=np.float64)
        self._decoder_trace = np.asarray(bm.to_numpy(decoder.trace.matrix), dtype=np.float64)
        if self.solid_shape is not None and self.solid_shape.shape != (len(self._i), self._rigid.shape[0]):
            raise ValueError(f"solid_shape 形状须为 {(len(self._i), self._rigid.shape[0])}, "
                             f"得到 {self.solid_shape.shape}")
        self.report: dict = {}

    def _predict(self, modulus: Any) -> Any:
        """由归一化模量预测并补全内部形函数 ``(B, n_i, n_q)``, 与 ``modulus`` 同设备."""
        from soptx.ml.substructure.inference import predict_independent_outputs

        outputs = predict_independent_outputs(self.network, modulus, "shape")
        shapes = self.decoder.shape_codec.decode(outputs)
        if tuple(shapes.shape) != (len(modulus), len(self._i), self._rigid.shape[0]):
            raise ValueError(f"补全后的形函数形状 {tuple(shapes.shape)} 不符")
        return shapes

    @staticmethod
    def _diagnostics(stiffness: Any, K_bb_trace: Any, R: Any, D: Any) -> dict:
        """对一批角点刚度计算三项比值, 取回主机为 numpy 数组."""
        def frobenius(matrices):
            return bm.sqrt(bm.sum(matrices * matrices, axis=(-2, -1)))

        rigid = frobenius(stiffness @ R) / frobenius(stiffness)
        excess = (bm.linalg.eigvalsh(stiffness - K_bb_trace)[:, -1]
                  / bm.linalg.eigvalsh(K_bb_trace)[:, -1])
        reduced = bm.linalg.eigvalsh(bm.matrix_transpose(D) @ stiffness @ D)
        rcond = reduced[:, 0] / reduced[:, -1]
        return {key: np.asarray(bm.to_numpy(value), dtype=np.float64) for key, value in
                dict(rigid_residual=rigid, excess_ratio=excess, reduced_rcond=rcond).items()}

    def __call__(
        self,
        stiffness_batches: Any,
        interface_space: Any,
        *,
        i_dofs: Any,
        b_dofs: Any,
        modulus_lookup: Callable[[int, int], Any],
        timings: Optional[dict] = None,
    ) -> tuple[Any, list]:
        """消费局部刚度批次, 用预测形函数组装角点系统并保存恢复矩阵.

        Parameters
        ----------
        stiffness_batches : iterable
            从 0 开始连续的 ``(start, end, K_local)`` 批次.
        interface_space : InterfaceSpace
            linear_corner 接口空间.
        i_dofs, b_dofs : array_like
            内部和边界自由度编号, 须与解码器的参考子结构一致.
        modulus_lookup : callable
            ``modulus_lookup(start, end)`` 返回该批按参考单元顺序排列的归一化模量 ``(B, n_cells)``.
        timings : dict, optional
            写入分项墙钟时间: local_stiffness 为获取上游批次, inference 为推理与补全,
            basis_assembly 为基函数拼接与变分式, diagnostics 为三项比值, global_csr 为
            散加与编排; local_condensation_total 为中间三项之和.

        Returns
        -------
        system : InterfaceSystem
            全局角点接口刚度.
        recovery_batches : list
            ``(start, end, N_i)``, 满足 ``u_i = N_i q``.
        """
        measured = {key: 0.0 for key in
                    ("local_stiffness", "inference", "basis_assembly", "diagnostics", "global_csr")}
        total_started = perf_counter()
        basis_trace = interface_space.trace_basis
        if basis_trace.name != "linear_corner":
            raise ValueError("形函数预测缩聚仅支持 linear_corner.")
        i = np.asarray(bm.to_numpy(bm.asarray(i_dofs)), dtype=np.int64)
        b = np.asarray(bm.to_numpy(bm.asarray(b_dofs)), dtype=np.int64)
        if not (np.array_equal(i, self._i) and np.array_equal(b, self._b)):
            raise ValueError("分析器的内部/边界自由度编号与解码器的参考子结构不一致.")
        P = basis_trace.matrix
        P_host = np.asarray(bm.to_numpy(P), dtype=np.float64)
        if (P_host.shape != self._decoder_trace.shape
                or not np.allclose(P_host, self._decoder_trace, rtol=0, atol=1e-12)):
            raise ValueError("接口空间的角点迹矩阵与解码器的不一致.")
        # 计算设备取接口迹矩阵所在设备, 与分析器的局部刚度及接口散加一致; 网络须已在该设备上
        device = bm.get_device(P)
        import torch
        if any(p.device != torch.device(device) for p in self.network.parameters()):
            raise ValueError(f"形函数网络须与接口迹矩阵位于同一设备 {device}.")
        b_device = bm.asarray(b, device=device)
        order = bm.asarray(self._order, device=device)
        R = bm.asarray(self._rigid, device=device)
        D = bm.asarray(self._deformation, device=device)
        solid = None if self.solid_shape is None else bm.asarray(self.solid_shape, device=device)
        K_trace = bm.matrix_transpose(P)
        n_dof = len(i) + len(b)
        recovery_batches: list = []
        collected = {key: [] for key in DIAGNOSTIC_KEYS}
        counts = dict(predicted=0, uniform_replaced=0)

        def batches():
            iterator = iter(stiffness_batches)
            expected_start = 0
            while True:
                started = perf_counter()
                try:
                    start, end, local = next(iterator)
                except StopIteration:
                    measured["local_stiffness"] += perf_counter() - started
                    break
                measured["local_stiffness"] += perf_counter() - started
                if (start != expected_start or end <= start
                        or tuple(local.shape) != (end - start, n_dof, n_dof)):
                    raise ValueError("局部刚度批次的区间或形状不符.")
                expected_start = end

                started = perf_counter()
                modulus = bm.asarray(modulus_lookup(start, end), dtype=bm.float64, device=device)
                if tuple(modulus.shape) != (end - start, self._n_cells):
                    raise ValueError(f"模量批次形状 {tuple(modulus.shape)} 须为 {(end - start, self._n_cells)}")
                shapes = self._predict(modulus)
                counts["predicted"] += end - start
                if self.uniform_threshold is not None:
                    uniform = (bm.max(modulus, axis=1) - bm.mean(modulus, axis=1)) < self.uniform_threshold
                    n_uniform = int(bm.sum(uniform))
                    if n_uniform:
                        shapes = bm.where(uniform[:, None, None], solid, shapes)
                        counts["uniform_replaced"] += n_uniform
                measured["inference"] += perf_counter() - started

                started = perf_counter()
                trace_rows = bm.broadcast_to(P, (end - start, ) + tuple(P.shape))
                basis = bm.concat((trace_rows, shapes), axis=1)[:, order]
                stiffness = bm.matrix_transpose(basis) @ local @ basis
                # 变分式结果对称到舍入; 对称化后交给 CG, 与精确路径的对称性一致
                stiffness = 0.5 * (stiffness + bm.matrix_transpose(stiffness))
                recovery_batches.append((start, end, shapes))
                measured["basis_assembly"] += perf_counter() - started

                started = perf_counter()
                K_bb_trace = K_trace @ local[..., b_device[:, None], b_device] @ P
                for key, value in self._diagnostics(stiffness, K_bb_trace, R, D).items():
                    collected[key].append(value)
                measured["diagnostics"] += perf_counter() - started
                yield TraceStiffnessBatch(start, end, stiffness)

        system = interface_space.assemble(batches())
        measured["global_csr"] = max(0.0, perf_counter() - total_started - sum(measured.values()))
        measured["local_condensation_total"] = sum(
            measured[key] for key in ("inference", "basis_assembly", "diagnostics"))
        if timings is not None:
            timings.update(measured)
        self.report = dict(
            counts,
            **{key: _summary(np.concatenate(values)) for key, values in collected.items()},
        )
        return system, recovery_batches
