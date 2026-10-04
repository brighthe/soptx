"""局部预测及精确矩阵对照, 不负责采样或文件保存."""

import numpy as np
import torch
from soptx.backend import backend_manager as bm
from soptx.fem.substructure import ShapeFunctionCondensation
from .independent_checkpoints import load_independent_network

def load_local_model(checkpoint_path, provider, route, device="cpu"):
    """加载单条预测路线并保留权重来源.

    Parameters
    ----------
    checkpoint_path : str or Path
        最佳权重文件.
    provider : IndependentTargetProvider
        当前子结构配置与标签提供器.
    route : str
        shape 或 stiffness.
    device : str
        PyTorch 设备标识.

    Returns
    -------
    torch.nn.Module
        已进入评估模式的 float64 网络, 来源保存在 local_validation_source.
    """
    model, source = load_independent_network(checkpoint_path, provider.metadata(), route=route)
    source = {**source, "route": route, "provider_metadata": provider.metadata()}
    model.local_validation_source = source
    return model.to(device=device, dtype=torch.float64).eval()


def _relative(numerator, denominator):
    """计算相对量并显式处理零参考范数.

    Parameters
    ----------
    numerator, denominator : float
        分子及参考范数.

    Returns
    -------
    float or None
        零参考范数时为 None, 其余为有限比值.
    """
    if denominator == 0.0:
        return None
    result = float(numerator / denominator)
    if not np.isfinite(result):
        raise ValueError("相对指标产生非有限值")
    return result


class LocalPredictionEvaluator:
    """复用子结构配置进行批量局部预测诊断.

    Parameters
    ----------
    network : torch.nn.Module
        待评价网络.
    provider : IndependentTargetProvider
        精确矩阵与约束补全提供器.
    route : str
        shape 或 stiffness.
    device : str
        网络推理设备.
    """

    def __init__(self, network, provider, route, device="cpu"):
        if route not in ("shape", "stiffness"):
            raise ValueError("route 必须为 shape 或 stiffness")
        source = getattr(network, "local_validation_source", None)
        if source is not None and (
            source.get("route") != route or source.get("provider_metadata") != provider.metadata()
        ):
            raise ValueError("加载模型的路线或子结构配置与当前验证不一致")
        network.to(device=device, dtype=torch.float64).eval()
        rigid, deformation, interior = provider.prototype.trace_interface_bases(provider.trace)
        q = np.asarray(bm.to_numpy(rigid), dtype=np.float64)
        phi = np.asarray(bm.to_numpy(interior), dtype=np.float64)
        complement = np.asarray(bm.to_numpy(deformation), dtype=np.float64)
        builder = None
        if route == "shape":
            builder = ShapeFunctionCondensation(
                provider.prototype.i_dofs, provider.prototype.b_dofs,
                rigid_basis=rigid, deformation_basis=deformation,
                rigid_interior=interior, trace=provider.trace,
            )
        self.network, self.provider, self.route, self.device = network, provider, route, device
        self.q, self.phi, self.complement, self.builder = q, phi, complement, builder

    def evaluate(self, inputs):
        """计算一批给定材料场的预测误差和约束诊断.

        Parameters
        ----------
        inputs : array_like
            形状为 (batch, n_cells) 的归一化杨氏模量.

        Returns
        -------
        list[dict]
            每个样本的指标、特征值分类及未定义相对量名称.
        """
        network, provider, route, device = self.network, self.provider, self.route, self.device
        q, phi, complement, builder = self.q, self.phi, self.complement, self.builder
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or len(inputs) == 0:
            raise ValueError("inputs 必须为非空二维材料数组")
        network.eval()
        records = []
        exact = provider.exact_matrices(inputs)
        if any(not np.isfinite(value).all() for value in exact.values()):
            raise ValueError("精确参考包含非有限值")
        with torch.inference_mode():
            output = network(torch.as_tensor(inputs, dtype=torch.float64, device=device))
            if tuple(output.shape) != (len(inputs), provider.codecs[route].n_output):
                raise ValueError("网络输出形状与当前路线不一致")
            if not torch.isfinite(output).all().item():
                raise ValueError("网络输出包含非有限值")
            prediction = provider.codecs[route].decode(output).cpu().numpy()
        if not np.isfinite(prediction).all():
            raise ValueError("预测矩阵包含非有限值")
        if route == "shape":
            predicted_stiffness = np.asarray(bm.to_numpy(builder.assemble_reduced_stiffness(
                bm.asarray(exact["local_stiffness"], dtype=bm.float64),
                bm.asarray(prediction, dtype=bm.float64),
            )), dtype=np.float64)
        else:
            predicted_stiffness = prediction
        for offset in range(len(inputs)):
            metrics = {}
            if route == "shape":
                b = prediction[offset]
                reference_b = exact["shape"][offset]
                metrics["shape_relative_error"] = _relative(np.linalg.norm(b - reference_b), np.linalg.norm(reference_b))
                metrics["rigid_reproduction_residual"] = float(np.linalg.norm(b @ q - phi))
                metrics["rigid_reproduction_relative_residual"] = _relative(
                    metrics["rigid_reproduction_residual"], np.linalg.norm(phi))
            k = predicted_stiffness[offset]
            reference_k = exact["stiffness"][offset]
            metrics["stiffness_relative_error"] = _relative(np.linalg.norm(k - reference_k), np.linalg.norm(reference_k))
            metrics["symmetry_residual"] = float(np.linalg.norm(k - k.T))
            metrics["rigid_nullspace_residual"] = float(np.linalg.norm(k @ q))
            metrics["symmetry_relative_residual"] = _relative(
                metrics["symmetry_residual"], np.linalg.norm(k))
            metrics["rigid_nullspace_relative_residual"] = _relative(
                metrics["rigid_nullspace_residual"], np.linalg.norm(k) * np.linalg.norm(q))
            if not np.isfinite(k).all():
                raise ValueError("重构刚度包含非有限值")
            projected = complement.T @ ((k + k.T) * 0.5) @ complement
            eigenvalues = np.linalg.eigvalsh(projected)
            minimum = float(eigenvalues[0])
            tolerance = float(100 * np.finfo(np.float64).eps * np.max(np.abs(eigenvalues)))
            category = "negative" if minimum < -tolerance else "positive" if minimum > tolerance else "near_zero"
            metrics["min_deformation_eigenvalue"] = minimum
            metrics["eigenvalue_tolerance"] = tolerance
            if any(value is not None and not np.isfinite(value) for value in metrics.values()):
                raise ValueError("局部诊断产生非有限值")
            records.append({**metrics, "deformation_spectrum": category,
                            "undefined_metrics": [key for key, value in metrics.items() if value is None]})
        return records
