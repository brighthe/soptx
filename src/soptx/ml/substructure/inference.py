"""独立矩阵条目网络的在线推理."""


def predict_independent_outputs(model, modulus, route):
    """执行 CPU float64 推理, 返回尚未约束补全的独立条目.

    Parameters
    ----------
    model : torch.nn.Module
        位于 CPU 且参数为 float64 的已训练网络.
    modulus : numpy.ndarray
        形状为 (batch, n_cells) 的归一化杨氏模量, 按 FE cell 顺序排列.
    route : str
        shape 或 stiffness, 用于区分错误信息所属路线.

    Returns
    -------
    numpy.ndarray
        形状为 (batch, n_output) 的有限 float64 独立条目.

    Raises
    ------
    ValueError
        网络参数设备或精度不符, 或输出形状及有限性检查失败.
    """
    import numpy as np
    import torch

    parameters = tuple(model.parameters())
    if any(parameter.device.type != "cpu" for parameter in parameters):
        raise ValueError(f"{route} 网络必须位于 CPU.")
    if any(parameter.dtype != torch.float64 for parameter in parameters):
        raise ValueError(f"{route} 网络必须使用 torch.float64.")
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(modulus).to(dtype=torch.float64))
    result = np.asarray(prediction.detach().cpu().numpy(), dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != modulus.shape[0]:
        raise ValueError(f"{route} 网络输出形状 {result.shape} 与 batch 不一致.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{route} 网络输出包含非有限值.")
    return result
