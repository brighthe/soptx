"""独立矩阵条目网络的在线推理."""


def predict_independent_outputs(model, modulus, route):
    """执行 float64 推理, 返回尚未约束补全的独立条目.

    Parameters
    ----------
    model : torch.nn.Module
        参数为 float64 的已训练网络. numpy 输入时须位于 CPU, torch 输入时须与输入位于同一设备.
    modulus : numpy.ndarray or torch.Tensor
        形状为 (batch, n_cells) 的归一化杨氏模量, 按 FE cell 顺序排列.
    route : str
        shape 或 stiffness, 用于区分错误信息所属路线.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        形状为 (batch, n_output) 的有限 float64 独立条目; 类型与 ``modulus`` 相同, torch 输出与输入同设备.

    Raises
    ------
    ValueError
        网络参数设备或精度不符, 或输出形状及有限性检查失败.
    """
    import numpy as np
    import torch

    parameters = tuple(model.parameters())
    tensor_input = isinstance(modulus, torch.Tensor)
    if tensor_input:
        if any(parameter.device != modulus.device for parameter in parameters):
            raise ValueError(f"{route} 网络须与输入位于同一设备 {modulus.device}.")
    elif any(parameter.device.type != "cpu" for parameter in parameters):
        raise ValueError(f"{route} 网络必须位于 CPU.")
    if any(parameter.dtype != torch.float64 for parameter in parameters):
        raise ValueError(f"{route} 网络必须使用 torch.float64.")
    model.eval()
    inputs = modulus if tensor_input else torch.from_numpy(np.asarray(modulus))
    with torch.no_grad():
        prediction = model(inputs.to(dtype=torch.float64)).detach()
    if prediction.ndim != 2 or prediction.shape[0] != modulus.shape[0]:
        raise ValueError(f"{route} 网络输出形状 {tuple(prediction.shape)} 与 batch 不一致.")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError(f"{route} 网络输出包含非有限值.")
    if tensor_input:
        return prediction.to(dtype=torch.float64)
    return np.asarray(prediction.cpu().numpy(), dtype=np.float64)
