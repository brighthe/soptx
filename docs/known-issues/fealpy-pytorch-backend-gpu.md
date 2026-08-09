# 上游缺陷：FEALPy 4.0.0 PyTorch 后端 GPU 路径缺失

FEALPy 4.0.0 的 PyTorch 后端有两处算子缺失导致 GPU 路径无法走通，另有一处稀疏
矩阵封装缺口导致 `COOTensor`/`CSRTensor` 不暴露 `device` 属性。

| # | 缺陷 | 位置 | 修复 |
| --- | --- | --- | --- |
| 1 | 缺少 `take_along_axis` | `pytorch_backend.py` | 新增 staticmethod，包装 `torch.take_along_dim`，自动将 indices 转 long |
| 2 | 缺少 `unique_counts` | `pytorch_backend.py` | 新增 staticmethod，包装 `torch.unique(return_counts=True)` |
| 3 | `COOTensor`/`CSRTensor` 无 `device` 属性 | `sparse/coo_tensor.py`<br>`sparse/csr_tensor.py`<br>`sparse/sparse_tensor.py` | 基类新增 abstract property，子类返回底层 tensor 的 `.device` |

缺陷 1 和 2 阻断 GPU 求解链路；缺陷 3 不影响计算正确性，但导致用户无法用统一
方式确认数据落点。

---

## 缺陷 1：`take_along_axis` 缺失

**症状**：`bm.set_backend("pytorch")` 后装配直接抛异常：
```
AttributeError: 'PyTorchBackend' object has no attribute 'take_along_axis'.
      Did you mean: 'apply_along_axis'?
```

**根因**：NumPy 有 `np.take_along_axis`，PyTorch 对应为 `torch.take_along_dim`
（参数名 `axis` → `dim`）。`FUNCTION_MAPPING` 同名映射不到，且未被显式映射表
覆盖。此外 `torch.take_along_dim` 要求 indices 为 `Long`（int64），而上游调用处
传了 `dtype=bm.uint8`，直接映射也会抛 `RuntimeError`。

**修复**：
```python
# pytorch_backend.py, PyTorchBackend 类内
@staticmethod
def take_along_axis(x, indices, /, *, axis):
    return torch.take_along_dim(x, indices.long(), dim=axis)
```

## 缺陷 2：`unique_counts` 缺失

**症状**：修复缺陷 1 后，`apply_bc()` 抛异常：
```
AttributeError: 'PyTorchBackend' object has no attribute 'unique_counts'
```

**根因**：`np.unique_counts`（NumPy 2.0+）的 PyTorch 对应是
`torch.unique(return_counts=True)`，不是独立函数名。

**修复**：
```python
# pytorch_backend.py, PyTorchBackend 类内
@staticmethod
def unique_counts(x, /):
    values, counts = torch.unique(x, return_counts=True)
    return values, counts
```

## 缺陷 3：`COOTensor` / `CSRTensor` 不暴露 `device`

**症状**：`mesh.device` ✅，但 `K.device` ❌：
```python
>>> mesh.device   # cuda:0
>>> K.device      # AttributeError
>>> K.data.device # cuda:0  ← 只能钻到内部
```

**根因**：`Mesh` 和 `Tensor` 都暴露 `.device`，`COOTensor`/`CSRTensor` 同为
fealpy 数据容器却没有，排查路径不统一。

**修复**：
```python
# sparse_tensor.py 基类
@property
def device(self): raise NotImplementedError

# coo_tensor.py
@property
def device(self):
    return self._indices.device

# csr_tensor.py
@property
def device(self):
    return self._col.device
```

---

## 对 SOPTX 的影响

- 缺陷 1–2：`examples/gpu_elasticity/minimal_demo.py` 依赖此修复。
  数值上无静默错误——直接抛异常，不会产生假数据。
- 缺陷 3：与计算正确性无关，仅影响调试体验。

---

## 回归测试

**三处修复均无 pytest 覆盖。** merge 上游后补丁若被覆盖，只能靠
`examples/gpu_elasticity/minimal_demo.py` 在 GPU 机器上运行时暴露，CPU CI 检查
不到。缺陷 1–2 会直接抛 `AttributeError`（不产生假数据），缺陷 3 不影响计算，
因此暂未补测试；若后续 GPU 路径进入常规验证，应在 fork 内补
`tests/backend/` 下的最小用例。

## 环境与版本对照

| 检出 | 状态 |
| --- | --- |
| 上游 `suanhai/develop` (`2f17532`, `v4.0.0-alpha-15`) | 缺陷 1–3 均存在 |
| 本地 fork | 全部已修复：`ceb0c61`（缺陷 1、2）、`88cf4fa`（缺陷 3） |
