# 功能增强：CG 可插拔内积与 EntityMPI.dot()

为 `fealpy.solver.cg` 增加可插拔内积接口，`fealpy.distributed.EntityMPI` 增加内积工厂方法，使分布式 MPI 求解器可以复用同一个 CG 实现。soptx 不再需要维护独立的 CG 副本。

| # | 改动 | 位置 | 内容 |
| --- | --- | --- | --- |
| 1 | CG 可插拔内积 + 真残差刷新 | `solver/cg.py` | 新增 `dot_product`、`residual_refresh` 参数 |
| 2 | EntityMPI 内积工厂 | `distributed/entity_mpi.py` | 新增 `dot()` 方法，返回重叠修正后的 `(dot_fn, norm_fn)` |

---

## 改动 1：`fealpy.solver.cg` 可插拔内积

### 新增参数

```python
def cg(A, b, x0=None, M=None, *,
       # ... 现有参数不变 ...
       dot_product: Callable | None = None,   # 自定义内积 dot(x, y) -> float
       residual_refresh: int = 0,             # >0 时每 N 步算真残差
       ) -> Tensor | tuple[Tensor, dict]:
```

**行为**：

- `dot_product is None`：默认 `bm.sum(bm.conj(x) * y)`，与现有行为完全一致，向后兼容
- `dot_product` 非空：全部内积（curvature、residual squared、norm）走自定义函数
- `residual_refresh=0`：只靠递推残差判收敛（现有行为）
- `residual_refresh>0`：每 N 步算 `||b - A@x||` 真残差，防递推残差漂移

### 新增 info 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `converged` | `bool` | 是否收敛 |
| `breakdown` | `str | None` | curvature ≤ 0 或 residual ≤ 0 时非空 |
| `true_residual` | `float | None` | `residual_refresh>0` 时记录最后一次真残差 |
| `recursive_residual` | `float` | 递推残差（标准 CG 收敛判据） |

---

## 改动 2：`EntityMPI.dot()`

```python
class EntityMPI:
    def dot(self, local_size: int):
        """返回重叠修正后的内积函数对 (dot, norm).

        用 refs() 除以每个 DOF 的共享 rank 数，再经 MPI.allreduce(MPI.SUM)
        得到全局内积，避免 overlap 副本被重复计数。
        """
        references = self.refs(local_size)
        comm = self._comm

        def _dot(x, y):
            local = bm.sum(bm.conj(x) * y / references)
            return float(comm.allreduce(float(bm.real(local)), op=MPI.SUM))

        def _norm(x):
            return max(_dot(x, x), 0.0) ** 0.5

        return _dot, _norm
```

---

## 对 SOPTX 的影响

- `examples/matrix_free_elasticity/cg.py` **已删除**。
- `analyzer.py`、`solve.py`、`tests/test_cg.py` 已改为调用 `fealpy.solver.cg`
  并注入 `dof_comm.dot(local_size)[0]` 作为 `dot_product`。
- `examples/gpu_elasticity/` 不受影响——它走 `dot_product=None` 的默认路径。
- 其他 Krylov 子空间方法（gmres、bicgstab 等）如需 MPI 支持，同上做法注入即可。

---

## 环境与版本对照

| 检出 | 状态 |
| --- | --- |
| `fealpy` (`2f17532`, `v4.0.0-alpha-15`) | CG 不支持 `dot_product`、EntityMPI 无 `dot()` |
| `fealpy_stable` (`stable` 分支) | 已实现 |
