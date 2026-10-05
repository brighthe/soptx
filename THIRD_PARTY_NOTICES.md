# 第三方代码声明

SOPTX 自有代码采用 `GPL-3.0-only`（见 [`LICENSE`](LICENSE)）。本仓库中以下代码源自第三方
项目，按其原许可证使用。`reference_code/` 是待归档、许可证未核实的参考代码，不进入 wheel，
不在本页范围内。

## FEALPy

| 项 | 内容 |
|---|---|
| 项目 | FEALPy: Finite Element Analysis Library in Python |
| 上游 | [`suanhaitech/fealpy`](https://github.com/suanhaitech/fealpy)（原 `weihuayi/fealpy`） |
| 移植源 | vendor fork `brighthe/fealpy`（私有），分支 `main` @ `f474a5775`，含 fork 上的全部存活补丁 |
| 版权 | FEALPy is Copyright (C) Huayi Wei（FEALPy `COPYRIGHT.txt`） |
| 许可证 | GNU General Public License, version 3 or (at your option) any later version（`GPL-3.0-or-later`），与 SOPTX 的 `GPL-3.0-only` 兼容 |
| 移植时间 | 2026-10（搬入 `bf2f362`，引用切换 `b55f265`，依赖移除 `6712050`） |

SOPTX 不再依赖 FEALPy 发行版或 fork。下列代码移植后即为 SOPTX 的一部分，此后以 SOPTX
仓库为准演化，不再跟随上游；每个移植文件的头部注明了源文件路径与源提交。

### 移植范围

| SOPTX 位置 | 来源 | 说明 |
|---|---|---|
| `src/soptx/backend/` | `fealpy/backend/` | 仅 `base`、`manager`（含 `manager.pyi`）、numpy 与 pytorch 后端；logger 改为 `logging.getLogger(__name__)`，后端加载路径改为 `soptx.backend`。jax、cupy、paddle、mindspore、taichi 后端未移植 |
| `src/soptx/typing.py`、`src/soptx/decorator/` | `fealpy/typing.py`、`fealpy/decorator/` | 原样 |
| `src/soptx/sparse/`、`src/soptx/quadrature/` | `fealpy/sparse/`、`fealpy/quadrature/` | 原样 |
| `src/soptx/mesh/`（`structured_box.py`、`structured_triangle.py` 除外） | `fealpy/mesh/`（v0.4 网格树） | 只保留三角形、四边形、四面体、六面体四类经典网格的依赖闭包；未移植绘图、半边结构、局部加密与粗化、多边形等网格工厂、读写与合并；`schema/classic` 中 3 处 `fealpy.quadrature` 绝对导入改为相对导入 |
| `src/soptx/functionspace/` | `fealpy/functionspace/` 中 `space`、`function`、`dofs`、`lagrange_fe_space`、`tensor_space`、`functional`、`utils` | 2 处 `fealpy.decorator` 绝对导入改为相对导入；其余空间与 `functionspace()` 工厂未移植。同目录的 `huzhang_fe_space*.py` 是 SOPTX 自有代码，不在本行范围内 |
| `src/soptx/fem/integrator.py`、`form.py`、`_bilinear_form_base.py`、`_linear_form_base.py` | `fealpy/fem/integrator.py`、`form.py`、`bilinear_form.py`、`linear_form.py` | 两个 Form 基类改名以避开 SOPTX 同名子类 |
| `src/soptx/fem/functional.py`、`src/soptx/fem/coef.py` | `fealpy/functional.py`、`fealpy/utils/utils.py` | `coef.py` 只保留 `process_coef_func`、`is_scalar`、`is_tensor`、`fill_axis` |
| `src/soptx/solvers/cg.py`、`src/soptx/solvers/direct.py` | `fealpy/solver/cg.py` @ `40016dc56`、`fealpy/solver/direct.py` @ `30ca15599` | 早于本次移植，文件头已注明 |
| `src/soptx/fem/distributed/` 中 `mesh.py`、`entity_mpi.py`、`space.py` | `fealpy/distributed/`（`distributed_mesh.py`、`entity_mpi.py`、`distributed_space.py`） | 早于本次移植（`abd0945`），文件头未注明；来源由与 fork 逐行比对推断，相同行分别约 192/256、166/292、80/286 |
| `examples/pinn_elasticity/_pinn_support.py` | `fealpy/ml/`（`grad.py`、`sampler/`、`modules/module.py`） | 只保留示例用到的部分，修正 v0.4 下 `quadrature_formula` 的调用 |

fork 上存活补丁在 SOPTX 中的落点与移植后遗留问题见
[`docs/known-issues/README.md`](docs/known-issues/README.md)。

### FEALPy 版权声明中的其他条款

FEALPy `COPYRIGHT.txt` 在 GPL 声明之外还写有以下两段，原文照录：

> Please note that although FEALPy is freely available, it is copyrighted by the author and may not be sold or included in commercial products without a license.

> If you use FEALPy in any program or publication, please acknowledge its author by adding a reference to: **H. Wei and Y. Huang, FEALPy: Finite Element Analysis Library in Python, https://github.com/weihuayi/fealpy, Xiangtan University, 2017-2021.**
