# 传统有限元拓扑优化: EA 与 FA 对照

本目录验证传统有限元 (Q1 六面体, Lagrange 位移元) 下 EA 与 FA 两种刚度算子层级的拓扑优化结果一致性与计算代价。两条流程除算子层级外完全相同, 由同一脚本以 `--level` 切换, 算例为 Huang2023 的三维 MBB 梁首档。

| 文件 | 内容 |
|---|---|
| [`run_fem_ea_vs_fa.py`](run_fem_ea_vs_fa.py) | 运行入口, `--level ea` 为一份参考单元刚度乘逐单元系数、不形成全局矩阵, `--level fa` 为全局 CSR 稀疏矩阵 |
| [`fem_ea_vs_fa.md`](fem_ea_vs_fa.md) | EA 与 FA 公共实验参数、结果一致性与耗时、内存对照 |
| [`sync_results.py`](sync_results.py) | 把选定的两侧运行从 `outputs/` 固化到 `results/` |
| [`results/`](results/) | 入库的结果, `fem_ea_vs_fa.md` 的数据来源: 两侧 `config.json`, `summary.json`, `history.json` 与末轮场比对 `final_fields.json` |

公共参数统一见 [实验设置](fem_ea_vs_fa.md#2-实验设置)。

当前运行入口统一采用 `rtol=CG_TOLERANCE`、`atol=0`、`rtol_reference='rhs'`、
`norm_type='unpreconditioned'`、`verify_residual=True`, 由 CG 根据施加边界条件后的实际
右端项计算停机阈值, 不再手算 `load_norm`. EA 与 FA 共用这套设置.
已有结果保留生成时的配置; 本次容差配置更新尚未运行测试或重新生成对照结果.

## 运行

在仓库根目录、已安装 SOPTX 的 Python 环境中执行, 两侧各为一次独立运行:

```bash
python experiments/topopt_simp_fem/run_fem_ea_vs_fa.py --level ea
python experiments/topopt_simp_fem/run_fem_ea_vs_fa.py --level fa
```

线程固定为单线程: 脚本顶部的 `NUM_THREADS = 1` 在导入 numpy/scipy 之前写入全部 BLAS / OpenMP 线程变量, 覆盖 shell 中的已有设置, 实际取值与已加载的 BLAS 库记入 `config.json` 的 `threads`。EA 的单元作用经由 BLAS, FA 的稀疏矩阵向量积与 `bincount` 不经由 BLAS, 多线程会让对照混入库的线程支持差异; 且不设置时 NumPy 链接的 BLIS 默认单线程, SciPy 内置的 OpenBLAS 默认用满全部核, 口径不一。

每次运行只跑一侧: 内存按进程 RSS 记录, 两侧同进程连跑会互相污染。命令行有必填的 `--level` 与可选的 `--output-dir`, 其余设置 (后端与设备、网格、MGCG 选项、优化迭代上限与 VTU 场) 均为脚本顶部常量, 改设置须改常量。

结果默认写入本目录的 `outputs/{ea,fa}_hex_390x65x65_cg/`, `--output-dir` 可改输出根目录 (子目录名不变)。`outputs/` 是不入库的临时目录, 输出目录已存在时脚本直接清空再重写; 需要保留的结果先用 `sync_results.py` 固化到 `results/` 或自行备份。

## 输出与结果固化

运行脚本把结果写入 `outputs/{ea,fa}_hex_390x65x65_cg/` (不入库), 再由 `sync_results.py` 固化到 `results/`。

在仓库根目录执行 `python experiments/topopt_simp_fem/sync_results.py` 固化默认运行；可用 `--ea`、`--fa` 与 `--results-dir` 指定来源与目标。固化两侧 JSON 和末轮场比对统计，NPY 本身不入库。

## ParaView 帧

每轮帧 `iterations/iter_XXXX.vtu` 与 `evolution.pvd` 体积大，运行时先写在 WSL 的结果目录，第 0 帧 `iter_0000.vtu` 是初始设计经过滤与实体保留单元处理后的物理密度 (只含 `density`)，在计时之前写出，便于查看固定单元；由 `soptx.postprocess.iteration_frames.IterationFrameWriter` 原子写出，跨文件系统写入不计入逐轮耗时。运行结束 (`summary.json` 写出) 后，脚本自动把帧迁移到脚本顶部 `VIEW_ROOT` (`/mnt/c/workspace/soptx-results/topopt_simp_fem`) 下的同名子目录，在 Windows ParaView 中打开其中的 `evolution.pvd` 即可；WSL 的结果目录只保留 json、npy 与末帧、初始帧的副本 `result_final.vtu`、`result_initial.vtu`，迁移位置记在 `visualization_location.json`。

- 查看目录已存在时先清空再迁移，与 `outputs/` 重跑即清空的规则一致；其中出现 `iterations/`、`evolution.pvd`、`.visualization_sync.json` 以外的条目时拒绝清空并跳过迁移。
- `/mnt/c/workspace/soptx-results` 不存在或迁移失败时只打印提示，帧留在 WSL 的结果目录，运行不报错。
- 运行中断或迁移被跳过时，可在本实验目录下手动补迁，例如：

```bash
python -m soptx.postprocess.sync_visualization --source-dir outputs/fa_hex_390x65x65_cg --destination-dir /mnt/c/workspace/soptx-results/topopt_simp_fem/fa_hex_390x65x65_cg --move
```

`--move` 要求结果目录已有 `summary.json`；中断的运行没有该文件，此时去掉 `--move` 只复制，确认后自行删除 WSL 中的帧。
