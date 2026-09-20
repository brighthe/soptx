# 子结构 PIML GPU 实验

本模块负责子结构 PIML 各环节的 CPU/GPU 性能对照。方法实现与精度基线由
`src/soptx/` 和 `examples/piml_substructure_elasticity/` 维护，本目录不另建训练算法。

| stage | 内容 | 当前状态 |
|---|---|---|
| `data_generation` | 密度与精确标签生成 | 计划中；训练准备成本已单列 |
| `training` | 离线形函数网络训练 | 有运行入口，新基线待运行验证 |
| `inference` | 在线形函数预测 | 计划中 |
| `operator_construction` | 变分局部刚度构造 | 计划中 |
| `global_solve` | 全局接口求解 | 计划中 |
| `recovery` | 内部位移恢复 | 计划中 |
| `end_to_end` | 完整在线结构分析 | 计划中 |

## 1. 优先复现已有精度基线

既有精度入口为
[`verify_shape_function_route.py`](../../examples/piml_substructure_elasticity/verify_shape_function_route.py)。
它与 GPU 对照共享 `_shape_function.py` 的标签、形函数恢复、全批量 Adam 训练和
留出集评价；全局精度直接复用原 `step4_solution_layer`，没有第二套求解公式。

`training_full_trace_2d` 从示例的 `_common.py` 读取训练默认值：

- 完整接口 `full_trace`，单块 5×5 Q1 单元；不是角点线性迹模型。
- 2000 个训练样本和独立的 200 个留出样本，密度均匀分布于 `[0.3, 1.0]`。
- 完整接口形函数的变形子空间分量作为输出，刚体部分由构造保持。
- 两层 128 宽度 SiLU MLP，FP32，全批量 Adam，4000 epochs，学习率 0.005。
- 默认网络种子 2026。为保留历史采样口径，数据种子固定取 `SHAPE_SEED`，留出偏移 555；旧示例 `--seed` 改变网络初值，不改变这两组密度样本。

CPU 与 CUDA 共用标签、初始参数和工作量。计时包括优化器创建、前后向、参数更新及
逐轮 `loss.item()`，不含逐轮验证或选模；与旧探索 case 的计时范围不同，不可混比。
最终参数均转回 CPU，在计时之外使用同一评价器计算 raw 局部刚度误差与全局位移、
柔度和回退数量。`completed` 只表示执行完成，精度仍需审核，不能自动认定复现成功。

## 2. 运行入口

从 SOPTX 根目录运行。测试、训练和 benchmark 均需按仓库规则取得执行授权。

```bash
python experiments/piml_substructure_gpu/run.py --list
python experiments/piml_substructure_gpu/run.py --help

# 第一步：复现 examples 的完整精度验证，使用新目录保留历史结果
python examples/piml_substructure_elasticity/verify_shape_function_route.py --output-dir examples/piml_substructure_elasticity/outputs/full_trace_recheck

# 精度确认后，再执行同一基线的设备对照
python experiments/piml_substructure_gpu/run.py --pair training_full_trace_2d

# 小工作量仅验证执行链，不代表精度验收；full_trace 不接受小批量覆盖
python experiments/piml_substructure_gpu/run.py --case training_full_trace_2d_cpu --samples 32 --epochs 2 --output-dir experiments/piml_substructure_gpu/outputs/full_trace_smoke
```

`--samples` 对 full_trace 表示训练样本数，另有固定留出集；对旧探索 case 表示
训练和验证的总样本数。full_trace 的 batch size 始终等于训练样本数。CPU 线程数
取 `cases.toml`，默认 1；三个 BLAS/OpenMP 环境变量在导入数值库前设置。

## 3. 旧探索 case 与结果边界

`training_linear_corner_2d` 保留为探索性对照：低密度混合采样、随机拆分、mini-batch
训练，与申请书已有精度证据不是同一对象。其已生成 JSON 保持原样，不用新配置
覆盖或重新解释。新 full_trace case 使用独立 ID 和结果文件。

两个设备的配置、数据与初值摘要及训练步数一致后，才生成固定预算时间比；这不
代表达到同一精度的时间比。首版每端单次运行、CPU 后 CUDA、不预热，需要重复和
顺序交替才能形成稳定性能结论。没有 CUDA 时直接失败，不回退 CPU。

CPU `ru_maxrss` 是同一进程生命周期高水位，不能作为单独训练内存峰值或与 GPU
显存直接比较。CUDA 记录训练区间 allocator 峰值。结果保存到已被 Git 忽略的
`outputs/`，已有同名文件拒绝覆盖；再次运行需指定新 `--output-dir`。本实验不输出
部署 checkpoint。新基线尚未运行，不能将历史精度数字写成新模型的实测结果。

## 历史批量推理与重构脚本

`benchmark_gpu_speedup.py` 从示例目录迁入, 测量批量推理与矩阵重构, 不测离线训练。它尚未接入本目录 case 注册表, 不可将其耗时与 training case 混比。新输出默认写入本目录 outputs/, 旧输出保留原位。迁移后未运行。
