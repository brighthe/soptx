# PIML 子结构分析

本目录整理 PIML 子结构方式及其精度证据, 当前 `run.py` 提供二维/三维 `linear_corner` 与 `full_trace` 下的局部样本生成、15 隐藏层网络监督训练与结构求解验证入口. 方法与配置见 [results_analysis.md](results_analysis.md).

## 模块分工

- `walkthrough_training.py`: 离线样本生成或复用及网络训练走查入口.
- `walkthrough_analysis.py`: 在线权重恢复与局部预测走查入口, 当前执行到两条路线的局部缩聚刚度 $K_s^j$, 全局装配、求解和恢复保留为注释.

- `src/soptx/fem/substructure/layout.py`: 整体尺寸、有限元上下文、材料场切分重排及局部到全局自由度映射.
- `src/soptx/fem/substructure/assembler.py`: 组合 `StructuredSubstructureLayout`, 负责接口、宏观和 trace 系统装配.
- `src/soptx/fem/substructure/reduction_adapter.py` 与 `recovery.py`: 统一局部缩聚结果并恢复完整位移; `GlobalAssembler` 保留旧入口转发.
- `src/soptx/fem/substructure/independent_targets.py`: 精确内部延拓与刚度标签, 独立条目编码及可微约束补全. IndependentPredictionDecoder 基于已有参考子结构构造接口空间和 codec, IndependentTargetProvider 复用该能力生成离线标签.
- `src/soptx/ml/substructure/independent_training.py`: 网络构建、单组样本生成、训练数据组织和监督训练.
- `src/soptx/ml/substructure/nets.py`: 两条路线共用的单网络 `IndependentOutputNet` 与分组网络 `SplitOutputNet`.
- `src/soptx/ml/substructure/independent_checkpoints.py`: 通用独立条目权重加载、子结构配置恢复、在线 decoder 契约比较与整体分析网络组合校验.
- `src/soptx/ml/substructure/inference.py`: CPU float64 独立条目网络推理及输出检查, 由在线走查与整体分析共同复用.
- `src/soptx/ml/substructure/validation.py`: 模型加载、批量局部预测及误差与约束指标计算.
- `local_validation.py`: 独立测试采样、核心验证调用、统计汇总与结果保存.
- `run.py`: 参数解析与流程调用, 并实现同接口空间的结构求解、精确缩聚对比及结果保存.

参考子结构 (`IndependentTargetProvider.prototype`) 是理论文档中 "几何、离散、材料参数域与接口表示相容" 条件的实现载体: 网络在其上生成标签并训练, 在线复用于所有与之相容的子结构, 即学习映射与子结构编号无关. 尺寸、网格划分、材料标度与接口空间任一项不同, 都需要对应配置的样本与权重.

输入是各细单元的归一化杨氏模量, 不是再次经过 SIMP 插值的设计密度. `--trace-kind` 选择 `linear_corner` 或 `full_trace` 接口空间, 默认 `linear_corner`; `--dim` 选择二维或三维, `--n-fine` 指定每个方向的细单元数 (至少为 2), 默认仍为三维、每方向 5 个单元. `--cell-size` 指定子结构各方向尺寸, 数量须与 `--dim` 一致, 各值须有限且为正; 未指定时采用单位正方形或单位立方体. 例如二维矩形使用 `--dim 2 --cell-size 2.0 1.0`, 三维长方体使用 `--dim 3 --cell-size 2.0 1.0 1.0`. 泊松比 0.3; 二维通过 `--hypothesis plane_stress` 或 `--hypothesis plane_strain` 选择平面应力或平面应变, 默认平面应力; 三维不接受 `--hypothesis`, 使用三维各向同性线弹性. 输入维度、独立输出维度及约束补全均来自有限元子结构. 两条路线均支持按网络数量连续划分独立输出, 各组大小最多相差 1; 直接刚度路线预测独立矩阵条目, 不预测 Cholesky 因子.

| `linear_corner` 配置 (每方向 5 个细单元) | 二维 | 三维 |
|---|---:|---:|
| 网络输入维度 | 25 | 125 |
| 内部位移自由度 | 32 | 192 |
| 接口位移自由度 | 8 | 24 |
| 形函数独立输出数 | 160 | 3456 |
| 刚度独立输出数 | 15 | 171 |

空间维数、划分、独立条目编号和补全方式记录在数据集及模型元数据中. 使用已有数据训练时, 从 `manifest.json` 自动恢复接口空间、尺寸、各方向网格划分、泊松比及材料假设, 不接受重复指定 `--trace-kind`、`--dim`、`--n-fine`、`--cell-size` 或 `--hypothesis`. 改变接口空间、维度、划分、尺寸或材料假设需要使用对应配置的标签和网络权重.

## 使用

`run.py` 按构建网络、准备训练与验证样本、训练并保存模型的顺序组织调用. `--all` 执行样本生成与训练流程, 不自动执行结构求解; `--generate-samples` 仅生成样本, `--train --dataset` 先读取数据集配置并恢复 provider, 再构建网络并训练.

以下命令在仓库根目录的 WSL 环境执行.

```bash
# 查看入口与参数, 不生成样本.
~/miniconda3/envs/ihpcm/bin/python experiments/analysis_capability_piml_substructure/run.py --help

# 三维: 依次构建网络、生成样本并训练两条路线.
~/miniconda3/envs/ihpcm/bin/python experiments/analysis_capability_piml_substructure/run.py --all --trace-kind linear_corner --dim 3 --n-fine 5

# 二维: 使用相同流程.
~/miniconda3/envs/ihpcm/bin/python experiments/analysis_capability_piml_substructure/run.py --all --trace-kind linear_corner --dim 2 --n-fine 5

# 分批生成 400,000 个训练样本及 40,000 个独立验证样本.
~/miniconda3/envs/ihpcm/bin/python experiments/analysis_capability_piml_substructure/run.py --generate-samples --trace-kind linear_corner

# 使用上一步输出的数据集目录, 分别训练两条路线.
~/miniconda3/envs/ihpcm/bin/python experiments/analysis_capability_piml_substructure/run.py --train --dataset <数据集目录> --route both
```

生成样本与 `--all` 示例显式选择 `linear_corner`; 使用 `full_trace` 时改为 `--trace-kind full_trace`. 单独训练只需选择对应数据集, 入口恢复子结构配置并校验独立分量编号.

`--route shape` 或 `--route stiffness` 可单独训练一路. `--num-networks N` 指定每条所选路线的网络数量, 例如 `--route stiffness --num-networks 4`; 选择 `both` 时两条路线各使用 N 个网络. 未指定时保留形函数 4 个、刚度 1 个. 数量必须为正且不超过所选路线的独立输出数. 模型记录保存实际数量与输出分组. 默认训练设备为 CPU, 可显式使用 `--device cuda`. 通过 `--optimizer adam|adamw|sgd` 选择优化器, 默认 Adam; `--lr` 指定学习率, `--weight-decay` 指定权重衰减, `--momentum` 指定 SGD 动量 (非零值仅适用于 SGD). 默认学习率 0.001、权重衰减 0、动量 0、batch size 256、最多 500 轮、提前停止 patience 40. 切换优化器时应相应设置学习率. Python 接口使用 `TrainingConfig(optimizer="adam", optimizer_params={"lr": 1e-3, "weight_decay": 0.0}, ...)`; SGD 的 `momentum` 放入同一字典. 命令行选项统一转换为 `optimizer_params`, 不再使用顶层 `learning_rate`、`weight_decay` 或 `momentum` 配置字段. 优化器配置随训练记录与最佳权重保存. 验证损失连续 10 轮未改善时学习率减半, 下限 0.000001. 每条路线独立选取最佳验证权重.

输出放在 `outputs/independent_15_layer/samples/<UTC timestamp>/` 或 `outputs/independent_15_layer/training/<UTC timestamp>/`. 使用 `--all` 时顺序执行样本生成与训练, 两类输出共享时间戳; 分步入口可复用已有数据集, 均不覆盖历史目录. 实际采样下界默认 0.000001, 用于避免零刚度奇异; 该下界是实验设置, 记录于数据集元数据.

## 局部预测验证

统一入口为 `run.py --validate-local`, 必须提供 `--checkpoint`, 预测路线从权重内部的 `route` 元数据读取, 不接受 `--route`. 接口空间、尺寸、网格划分、泊松比、材料假设及网络结构从权重恢复, 不接受重复指定 `--dim`、`--cell-size`、`--n-fine`、`--trace-kind`、`--hypothesis` 或 `--num-networks`. 独立测试材料输入重新采样, 不读取训练集或验证集数组.

```bash
# 在 experiments/analysis_capability_piml_substructure 目录运行.
TRAINING_DIR="$HOME/codespace/data/soptx/piml_substructure/independent_15_layer/training/20260922T065924289373Z"
# 预测路线与子结构配置从权重元数据自动恢复，重新生成独立测试样本。
python run.py --validate-local \
  --checkpoint "$TRAINING_DIR/shape_best.pt" \
  --n-test 1000 --test-batch-size 32 --min-modulus 1e-6 \
  --seed 2027 --device cpu
```

`--test-batch-size` 与训练的 `--batch-size` 分别设置. 小样本运行检查可设为 `--n-test 8 --test-batch-size 2`. 默认结果写入 `outputs/independent_15_layer/local_validation/<时间戳>_<路线>/`; `--output-dir` 指定输出根目录, 与其他执行入口保持一致.

`src/soptx/ml/substructure/validation.py` 提供 `load_local_model()` 与 `LocalPredictionEvaluator`; 实验模块 `local_validation.py` 提供 `evaluate_local_predictions()`, 调用示例见 [results_analysis.md 第五节](results_analysis.md#5-局部预测验证). 从仓库根目录以 `experiments.analysis_capability_piml_substructure.local_validation` 导入. 每次只验证 shape 或 stiffness 一条路线; 刚度局部验证只需 stiffness_best.pt, 不依赖形函数权重. 整体分析的加载要求保持不变.

局部参考复用 `IndependentTargetProvider.exact_matrices()`. 形函数路线同时评价内部形函数与变分重构刚度, 刚度路线评价直接预测刚度; 两者记录对称性、刚体约束残差和变形子空间最小特征值. 使用测试随机流 `SeedSequence(seed, spawn_key=(2,))`, 与训练及验证使用的子流分开. 每次写入新目录, 不覆盖历史结果. 支持 CPU 或 CUDA 网络推理, 精确参考由有限元后端计算. 当前未运行数值验证, 不将实现接入视为精度验收通过.

## 离线训练与在线分析走查

`walkthrough_training.py` 展示离线阶段. 指定 `--generate-samples` 时, 由 `--dim`、局部问题配置和样本参数生成样本后训练; 默认三维、每方向 5 个细单元、400,000 个训练样本、40,000 个验证样本及 500 轮训练. 默认复用 `~/codespace/data/soptx/piml_substructure/independent_15_layer/samples/20260922T065924289373Z`, 可通过 `--samples-dir` 更换目录; 从 `manifest.json` 恢复空间维数、尺寸、细划分、泊松比、材料假设、接口空间和独立分量编号, 并拒绝重复指定这些局部问题及样本生成参数. `--route shape` 只训练形函数网络; `--route stiffness` 同时训练刚度网络与内部位移恢复所需的形函数网络.

`walkthrough_analysis.py` 展示在线阶段. `--shape-dir` 指定 `shape_best.pt` 所在目录; `--route stiffness` 还需用 `--stiffness-dir` 指定 `stiffness_best.pt` 所在目录. 两个选项默认指向 `~/codespace/data/soptx/piml_substructure/independent_15_layer/training/20260922T065924289373Z`. 脚本从权重恢复局部问题配置和网络, 通过 `StructuredSubstructureLayout` 创建整体有限元布局, 由 `build_modulus_substructures` 创建按归一化模量线性装配的公共参考子结构, 再基于同一 `prototype` 构造 `IndependentPredictionDecoder`; decoder 的离散配置、独立分量编号和数值基须与权重元数据相容.

在线流程按“密度场 -> $E$ 的 SIMP 插值 -> FE cell 顺序的归一化模量 -> 网络预测 -> $K_s^j$”组织. `--E-simp-penalty` 设置 $E$ 的 SIMP 指数, 默认 3; 空材料杨氏模量显式取 0, 与精确走查的 `rho_min=0` 对齐. `--local-batch-size` 默认 32, 同时限制网络推理批量及 shape 路线的局部刚度装配批量. shape 路线每批先预测并补全内部延拓, 再装配当前批的 $K^j$ 并立即计算 $(N^j)^T K^j N^j$, 不保存全量细网格刚度或内部延拓. stiffness 路线直接补全预测的 $K_s^j$, 不装配细网格 $K^j$; `shape_best.pt` 仍为后续内部位移恢复所需, 但当前阶段不执行 shape 推理. 当前可执行代码止于 `local_condensed_stiffness`; 全局接口装配、求解和内部位移恢复仍保留为未启用草稿, 不作为已执行或已通过数值验证的流程.

整体求解域通过 `--domain` 指定, 默认 `0 78 0 13 0 13`; 各方向下端须为 0. `--n-sub` 默认 `78 13 13`. 每方向域长除以子结构数须与权重 `cell_size` 一致, 否则在加载网络前报错. 二维权重须同时指定二维 `--domain` 和 `--n-sub`.

以下命令在仓库根目录运行, 当前 Python 环境须已安装此工作区的 `soptx`:

```bash
DATA_DIR="$HOME/codespace/data/soptx/piml_substructure/independent_15_layer"

# 从头生成三维样本并训练形函数路线:
python -m experiments.analysis_capability_piml_substructure.walkthrough_training \
  --generate-samples --dim 3 --route shape

# 复用正式样本, 重新训练直接刚度及恢复所需的形函数网络:
python -m experiments.analysis_capability_piml_substructure.walkthrough_training \
  --samples-dir "$DATA_DIR/samples/20260922T065924289373Z" \
  --route stiffness

# 加载训练结果并执行在线走查:
python -m experiments.analysis_capability_piml_substructure.walkthrough_analysis \
  --shape-dir "$DATA_DIR/training/20260922T065924289373Z" \
  --domain 0 2 0 1 0 1 --n-sub 2 1 1 --route shape --seed 0
```

新产物写入 `<outputs-root>/independent_15_layer/{samples,training}/<UTC 时间戳>/`, `--outputs-root` 默认为仓库外的 `~/codespace/data/soptx/piml_substructure/`. 相对路径均以对应脚本目录为基准. 正式默认样本规模约占 13 GB; 演示规模须显式设置 `--n-train`、`--n-validation`、`--epochs` 并将 `--outputs-root` 指向临时目录. 文件清单与 sha256 见数据根目录下的 `SHA256SUMS`, 在该目录运行 `sha256sum -c SHA256SUMS` 校验; 重新生成命令见上方代码块.

`walkthrough_analysis.py` 的 `--mem-limit-gb` 限制进程虚拟地址空间, 默认 35 GiB; `--local-batch-size` 只限制当前网络输入批次以及 shape 路线当前批次的细网格 $K^j$ 和内部延拓. 脚本仍保留全部 $K_s^j$ 供后续全局装配使用, `full_trace` 或大量子结构下该数组仍可能成为主要内存开销. 此限制不会自动选择可行的批量或缩减 $K_s^j$.

## 结构求解验证

整体分析实现位于 `run.py` 的 `run_analysis()` 及同文件辅助函数中. 调试时以 `run.py --analyze` 为入口, 工作目录设为本实验目录, 直接在 `run_analysis()` 中设置断点; 装配、求解和恢复对应 `_assemble()`、`_solve()`、`_recover()`.

`--analyze` 加载已有最佳权重, 不生成训练样本或更新网络. 从 `shape_best.pt` 恢复子结构尺寸、网格划分、泊松比、材料假设和接口空间, 再拼接悬臂算例; 使用 NumPy 有限元后端与 CPU float64 网络. 不接受重复指定 `--dim`、`--cell-size`、`--n-fine`、`--trace-kind` 或 `--hypothesis`. `--n-sub` 指定各方向子结构数, 根据恢复的空间维数, 默认二维为 `2 1`, 三维为 `2 1 1`.

```bash
# 在 experiments/analysis_capability_piml_substructure 目录运行.
TRAINING_DIR="outputs/independent_15_layer/training/<时间戳>"
python run.py --analyze \
  --checkpoint-dir "$TRAINING_DIR" \
  --n-sub 78 13 13 --route shape --seed 2026 --device cpu
```

整体分析每次只接受 `shape` 或 `stiffness`, 默认 `shape`, 拒绝 `both`; 比较两条路线时分别运行并使用相同的 `--seed` 和 `--n-sub`. `--route shape` 需要 `shape_best.pt`; `--route stiffness` 需要同一目录中的 `shape_best.pt` 和 `stiffness_best.pt`, 刚度路线使用形函数网络恢复内部位移. 网络数量从 checkpoint 恢复, 不接受 `--num-networks`. 每份权重均校验其网络结构与输出分组; 两份权重的接口空间、材料、几何、离散配置和独立分量编号必须一致, 不要求网络数量或训练轮次相同. `full_trace` 使用相同入口, 但须提供该接口空间的权重.

输出位于 `outputs/independent_15_layer/analysis/<UTC timestamp>/`, 保存配置、权重路径与 SHA256、材料输入、位移和误差汇总. 精确基准采用相同接口空间、材料、载荷和约束. 验收分别检查局部刚度、接口位移、内部位移和柔度误差, 以及自由接口平衡残差. 小残差不代表代理预测准确, 当前不预设代理精度通过阈值.

预测局部刚度在变形子空间上非正定时, 对应路线记录为失败, 不回退到精确刚度. 任一路线失败时入口以非零状态退出, 保留 summary.json 中的原因. COMPLETED 仅表示流程完成, 不代表通过代理精度验收. 材料场由 `--seed` 固定, 每个细单元的归一化杨氏模量在 $[10^{-6}, 1)$ 内独立均匀采样; 左端固支, 自由端角节点沿最后一个坐标方向施加单位负向力. 本入口尚待数值验证, 上述命令不代表已有运行结果.

## 小规模接通检查

正式生成全量数据前, 可先使用以下配置检查数据与训练流程. 这些命令会生成少量样本并更新网络权重, 其结果只用于接通检查.

```bash
~/miniconda3/envs/ihpcm/bin/python experiments/analysis_capability_piml_substructure/run.py --all --trace-kind linear_corner --dim 2 --n-fine 5 --n-train 8 --n-validation 4 --generation-batch-size 2 --epochs 1 --batch-size 2
~/miniconda3/envs/ihpcm/bin/python experiments/analysis_capability_piml_substructure/run.py --all --trace-kind linear_corner --dim 3 --n-fine 5 --n-train 8 --n-validation 4 --generation-batch-size 2 --epochs 1 --batch-size 2
```

## 验收与边界

样本生成应得到完整的数据集记录, 输入与标签均为有限值, 维度符合两条路线的约定. 训练应输出每轮有限的训练/验证损失, 保存最佳模型和训练记录. 验证集仅用于选模与调整学习率, 不充当独立测试结果.

训练入口不启用后期一致性损失; `--analyze` 单独执行固定材料结构求解验证, 不执行拓扑优化. 输出连续分组与独立条目消元是明确的实验实现选择, 不作为论文原始索引划分的复现声明.

`cases.toml`、`config.py`、`collect.py`、`provenance.py`、`figure_data/` 及历史 `outputs/` 保留既有工况与图件证据; 当前入口不提供原来的 `--list`、`--case`、`--task`、`--collect` 或 `--check-network` 命令. 精确缩聚基准归 [../analysis_capability_substructure/](../analysis_capability_substructure/), GPU 性能归 [../piml_substructure_gpu/](../piml_substructure_gpu/).

## 迁入的研究材料

- `legacy_examples_results.md`: 原示例报告的历史归档.
- `collect_ood_probe_trajectory.py`: OOD 密度轨迹采集.
- `prototypes/plot_local_recovery.py`: 合成预测场的云图版式原型, 不作为力学精度证据.

单网络统一使用 `IndependentOutputNet`, 多网络使用 `SplitOutputNet`. 旧 `DirectStiffnessNet` 类名保留为兼容入口; 分析加载器接受旧刚度单网络权重, 并在校验输出分组和排序后转换旧形函数单网络的包装层权重键.
