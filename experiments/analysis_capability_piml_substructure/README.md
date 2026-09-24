# PIML 子结构分析

本目录整理 PIML 子结构方式及其精度证据, 当前 `run.py` 提供二维/三维 `linear_corner` 与 `full_trace` 下的局部样本生成、15 隐藏层网络监督训练与结构求解验证入口. 方法与配置见 [results_analysis.md](results_analysis.md).

## 模块分工

- `src/soptx/fem/substructure/independent_targets.py`: 精确内部延拓与刚度标签, 独立条目编码及可微约束补全.
- `src/soptx/ml/substructure/independent_training.py`: 网络构建、材料采样、分批数据写入和监督训练.
- `src/soptx/ml/substructure/nets.py`: 多网络形函数预测与直接刚度预测.
- `checkpoints.py`: 独立条目权重加载与配置校验.
- `analysis.py`: 同接口空间的结构求解与精确缩聚对比.
- `run.py`: 参数解析与上述流程调用.

输入是各细单元的归一化杨氏模量, 不是再次经过 SIMP 插值的设计密度. `--trace-kind` 选择 `linear_corner` 或 `full_trace` 接口空间, 默认 `linear_corner`; `--dim` 选择二维或三维, `--n-fine` 指定每个方向的细单元数 (至少为 2), 默认仍为三维、每方向 5 个单元. 子结构采用单位正方形或单位立方体, 泊松比 0.3; 二维采用 plane_stress (平面应力). 输入维度、独立输出维度及约束补全均来自有限元子结构. 两条路线均支持按网络数量连续划分独立输出, 各组大小最多相差 1; 直接刚度路线预测独立矩阵条目, 不预测 Cholesky 因子.

| `linear_corner` 配置 (每方向 5 个细单元) | 二维 | 三维 |
|---|---:|---:|
| 网络输入维度 | 25 | 125 |
| 内部位移自由度 | 32 | 192 |
| 接口位移自由度 | 8 | 24 |
| 形函数独立输出数 | 160 | 3456 |
| 刚度独立输出数 | 15 | 171 |

空间维数、划分、独立条目编号和补全方式记录在数据集及模型元数据中. 使用已有数据训练时, `--trace-kind`、`--dim` 和 `--n-fine` 必须与数据集一致. 改变接口空间、维度或划分需要分别生成标签、构建和训练网络.

## 使用

`run.py` 按构建网络、准备训练与验证样本、训练并保存模型的顺序组织调用. `--all` 执行样本生成与训练流程, 不自动执行结构求解; `--generate-samples` 仅生成样本, `--train --dataset` 构建网络后使用已有样本训练.

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
~/miniconda3/envs/ihpcm/bin/python experiments/analysis_capability_piml_substructure/run.py --train --trace-kind linear_corner --dataset <数据集目录> --route both
```

上述示例显式选择 `linear_corner`; 使用 `full_trace` 时改为 `--trace-kind full_trace`, 训练时须读取对应接口空间的数据集. 数据集元数据记录接口空间, 训练入口校验其与当前配置一致.

`--route shape` 或 `--route stiffness` 可单独训练一路. `--num-networks N` 指定每条所选路线的网络数量, 例如 `--route stiffness --num-networks 4`; 选择 `both` 时两条路线各使用 N 个网络. 未指定时保留形函数 4 个、刚度 1 个. 数量必须为正且不超过所选路线的独立输出数. 模型记录保存实际数量与输出分组. 默认训练设备为 CPU, 可显式使用 `--device cuda`. 训练默认 Adam、学习率 0.001、batch size 256、最多 500 轮、提前停止 patience 40. 验证损失连续 10 轮未改善时学习率减半, 下限 0.000001. 每条路线独立选取最佳验证权重.

输出放在 `outputs/independent_15_layer/samples/<UTC timestamp>/` 或 `outputs/independent_15_layer/training/<UTC timestamp>/`. 使用 `--all` 时顺序执行样本生成与训练, 两类输出共享时间戳; 分步入口可复用已有数据集, 均不覆盖历史目录. 实际采样下界默认 0.000001, 用于避免零刚度奇异; 该下界是实验设置, 记录于数据集元数据.

## 结构求解验证

`--analyze` 加载已有最佳权重, 不生成训练样本或更新网络. 当前采用单位子结构拼接的悬臂算例、NumPy 有限元后端与 CPU float64 网络. `--n-sub` 指定各方向子结构数, 默认二维为 `2 1`, 三维为 `2 1 1`.

```bash
# 在仓库根目录执行; 使用主工作区中已有的权重.
CHECKPOINT_DIR="/home/brighthe/workspace/soptx/experiments/analysis_capability_piml_substructure/outputs/independent_15_layer/training/20260922T065924289373Z"
~/miniconda3/envs/ihpcm/bin/python experiments/analysis_capability_piml_substructure/run.py \
  --analyze --checkpoint-dir "$CHECKPOINT_DIR" \
  --dim 3 --n-fine 5 --trace-kind linear_corner \
  --n-sub 2 1 1 --route both --seed 2026 --device cpu
```

`--route shape` 需要 `shape_best.pt`; `--route stiffness` 或 `both` 需要同一目录中的 `shape_best.pt` 和 `stiffness_best.pt`, 刚度路线使用形函数网络恢复内部位移. 网络数量从 checkpoint 恢复, 不接受 `--num-networks`. 权重的接口空间、材料、几何、离散配置和输出分组必须一致. `full_trace` 使用相同入口, 但须提供该接口空间的权重.

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
