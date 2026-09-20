# PIML 子结构拓扑优化实验 (PIML Substructure Topology Optimization)

本实验模块在 2D MBB 梁基准上，完整打通并验证 **基于 PIML 局部力学表示的多步迭代结构拓扑优化闭环**。

---

## 核心目标

1. **复现完整拓扑优化闭环**：基于 Huang 2023 论文式 (17) 变分能量构造局部刚度，并在细观尺度恢复位移场 $\boldsymbol{u}_h = [\boldsymbol{u}_b; \boldsymbol{u}_i]$，执行 SIMP 灵敏度分析与优化准则 (OC) 更新；
2. **闭环收敛性与保真度验证**：与精确子结构有限元 (Schur 补) 拓扑优化基线进行 1-to-1 全流程对比，验证 PIML 在多步迭代下不发散、不崩溃，并以最终柔度相对误差 $<3\%$、fallback rate $<10\%$ 作为当前验收目标；
3. **输出图件与证据**：落盘收敛历程、最终拓扑构型及迭代耗时对比。

---

## 神经网络架构与复现参考

详见分析报告：[`results_analysis.md`](./results_analysis.md#12-huang-2023-神经网络配置矩阵与训练协议)
- **5 组网络配置矩阵**：覆盖 2D/3D、$m=5, 10$、精确边界迹与角点线性迹（含 3D 组 4 的 4 并联 15 层 MLP 结构）；
- **预测路线与物理损失**：路线 A（式 17 变分能量构造 + 细观位移恢复）与路线 B 机制，式 (17)/(18) 联合损失函数；
- **数据集生成与训练协议**：高斯随机场 (GRF) 合成数据与真实拓扑优化轨迹数据混合防 OOD 策略。

---

## 快速使用

### 1. 生成相互隔离的 Exact 训练与验证轨迹

```bash
python experiments/topopt_simp_piml_substructure/run.py \
  --case mbb_fea_baseline --volfrac 0.4 \
  --save-density-trajectory \
  --output-dir experiments/topopt_simp_piml_substructure/outputs/route_a_mixed_v2/trajectories/v040
python experiments/topopt_simp_piml_substructure/run.py \
  --case mbb_fea_baseline --volfrac 0.6 \
  --save-density-trajectory \
  --output-dir experiments/topopt_simp_piml_substructure/outputs/route_a_mixed_v2/trajectories/v060
```

`--save-density-trajectory` 只允许 Exact `fea_baseline` 工况，并保存每轮实际
参与 FE 求解的 pre-OC 局部密度。训练与验证按整条 trajectory 隔离，不能把
同一轨迹的不同迭代或子结构随机拆入两侧。`volfrac=0.5` canonical 工况不进入
训练或选模，只用于最终回归；由于此前已观察过该工况，它不属于严格 blind test。

### 2. 离线训练 Route A checkpoint

```bash
python experiments/topopt_simp_piml_substructure/train_route_a.py \
  --case mbb_piml_route_a \
  --training-trajectory experiments/topopt_simp_piml_substructure/outputs/route_a_mixed_v2/trajectories/v040/mbb_fea_baseline_trajectory.npz \
  --validation-trajectory experiments/topopt_simp_piml_substructure/outputs/route_a_mixed_v2/trajectories/v060/mbb_fea_baseline_trajectory.npz \
  --refit-all-trajectories \
  --output-path experiments/topopt_simp_piml_substructure/outputs/mbb_piml_route_a_checkpoint.pt
```

训练入口采用 synthetic/trajectory 平衡采样，由独立验证轨迹按来源统计 gate pass
与 fallback 前 raw 式 (17) 刚度误差。所有来源均单独报告；checkpoint 只按独立
validation trajectory 的最差 `p95` 刚度误差与最低 gate pass 组成的 physics
score 选取，避免 synthetic 极端分布或样本数劫持选模。启用
`--refit-all-trajectories` 后，以该阶段选出的 `best_epoch` 为固定轮数，使用两条
轨迹从头 refit，不再读取 canonical 或二次调参。checkpoint 记录轨迹 SHA256、
数据角色、选择阶段和 refit 阶段。

### 3. 运行 PIML 路线 A

```bash
python experiments/topopt_simp_piml_substructure/run.py \
  --case mbb_piml_route_a \
  --weight-path experiments/topopt_simp_piml_substructure/outputs/mbb_piml_route_a_checkpoint.pt
```

`run.py` 只加载并校验 checkpoint，不会在拓扑优化过程中隐式训练。

### 4. 诊断 gate-accepted 局部恢复的灵敏度误差

```bash
python experiments/topopt_simp_piml_substructure/run.py \
  --case mbb_piml_route_a \
  --weight-path experiments/topopt_simp_piml_substructure/outputs/mbb_piml_route_a_checkpoint.pt \
  --audit-piml-sensitivity \
  --audit-max-substructures 256 \
  --output-dir experiments/topopt_simp_piml_substructure/outputs/sensitivity_audit
```

audit 只处理通过现有 gate 的异质子结构：在同一个 PIML 全局接口位移迹下，
用 Exact Schur 恢复内部位移，并比较局部位移、单位刚度二次型、SIMP 灵敏度和
过滤后灵敏度。它不会修改 gate、正式梯度或 OC 更新，也不会重新求解 Exact 全局
接口系统，因此只能隔离局部恢复误差，不能直接当作新的 pre-solve gate。

报告写入 `<case>_sensitivity_audit.json`，记录 checkpoint SHA256、工况配置、
逐迭代计数及逐子结构密度/gate/误差指标。为避免大规模 3D 工况产生不可控的
Exact 批量与 JSON，默认每轮最多确定性抽取 256 个 accepted 子结构；只有覆盖
全部 accepted 子结构时才报告完整 hybrid filtered-sensitivity 指标。

### 5. 过渡使用旧裸权重

```bash
python experiments/topopt_simp_piml_substructure/run.py \
  --case mbb_piml_route_a \
  --weight-path experiments/topopt_simp_piml_substructure/outputs/mbb_piml_route_a_surrogate_net.pt \
  --allow-legacy-weight
```

旧裸 `state_dict` 不含物理和采样签名，只允许通过显式开关用于历史回归。

checkpoint 除物理签名外还登记网络架构（`hidden_dims` 与激活名），加载时与
`model_factory` 构造出的网络比对，不一致直接报错。激活不含参数，配错不会触发
权重形状错误，这层登记正是为堵住这一静默失配。本改动之前产出的 checkpoint 没有
该字段，加载时降级为 `RuntimeWarning` 并继续，架构须自行确认；重新训练即可补齐。

### 6. 运行精确 FEA 基线

```bash
python experiments/topopt_simp_piml_substructure/run.py --case mbb_fea_baseline
```

### 7. 汇总指标与生成收敛对比图

```bash
python experiments/topopt_simp_piml_substructure/collect.py
```
