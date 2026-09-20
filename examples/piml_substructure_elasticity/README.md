# PIML 子结构力学验证 (PIML Substructure Elasticity)

本目录验证 Huang et al. (2023) 的问题无关机器学习 (problem-independent machine learning, PIML)
子结构方法在二维固定密度弹性分析中的基本性质, 并演示公共接口的调用.
脚本随代码维护; 完整论文训练、拓扑优化轨迹和性能证据由 `experiments/` 组织.

## 验证范围

| 文件 | 接口空间 | 内容 |
|---|---|---|
| [`verify_shape_function_route.py`](verify_shape_function_route.py) | `full_trace`、`linear_corner` | 刚体不变性、式 (17) 二阶误差恒等式、半正定性、受控扰动斜率, 以及留出集和固定密度 MBB 梁的代理误差. |
| [`verify_stiffness_route.py`](verify_stiffness_route.py) | 仅 `full_trace` | 刚度预测参数化自检, 留出集及 MBB 梁上的刚度、位移、柔度误差, 刚体模态伪刚度诊断. |

当前子结构为二维 $5 \times 5$ 个 Q1 单元:
`full_trace` 保留 20 个边界节点的 40 个自由度;
`linear_corner` 通过式 (16) 的 $\mathbf{L}$ 将 8 个角点自由度映射到完整边界.
形函数网络在当前刚体补空间参数化下分别输出 $32 \times 37 = 1184$ 和 $32 \times 5 = 160$ 个分量.

两个脚本均以**同接口空间的精确缩聚**为比较基准.
`linear_corner` 的误差衡量代理预测的影响, 不包含角点降阶相对细网格 FEA 的误差.
基础子结构验证入口见 [`substructure_elasticity/`](../substructure_elasticity/README.md).

## 与论文的对应边界

- 两条路线分别对应直接预测缩聚刚度、预测形函数后使用式 (17) 构造刚度的思路,
  不表示论文的网络参数化、训练配置和全部算例已复现.
- 当前形函数网络学习刚体补空间中的分量 $M$; 刚度网络学习变形子空间上的 Cholesky 条目.
  这些是当前实现的选择, 不能将输出维数与训练损失直接当作论文原始设置.
- 当前训练使用各自预测分量的 MSE, 未实现论文描述的训练末期双网络一致性损失.
- 训练和留出采样的密度范围为 $[0.3, 1.0]$, 使用 SIMP 映射;
  不等同于论文中的归一化杨氏模量随机采样, 也不证明接近空洞材料时的预测能力.
- 全局评估使用一个固定密度二维 MBB 梁, 不证明跨几何、载荷和支承的泛化,
  也不验证拓扑优化灵敏度或优化收敛.
- 刚度路线的内部位移恢复使用精确形函数, 局部计时不表示纯网络推理加速比.
  `linear_corner` 刚度预测、三维算例和完整优化不在当前覆盖范围内.

## 实现职责

子结构构造、角点投影、约束求解与全场恢复调用 `src/soptx/fem/substructure/` 公共接口.
示例保留工况、训练设置、独立变分公式及误差检查, 不维护投影或求解算法副本.

形函数路线以 `LocalReductionBatchResult` 传递预测结果.
角点预测的恢复矩阵先延拓到完整边界, 再交给公共恢复接口;
该延拓仅在 $u_b = Lq$ 上使用, 不作为完整接口预测模型.
脚本检查延拓前后的恢复关系和投影刚度一致性.

## 验收与输出

两条路线区分数学/有限性检查、网络精度验收和运行时回退门禁.
输出按“配置摘要 → 公共误差表 → 路线专项诊断 → 门禁与结果文件”组织.
公共误差使用百分数, 微小残差使用科学计数法; `--verbose` 显示详细诊断.

形函数路线的硬性数学检查包括:

| 指标 | 验收条件 |
|---|---|
| 刚体密度无关性、解析刚体分量、精确形函数代入式 (17) 的相对偏差 | 不超过 $10^{-10}$ |
| 二阶误差闭式的相对偏差 | 不超过 $10^{-7}$ |
| 误差矩阵最小特征值除以精确刚度范数 | 不低于 $-10^{-10}$ |
| 固定扰动方向的 log-log 斜率 | $[1.98, 2.02]$ |
| 预测刚体约束、恢复延拓和刚度投影的相对偏差 | 不超过 $10^{-10}$ |
| 角点约束求解的平衡与约束相对残差 | 不超过 $10^{-8}$ |

刚度路线要求刚体基相对残差不超过 $10^{-10}$、完美代理参数化相对误差不超过 $10^{-4}$,
且完美代理不得回退; 所有数值结果必须有限.

训练后还检查主要训练和解层指标的有限性.
`full_trace` 独立故障注入要求 NaN 预测触发回退、输出维数截断触发契约异常;
倍数缩放与门禁翻转扫描仅作诊断, 不假定任意网络放大后都必须回退.
上述容差是数值一致性标准, 不是论文报告的代理精度.

两个入口均支持可选精度阈值, 使用相对误差小数, 如 `0.01` 表示 1%:

| 参数 | 检查指标 |
|---|---|
| `--max-ks-error` | 留出集和在役子结构的最大刚度相对误差 |
| `--max-displacement-error` | 接口位移和全场位移相对误差 |
| `--max-compliance-error` | 柔度相对误差 |

未设置的精度项仅报告, 不宣称通过验收.
数学检查通过也不等于代理精度达标.
形函数 `--skip-train` 不执行训练、留出和全局解层, 不能同时指定精度阈值.

形函数全局解层使用原始预测, 不启用回退门禁;
`full_trace` 的留出门禁独立诊断, 其回退不混入公共误差表.
`linear_corner` 尚未接入独立门禁诊断.
刚度路线的公共误差包含实际回退结果;
`--strict` 要求留出和全局均零回退, 本身不设置精度阈值.
刚度路线一旦指定任一精度阈值, 同样要求零回退, 避免精确回退掩盖预测误差.

结果 JSON 记录验收状态与阈值; 完成评估后的验收失败会先保存证据, 再以非零状态退出.
默认输出到本目录的 `outputs/`; 需要保留多次运行时请分别指定 `--output-dir`.
输入或求解过程中提前发生异常时, 不保证产生完整结果文件.

## 运行入口

在仓库根目录、已安装 SOPTX 和本地 FEALPy 的环境中执行.
训练耗时取决于硬件、线程配置及样本数; 以下小规模训练命令用于演示流程, 不承诺精度.

```bash
# 1. full_trace 解析检查, 不训练网络
python examples/piml_substructure_elasticity/verify_shape_function_route.py --skip-train

# 2. linear_corner 解析检查, 不训练网络
python examples/piml_substructure_elasticity/verify_shape_function_route.py --trace-basis linear_corner --skip-train

# 3. 形函数路线的小规模训练和解层评估
python examples/piml_substructure_elasticity/verify_shape_function_route.py --trace-basis linear_corner --n-train 500 --epochs 500

# 4. 刚度路线的小规模训练和解层评估
python examples/piml_substructure_elasticity/verify_stiffness_route.py --train-samples 500 --epochs 500
```

完整论文复现实验见 [`experiments/topopt_simp_piml_substructure/`](../../experiments/topopt_simp_piml_substructure/).
