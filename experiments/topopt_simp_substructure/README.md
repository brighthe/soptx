# topopt_simp_substructure — 精确子结构缩聚变密度拓扑优化

精确局部 Schur 缩聚下的 SIMP/OC 拓扑优化闭环，**不含任何 PIML 近似**。
唯一的算法轴是接口迹空间 `trace`：

| trace | 含义 | 相对 FA 的误差来源 |
| --- | --- | --- |
| `full_trace` | 保留全部接口自由度（`T = I`） | 无方法误差，代数等价，只剩实现误差与舍入 |
| `linear_corner` | 角点线性迹（Huang 2023 式 16） | 迹降阶误差 |

误差阶梯：**FA → `full_trace` → `linear_corner` → PIML 局部代理**。
本目录负责中间两级，PIML 两条路线由 `experiments/piml_substructure_topopt/` 负责。
`full_trace` 是整条链路的实现门禁：它与 FA 求解的是同一个离散系统，同工况下
逐迭代柔度、体积分数与最终拓扑应一致到舍入精度；一旦对不上，问题必定在
缩聚—恢复—伴随的实现里，与迹降阶和代理模型无关。

## 目录结构

```
topopt_simp_substructure/
├── README.md              本文件
├── cases.toml             工况注册表（唯一参数入口，全参数显式写全）
├── config.py              注册表加载与校验
├── pipeline.py            组件组装与正问题求解（装配/缩聚/求解/恢复/单元应变能）
├── run.py                 优化循环（SIMP + 锥形灵敏度滤波 + OC）与落盘
├── collect.py             跨工况结果汇总
├── compare.py             与 experiments/topopt_simp_fa 的同工况对照
├── provenance.py          运行溯源采集
├── results_analysis.md    结果与验收结论
└── outputs/<case-id>/     history.json / summary.json / density_final.npy /
                           density_final.vtu / topology.png(2D) / vtu/
```

## 使用方式

```bash
python run.py --list
python run.py --case mbb_2d_full_trace
python run.py --case mbb_3d_full_trace
python run.py --case mbb_2d_linear_corner
python run.py --case mbb_3d_linear_corner
python collect.py
python compare.py --case mbb_2d_full_trace     # 需先在 FA 侧注册同工况
```

`--max-iter` / `--volfrac` 可临时覆盖注册表取值（会记入 `summary.json` 的
`overrides` 字段），`--output-dir` 可改写输出根目录。

## 与相邻实验目录的关系

- **`experiments/topopt_simp_fa/`**：参照侧。`cases.toml` 的 `fa_reference`
  指向那里的同工况 id，`compare.py` 以它为基准做三阶段对照。
  **当前 FA 侧尚未注册 MBB 工况**，因此四条工况的 `fa_reference` 暂留空，
  `compare.py` 在字段为空时直接报错退出，不做静默降级。补齐 FA 侧 MBB 工况时
  需注意：本目录 OC 固定 `design_variable_min = 1e-3`，FA 侧同工况必须以
  `density_min = 1.0e-3` 注册，否则两条轨迹不可比（`compare.py` 的门禁会拦下）。
- **`experiments/piml_substructure_topopt/`**：下游。那里的 PIML 路线 A/B 与
  FEA 基线共用同一套 `linear_corner` 缩聚设置；本目录已注册的 `linear_corner`
  工况作为其精确参照。两个目录之间没有代码依赖，本目录不 import torch。

## 约定

- 优化循环参数与 `piml_substructure_topopt` 的 FEA 基线逐项一致：
  `move = 0.2`、`damping = 0.5`、`initial_lambda = 1e9`、`bisection_tol = 1e-4`、
  `design_variable_min = 1e-3`，收敛判据为连续 5 步 `|dC|/C < tol_change` 且
  `it >= 10`。改这些参数会同时破坏与 FA 侧和 PIML 侧的可比性。
- `full_trace` 要求载荷与位移约束全部落在子结构边界自由度上；落进子结构内部时
  `pipeline.py` 直接报错，不做静默丢弃。
- `reduction` 目前只接受 `exact_schur`：本目录不承载任何近似缩聚。
