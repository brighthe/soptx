# EA 变密度拓扑优化实验

本模块验证 EA/EbE（Element Assembly / Element-by-Element）精确 Matrix-Free 变密度拓扑优化闭环，并与 `experiments/topopt_simp_fa/` 在统一条件下对照。实验目录只负责工况注册、公共组件组装、结果验收、对照与证据汇总；有限元分析、材料插值、目标函数、体积约束、Filter 和 Optimizer 均调用 `src/soptx` 的公开接口。

与 FA 的唯一算子差异：`operator_level="ea"`，每次迭代不显式生成全局稀疏刚度矩阵，通过单元 gather、局部作用和 scatter-add 完成算子作用；求解器固定为 fealpy `cg`。

首轮范围限定为规则矩形／长方体区域。Triangle 由规则盒状网格系统剖分得到，不代表一般非结构化网格能力。

## 工况与运行

`cases.toml` 只按模型种类登记工况，一条 `[[cases]]` 对应一种「问题 + 几何 + 载荷 + 单元核」及其基准参数，当前 5 个：

| id | 网格 | 载荷路径 | 基准 filter / optimizer |
|---|---|---|---|
| `cantilever_corner_2d_concentrated` | tri 160×100 | 角点集中力 | sensitivity / OC |
| `bearing_2d_distributed` | quad 120×40 | 顶边均布面力 | sensitivity / OC |
| `half_mbb_2d_concentrated` | quad 60×20 | 左上角集中力 | sensitivity / OC |
| `cantilever_edge_3d_concentrated` | hex 30×10×2 | 右端底边集中力 | sensitivity / OC |
| `simply_supported_bridge_2d_distributed` | quad 120×40 | 顶边均布面力，桥面被动区 | sensitivity / OC |

参数变化不进注册表，一律用 `--override` 在基准上改字段。一次运行的标识 `run_id` 同时是产物目录相对 `outputs/` 的路径，分两层：第一层是工况 id，第二层是相对基准的参数标签，基准运行记 `base`：

| 命令 | 产物目录 |
|---|---|
| `run.py --case half_mbb_2d_concentrated` | `outputs/half_mbb_2d_concentrated/base/` |
| `run.py --case half_mbb_2d_concentrated --override filter_type=density` | `outputs/half_mbb_2d_concentrated/filter_type-density/` |
| `run.py --case half_mbb_2d_concentrated --override optimizer=mma filter_type=density` | `outputs/half_mbb_2d_concentrated/filter_type-density__optimizer-mma/` |

标签按字段名排序，取值取解析后的规范形式（`filter_radius=2.40` 与 `2.4` 同目录），只允许改 A/B/C/D 四轴字段，台账字段（`id`、`summary`、`fa_reference`）不可覆盖。FA 侧用同一组 override 得到同名目录。

## 对照分层

- **Tier 1（强对照）**：5 个工况的基准运行均填写 `fa_reference`，两侧同用 fealpy `cg`（`rtol=atol=1e-12`），只差算子层级。`compare.py` 执行参数一致性门禁、逐迭代轨迹对照、锁步对照（同一密度下的 u、dc、真残差）和阈值化拓扑对照。带 override 的运行用相同的 `--case ID --override ...` 指定，脚本在 FA 侧重放同一组 override 定位配对运行。
- **Tier 2（闭环能力）**：override 运行（density / projection / none 过滤、MMA 等）按 `collect.py` 验收条件独立判定 EA 闭环能力，不要求与 FA 配对。

## 目录结构

```text
experiments/topopt_simp_ea/
├── cases.toml          # 按模型种类登记的工况注册表
├── config.py           # 字段声明、校验、override 与 run_id 推导
├── pipeline.py         # soptx 公共组件组装 (operator_level 参数化)
├── run.py              # CLI 调度: 挑工况、派发给 driver
├── driver.py           # 单次运行的驱动与产物写入
├── provenance.py       # Git 与运行环境溯源
├── collect.py          # 结果验收与汇总 (枚举 outputs/ 下全部运行)
├── compare.py          # Tier 1 FA/EA 统一条件对照
├── results_analysis.md # 验收口径与结果边界
└── README.md           # 模块入口
```

`run.py` 只解析 `--list` / `--case` / `--all` / `--dry-run`，其余参数原样透传给 `driver.py`，由 `driver.py` 的 argparse 校验；`driver.py` 一次只跑一次运行，也可直接执行。

## 使用方式

```bash
cd /home/brighthe/workspace/soptx

# 列出注册工况 (基准运行)
python experiments/topopt_simp_ea/run.py --list

# 运行一个基准工况 / 全部基准工况 / 只打印派发计划
python experiments/topopt_simp_ea/run.py --case half_mbb_2d_concentrated
python experiments/topopt_simp_ea/run.py --all
python experiments/topopt_simp_ea/run.py --all --dry-run

# 在基准上改参数 (列表值用逗号); --override 只能与单个 --case 配对
python experiments/topopt_simp_ea/run.py --case half_mbb_2d_concentrated \
  --override filter_type=density
python experiments/topopt_simp_ea/run.py --case half_mbb_2d_concentrated \
  --override grid=120,40 filter_radius=4.8

# Tier 1 对照 (需要 FA 侧同名运行已完成)
python experiments/topopt_simp_ea/compare.py
python experiments/topopt_simp_ea/compare.py --case half_mbb_2d_concentrated \
  --override filter_type=density

# 汇总 outputs/ 下全部运行; 未运行的注册工况列在 pending_case_ids
python experiments/topopt_simp_ea/collect.py
```

程序不支持无参数自动运行，避免 VSCode 调试时误启动全部工况。加 `--timing` 输出优化阶段计时，`--quiet` 关闭逐迭代日志，两者可与 `--all` 并用。

每次运行的产物写入 `outputs/<run_id>/`：`summary.json`（含 `config` 参数快照与 `overrides` 原文）、`history.json`、`density_final.vtu`，以及 `vtu/` 下全部迭代密度与 `density_history.pvd` 时间序列。写入先落到同级的 `<参数标签>.partial/` 再整目录换上，不会出现半成品目录。对照结论写入 `outputs/compare/<run_id>.json`，汇总快照写入 `figure_data/ea_topopt_summary.json`；`collect.py` 在当前注册表上重放每个目录的 summary，重放不出、目录路径不符或 `config` 快照与当前注册表不一致的目录列为 `unclaimed_run_dirs`。当前工况尚未运行，不能据此声明 EA 变密度拓扑优化能力已经通过验收。
