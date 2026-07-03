# AI Agent 说明

本仓库采用与 `dut-postdoc` 一致的 AI 上下文组织方式。

人机协作方式（含「程序实现按子任务确认执行」）统一见 `ai/common/collaboration-conventions.md`（SSOT）；本文件不复写。

新开 AI/Codex 窗口续接工作前，优先阅读 hub：

```text
ai/common/status.md
```

然后在工作线表中定位目标工作线。继续 SOPTX Matrix-Free 工作时，阅读：

```text
ai/common/progress-frame8_matrix_free.md
```

Matrix-Free progress 和架构备忘录会指向三个主要文档：

```text
C:\workspace\dut-postdoc\research\postdoc-plan\defense-sprint\direction-1-piml-matrix-free\frame8_matrix_free_pipeline_guide.md
C:\workspace\soptx_heliang\docs\matrix_free_architecture_notes.md
```

（原 dut-postdoc 的 `soptx-matrix-free-integration-plan.md` 与 `matrix_free_math_principles.md`
已于 2026-07-02 被帧 8 guide 取代删除。）

默认工作判断：

1. `dut-postdoc` 保存研究计划、数学原则和总体路线。
2. `soptx_heliang` 保存当前代码实现、测试、接口决策和分支相关说明。
3. Matrix-Free 应通过 `LagrangeFEMAnalyzer` 的状态方程求解后端接入 SOPTX。
4. `LinearElasticIntegrator` 只消费 `coef` / `relative_stiffness`，不解释原始拓扑优化密度 `rho`。
5. assembled 与 matrix-free 的一致性测试是当前实现的安全网，修改相关代码后需要优先运行。
6. 当前 Matrix-Free `action()` 已在 2D/3D `standard` 路径使用积分点 contraction；未覆盖情况仍保留局部 `Ke @ xe` fallback，后续需要继续扩展多分辨率、更多后端/缓存策略和性能验证。

## Python 虚拟环境约定

本仓库开发和测试默认使用本地虚拟环境：

```text
C:\workspace\soptx_heliang\.venv
```

运行测试时优先使用：

```powershell
.\.venv\Scripts\python.exe -m pytest ...
```

不要直接使用系统 Python 或全局 pip。安装依赖时也应使用：

```powershell
.\.venv\Scripts\python.exe -m pip ...
```

当前 `.venv` 的 `include-system-site-packages = true`，因此它可能读取 Codex runtime 基础环境中的部分包。当前 FEALPy 通过本地源码路径接入：

```text
C:\workspace\fealpy_heliang
```

后续如需确认当前解释器和包来源，可在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -c "import sys, fealpy, soptx; print(sys.executable); print(fealpy.__file__); print(soptx.__file__)"
```

## 讨论沉淀提醒

每次完成一轮重要讨论、架构判断、验证结论或代码阶段推进后，AI agent 应主动提醒用户：

```text
是否需要把本轮讨论中的重要结论沉淀到 ai/common/progress-frame8_matrix_free.md 或 docs/matrix_free_architecture_notes.md？
```

默认判断：

1. 如果是“长期有效的上下文入口、当前进度、关键默认假设”，优先沉淀到对应工作线 progress 文档，例如 `ai/common/progress-frame8_matrix_free.md`。
2. 如果是“实现细节、架构解释、阶段验证方法、代码路径说明”，优先沉淀到 `docs/matrix_free_architecture_notes.md`。
3. 如果是“研究计划、数学原则、总体路线变化”，应提醒用户同步检查 `dut-postdoc` 中对应文档。
4. 如果只是临时调试输出、一次性命令或已过期判断，不需要沉淀。

这个提醒是本仓库 AI 工作流的一部分，目的是让讨论结果逐步变成可复用的项目上下文，而不是散落在聊天记录中。
