# ADR-0001：采用分层 `src/soptx` 布局

- 状态：Accepted
- 目标版本：`1.1.0.dev0 → 1.1.0 → 2.0.0`

## Context

旧仓库把 PDE、网格创建、日志、材料、有限元分析和拓扑优化职责交叉放在
`model`、`analysis`、`interpolation` 等目录中。根包还声明了已经不存在的
`pde/material/solver/filter/opt`，而测试、论文脚本和第三方参考代码可能进入安装包。

## Decision

采用 `core → materials/problems → fem → topology → visualization` 的单向依赖，
并使用 `src/` 布局和显式子包公共 API。Problem 只表达数学问题；Material 独立；
网格由 FEM workflow 或 example case 创建。

`1.1.x` 仅保留迁移表声明的旧公共路径，并通过薄转发发出一次
`DeprecationWarning`。内部、`old`、`backup` 和论文驱动不属于安装兼容承诺。
不复活已经不存在的根命名空间。`2.0.0` 删除公共兼容层。

## Consequences

- wheel 仅包含 `src/soptx` 中允许安装的包，排除 tests/examples/experiments/old。
- compatibility 模块不得复制已有 maintained 实现；archive 源码保存在
  `experiments/legacy/package_archive`，不进入 wheel。
- 新低层代码不能导入旧兼容命名空间或更高层。
- 孵化示例必须使用公共 API；可复用内核只有在具备第二消费者和独立测试后才晋级。
- `reference_code/` 在归档 tag 获得授权前只隔离并记录哈希，不删除。
