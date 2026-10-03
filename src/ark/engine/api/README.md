# ark.engine.api

**职责**：引擎对外的算子清单。每个算子有名字、描述、参数 schema 和执行体。

放什么：读状态类、下指令类、等待类（`wait_until`）、回看类，以及算子的注册表。

## 它不认识任何桥接

这一层是**引擎自己的语义**。MCP、命令行、测试，都只是它的消费者。

因为端口定义在 `contracts` 里，这一层**只依赖 `contracts`**，需要的具体实现由 `../runtime/wiring.py` 注入。所以它没有 `mcp` 字样，也不需要知道谁在调它。

**桥接可替换**这条设计承诺，就是靠这个包和 `../mcp/` 的分工兑现的。

**允许 import**：`ark.contracts`、`ark.engine.api`
**禁止 import**：其它 `ark.engine.*`、`ark.offline.*`

## 不在这里

- 翻译成 MCP 工具 → `../mcp/`
- 算子背后读的东西怎么定义 → `../state/`
