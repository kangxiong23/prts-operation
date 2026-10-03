# ark.engine.mcp

**职责**：把 `../api/` 的算子翻译成 MCP 工具，跑起来。

**这是 `ark.engine` 里唯一可以提到 MCP 的包。** 换一个桥接（自写插件、命令行、别的协议）只需要替换这一个目录，引擎其余部分不动。

## 为什么它值得单独一层

因为它承载的是**协议细节**：超时怎么设、图片怎么回、错误怎么表达、工具列表怎么动态变。这些东西会随对方版本变化，而引擎的业务不该跟着变。

**允许 import**：`ark.contracts`、`ark.engine.api`、`ark.engine.mcp`
**禁止 import**：其它 `ark.engine.*`、`ark.offline.*`

## 上机注意

调用超时默认 60 秒，而 `wait_until` 天然是长调用 —— **必须显式配置 `toolCallTimeoutMs`**，否则一个正常的等待会被掐断。

## 不在这里

- 算子本身 → `../api/`
