# DSH 平台事实

> 状态：已定

本机 DSH 版本 `0.2.0-rc.1`。下面每一条都来自本机安装目录的代码与文档，或来自 Inspect 接口的实时查询。凡未实际跑过的，都集中在最后一节。

它约束三处：`../10-规格/接口层/`（MCP 的边界）、`../../dsh/bundle/`（预设的作用域）、以及将来的可视化（slot 座位）。

原始论证过程见 `../90-归档/总体架构与技术栈.md`（已归档）。本文件是它的整理版，实现时看这份。

## 一、MCP 能给什么

| 能力 | 状态 |
| --- | --- |
| 工具定义（名称 / 描述 / 参数 schema / 输出 schema） | 支持，含输出 schema 校验 |
| 返回结构化 JSON | 支持 |
| 返回图片（PNG / JPEG / WebP / GIF） | 支持，直接进对话 |
| 单次调用可取消 | 支持 |
| 调用超时 | 默认 60 秒，**可配置** |
| 工具列表动态变化 | 支持，自动重新同步 |
| 资源（resources） | 支持，按需拉取，二进制不内联进上下文 |
| 服务器指令进系统提示词 | 支持，上限 32 KB |
| 结果走 DSH 同一套策略管线 | 支持 |

`toolCallTimeoutMs` 可配置，这是阻塞式 `wait_until` 能成立的硬前提。

## 二、MCP 给不了什么

- **没有进度通知。** 协议有 progress，但 DSH 这一侧没有接（`dsh-mcp-client` 里没有任何 progress 处理）。长调用期间界面上只有「调用中」。
- 服务器不能主动发起对话。**不影响** —— 我们选的就是阻塞式等待。
- 拿不到 DSH 宿主服务（timer / storage / jobs / spillStore / session 事件）。**不影响** —— 引擎自己管自己的文件和进程。
- 提示词模板、音频、嵌入资源、资源订阅、task 扩展均不支持。**都不用。**
- stdio 传输会清洗环境变量（去掉匹配 `*KEY|PASSWORD|SECRET|TOKEN*` 与 `DSH_*` 的名称）。我们走 HTTP，不受影响。

## 三、MCP 的作用域可以被预设限定

把 `@deepseek-ai/dsh-mcp-client` 写进预设的 `config.plugins`，它的工具与提示词段落**只对选中该预设的会话可见**。

证据：

- `dsh-agent-preset-registry/README.md`：「Plugin registrations inherit the preset scope, and the Agent scope's parent link controls visibility.」
- `dsh-tools` 的 `register()` 把工具落在注册方 context 所属的作用域层；读取时按调用方作用域解析。
- `dsh-mcp-client/lib/index.js` 的注释：「Agent-scoped MCP servers may reuse a namespace in another Agent, while global instances and duplicates inside one Agent remain mutually exclusive.」—— 两种模式**并列**，是选择而非必然。

附带好处：预设实例「eagerly activated once and shared」，MCP 连接只建一次，不会每个会话各起一份。

## 四、模式（预设）能带什么

| 想要的 | 怎么给 |
| --- | --- |
| 专属工具链 | 挂 `dsh-mcp-client` 一行 |
| 专属系统提示词 | 挂 `dsh-persona`，字段 `prefix`（必填） |
| 专属 skill | 挂 `dsh-skill-filesystem`，配 `customSkillDirs` |
| 专属工具包 | 挂现成的 `@deepseek-ai/dsh-tool-*` |

落盘是一个 bundle 目录两个文件（`package.json` 与 `cordis.patch.yml`），**零代码、零依赖**。

## 五、预设做不到的事

- **不能只是一个目录。** 注册中心不扫目录、不接受预设路径；必须装 bundle，而装 bundle 要一次高权限批准。
- **不是安全沙箱。** 官方原文：「Presets are not security sandboxes: YAML and plugins can execute Host code.」不能用模式限制 AI 只能操作游戏；真正的边界是权限预设与沙箱策略。
- **不能编辑已装预设的组合。** 管理器不能编辑 `agent` 预设组合，Web 插件页里预设条目是只读的。预设只能由 bundle 自带的补丁声明 —— 要改就得改 bundle 源目录再重装。
- **不能给已产出内容的会话换预设**（只能对空白会话切）。
- **覆盖是整体替换，不合并。** 复制官方 `standard` 全量再改的话，官方以后新增的行不会进你的副本。
- **不提供文件操作。** 不能靠预设分发 skill 文件或脚本，那些得另外落盘。
- **子代理不能自选预设**（`subagent` / `subagent_fork` 没有 `mode` 参数）。

## 六、客户端插件的座位（实时 Slot 树查实）

平台为每个 slot 标了 `replaceRisk`：

- `none` —— 追加式座位，注册只增加内容。**优先选这些。**
- `shadows-shipped-ui` —— 注册会替换掉官方 UI。别用。

| 座位 | 类型 | 风险 | 用途 |
| --- | --- | --- | --- |
| `tool.call.toolview` | keyed，keyDomain 开放 | 追加 | 给某一次工具调用的卡片做自定义渲染 |
| `sidebar.panellist` | list | none | 全局面板图标 |
| `main` | keyed，已被 `conversation` 占一个 key | 追加 | 中央大面板 |
| `sidebar.right.pane.tab.title`、`sidebar.right.tab.document.*` | keyed，尚无人占用 | 追加 | 会话级右侧面板标签页 |
| `settings.section` | list | none | 独立设置页 |
| `conversation.input.dock`、`conversation.composer.dock` | list | none | 输入框上下方的整宽条目 |
| `sidebar`、`sidebar.settings`、`conversation.session` 等 | — | shadows-shipped-ui | 不要用 |

**可视化的第一版应该做「给工具调用做自定义渲染」，而不是开面板** —— 理由是 `tool.call.toolview` 已占的 key 里一个 `mcp__*` 都没有，全是空位，而且图会自己出现在对话流里，用户不需要去开面板。

## 七、客户端插件的接口风险（官方自己划的线）

| 接口点 | 风险 | 变动后果 | 官方态度 |
| --- | --- | --- | --- |
| 主题 token（`--dsw-alias-*`） | 最低 | 外观降级 | 重命名「只会让外观降级，永远不会让渲染崩掉」 |
| 任何 Harness Client 包 | 最高 | 组件抛错 → 整块 slot 空白 | 「会无预警变化」，明确要求不要这么做 |
| slot key 名与 `ctx.slots.register` 选项 | 中高 | 面板不出现 | 真正的耦合点，无法消除 |
| `window.__ModuleLoader__` | 中 | client 半边不加载 | 契约本身，没有替代 |
| 给 session 事件加新 `type` | 禁止 | Session 拒绝重开 | 明令禁止 |
| 监听 `system-prompt/assemble` 改文本或工具 | 禁止 | 覆盖别的插件 | 应该用 `ctx.systemPrompt.section()` |

三条有利事实：失败是**局部的**（slot entry 崩了只是那一块空白，不带崩会话）；配置在补丁层，**升级后存活**；不兼容的 DSH peer 依赖会被**拒绝安装与激活**。

**纪律**：UI 插件是**纯消费者** —— 只从引擎读数据画图，不参与业务逻辑。违反这条，风险就从「最坏看不见面板」变成「可能修不动」。

## 八、配置契约的漂移风险（本机实证）

1. **本机有一份死代码。** `C:\Users\ASUS\.dsh\.agent-presets\liangshen\` 目录还在，但注册中心不扫描它；检索整个安装目录，读取 `.agent-presets` 的地方是零处。
2. **有一份过时的官方 skill。** `dsh-preset-authoring` 仍在教这个已经没人读的目录格式。
3. **字段级漂移是静默的。** 那个死目录用 `config.text`，而现在 `dsh-persona` 是 `prefix` 且必填。schema 校验不过 → 预设无法挂载，**且不报崩溃**。
4. **包名漂移同样静默失败** —— 包改名后预设仍留在列表里带诊断信息，但选它开不出会话。
5. **子智能体的模型枚举与 profile 配置不同步。** 实测：`list_subagent_models` 只报出 `deepseek-official` 与 `botcfcodexplus` 两个 provider，而 profile 的 `cordis.patch.yml` 里配了五个（另有 `opencode-go`、`openai`、`gemini-botcf`）。枚举读到的更像根目录那份旧的 `settings.yaml.imported`（`subagent-model-selection.allowedModels` 正好只有那两条）——**机制是推断**（未读源码）。**后果是实测的**：子智能体工具**选不到**未列出的模型，而 `workflow` 的 `agent()` 指定同一个 provider / model **能直连成功**（它不走那道白名单）。所以要在对照实验里用便宜模型，走 `workflow`。

版本锚点：`dsh`、`dsh-mcp-client`、`dsh-agent-preset` 全部是 `0.2.0-rc.1`。

**对策**：字段名一律从实时投影的 schema 或官方自带预设抄，绝不凭文档或记忆；bundle 源目录放工作区并纳入版本管理，当**可重建的派生物**；升级后新开**空白会话**，用 Inspect 查 `mcp__ark__*` 是否在工具面里（已加入的会话保留它启动时的版本，必须在会话外验）。

## 九、未实测

- 第三节的作用域结论来自**文档 + 已编译代码 + 实时 Config 投影**，**未经过运行时实测** —— 装 bundle 需要高权限，调研中的安装步骤被审批拒绝。
- **没有实际写过、装过、跑过任何一个客户端插件。** 注册 API 的实际行为、主题 token 的实际取值、升级后真实的破损程度，全部未验证。
- `includeDefaultRoots: false` 能否屏蔽已注册在宿主层的 skill：推断为「屏蔽不掉」，未验证。
- 预设内用相对路径挂自定义插件行：只有方言规则与历史先例，本版本无随附样例，未验证。
- 客户端可用的 Service 清单、主题 token 的完整取值：未查。
