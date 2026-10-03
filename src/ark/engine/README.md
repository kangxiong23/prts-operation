# ark.engine

**职责**：在线引擎。从真机取一帧像素，到 AI 能调用的一个算子，全在这一层。

六个子包**互不 import**，只各自依赖 `ark.contracts`。唯一的例外是 `mcp/` 依赖 `api/`（桥接层需要算子清单）。把它们接到一起是 `../runtime/wiring.py` 的事。

```
device → vision → state → timeline → control → api → mcp
```

上面这条箭头**不是 import 关系，是数据流向**。每一环之间靠 `contracts` 里的端口（Protocol）连，具体实现在 `runtime` 里注入。

## 为什么子包之间不许互相 import

因为"数据流向"和"编译依赖"是两回事。让 control 直接 import vision，改一次视觉层就可能动到控制层；靠注入就只动 wiring 一处。

**这条约束如果将来证明太紧，正确的做法是改规则并记一条决策，不是偷偷加 import。**

**允许 import**：`ark.contracts`、`ark.engine`
**禁止 import**：`ark.offline`、`ark.runtime`

## 不在这里

- 资产生成、难例采集、基准集 → `../offline/`
- 依赖注入与入口 → `../runtime/`
