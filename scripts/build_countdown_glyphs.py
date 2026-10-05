"""重建倒计时那套**粗体字形表**（`assets/features/glyphs_countdown/`）。

    .venv/Scripts/python.exe scripts/build_countdown_glyphs.py

**这个脚本是命令行入口，不是实现**：建表逻辑住
`src/ark/offline/assets_gen/countdown.py` —— 与费用那套（`assets_gen/glyphs.py`）
**并列**、同一条口径。放那边是因为它是**生成物**（读夹具、写资产），而且
`tests/unit/test_vision_bench.py` 要 import 它来验「同一批取材跑两次逐像素一致」。

**包还没装**（`pyproject.toml` 的 `[project.scripts]` 还是注释状态），所以这里自己把
`src/` 放进 `sys.path` —— 与 `scripts/bench.py` 一致。

它只写资产，**不读验收套件**：取材帧与验收帧是分开的两批。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.offline.assets_gen.countdown import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
