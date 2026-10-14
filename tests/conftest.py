r"""测试进程的临时目录：**钉在一个一定写得进去的地方**。

## 为什么要有这一个文件

本机沙箱下 `TEMP` / `TMP` 指的那个 `dsh-*` 目录，**在测试进程跑起来时可能已经不在了**
（一次实测：`TEMP` 与 `TMP` 都指着 `C:\Users\...\Temp\dsh-2el5jo`，而它不存在）。
`tempfile` 找不到能用的候选目录时，**回落到当前工作目录** —— 于是一次实测里：

    gettempdir() -> D:\Android\ark-ai-op
    mkstemp()    -> D:\Android\ark-ai-op\tmpgpql3ocg

也就是说**任何**一次 `mkstemp()` / `mkdtemp()` / `Path(gettempdir())` 都会在
**仓库根**下留东西：几个单元测试用的 `ark-roster-*` / `ark-*-tests` 目录，
以及跑完一轮之后冒出来的 0 字节 `tmp*` 文件（那是别人调 `mkstemp()` 留下的）。

所以这里把 `tempfile.tempdir` **钉死**在 `data/pytest_scratch/`（`data/` 本来就
gitignored），并显式建出来。**这不是把测试挪走，是把它们留在地上的东西挪到该在的地方。**

**它改的是进程级的 `tempfile.tempdir`** —— 所以它只在这一个测试进程里成立，
跑产品代码的进程不受影响。
"""

from __future__ import annotations

import pathlib
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRATCH = REPO / "data" / "pytest_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(SCRATCH)
