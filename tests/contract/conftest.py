"""把 tests/contract 放进 sys.path，好让测试能 import import_rules。

不依赖 pytest 的 basedir 插入行为，免得换个 pytest 版本就找不到模块。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
