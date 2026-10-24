"""把项目依赖从系统 site-packages 复制进 .venv.

系统里的 Python 在沙箱之外, 这个脚本能从那儿读, 但删不了 (WinError 5).
所以它只做复制, 绝不改动或删除系统 Python 里的任何东西.

也就是说"从系统环境里卸掉这些库"这一步在这里做不到, 得自己手动跑.
`scripts/uninstall_from_system.ps1` 干的就是这件事.

用法:
    .venv\\Scripts\\python.exe scripts\\copy_deps_to_venv.py [--apply]

不加 --apply 就是空跑, 只打印打算做什么.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SYS = Path(r"C:\Users\kwq\AppData\Local\Programs\Python\Python312\Lib\site-packages")
VENV = Path(r"C:\Users\kwq\OFDM\.venv\Lib\site-packages")

# requirements.txt 的完整依赖闭包: numpy, scipy, matplotlib, pandas,
# 再加上 matplotlib 和 pandas 导入时需要的模块.
DISTS = [
    # numpy 数值计算
    "numpy", "numpy-2.5.3.dist-info", "numpy.libs",
    # scipy 优化与指派算法
    "scipy", "scipy-1.18.1.dist-info", "scipy.libs",
    # matplotlib 画图
    "matplotlib", "matplotlib-3.11.2.dist-info", "mpl_toolkits", "pylab.py",
    "contourpy", "contourpy-1.4.0.dist-info",
    "cycler", "cycler-0.12.1.dist-info",
    "fontTools", "fonttools-4.66.0.dist-info",
    "kiwisolver", "kiwisolver-1.5.1.dist-info",
    "pyparsing", "pyparsing-3.3.3.dist-info",
    # pandas 读写 CSV
    "pandas", "pandas-3.0.6.dist-info", "pandas.libs",
    "dateutil", "python_dateutil-2.9.0.post0.dist-info",
    "six.py", "six-1.17.0.dist-info",
    "tzdata", "tzdata-2026.4.dist-info",
    # pillow, matplotlib 写图片时要用
    "PIL", "pillow-12.3.0.dist-info",
    # matplotlib 和 pandas 都要用的公共工具
    "packaging", "packaging-26.2.dist-info",
]


def main() -> int:
    apply_changes = "--apply" in sys.argv
    if not SYS.is_dir():
        print(f"错误：未找到系统 site-packages：{SYS}")
        return 1
    if not VENV.is_dir():
        print(f"错误：未找到 venv 的 site-packages：{VENV}")
        return 1

    print(f"source : {SYS}")
    print(f"target : {VENV}")
    print(f"模式   : {'执行' if apply_changes else '试运行（加 --apply 才执行）'}")
    print()

    done = skipped = failed = 0
    for name in DISTS:
        src = SYS / name
        dst = VENV / name
        if not src.exists():
            print(f"  [skip] {name} (absent in system)")
            skipped += 1
            continue
        if dst.exists():
            print(f"  [skip] {name} (already present in venv)")
            skipped += 1
            continue
        if not apply_changes:
            print(f"  [would copy] {name}")
            done += 1
            continue
        try:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"  [copy] {name}")
            done += 1
        except Exception as exc:                       # noqa: BLE001
            print(f"  [FAIL] {name}: {exc}")
            failed += 1

    print()
    verb = "copied" if apply_changes else "would copy"
    print(f"{verb}={done} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
