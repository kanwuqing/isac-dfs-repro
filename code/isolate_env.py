"""把项目依赖从系统 site-packages 迁进 .venv.

为什么用移动而不是复制: 要的效果是既把东西放进 venv, 又从系统 Python 里
拿掉. 移动一步两件事都办到了; 而且不像 pip uninstall 那样一去不回, 文件
只是换了个地方待着, 想找回来还在. 磁盘上什么都没删.

安全起见只碰白名单里的项目依赖. 非项目依赖的工具 (pip, setuptools, wheel,
uv, _distutils_hack, pkg_resources) 一律原样留在系统 Python 里, 因为
uv.exe 和 pip 自己就住在那儿, 动了工具链就废了.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SYS = Path(
    r"C:\Users\kwq\AppData\Local\Programs\Python\Python312\Lib\site-packages"
)
VENV = Path(r"C:\Users\kwq\OFDM\.venv\Lib\site-packages")

# 项目依赖的发行包, 含运行时依赖和它们的传递依赖, 再加
# 属于它们的辅助模块. 这里列的东西都可以从系统 Python 里搬走.
MOVE = [
    # numpy 数值计算
    "numpy", "numpy-2.5.3.dist-info", "numpy.libs",
    # scipy 优化与指派算法
    "scipy", "scipy-1.18.1.dist-info", "scipy.libs",
    "scipy-1.18.1-cp312-cp312-win_amd64.whl",
    # matplotlib 及其依赖
    "matplotlib", "matplotlib-3.11.2.dist-info", "mpl_toolkits", "pylab.py",
    "contourpy", "contourpy-1.4.0.dist-info",
    "cycler", "cycler-0.12.1.dist-info",
    "fontTools", "fonttools-4.66.0.dist-info",
    "kiwisolver", "kiwisolver-1.5.1.dist-info",
    "pyparsing", "pyparsing-3.3.3.dist-info",
    # pandas 及其依赖
    "pandas", "pandas-3.0.6.dist-info", "pandas.libs",
    "dateutil", "python_dateutil-2.9.0.post0.dist-info",
    "six.py", "six-1.17.0.dist-info",
    "tzdata", "tzdata-2026.4.dist-info",
    # 这个文件用项目自己的 venv 运行
    # pillow, matplotlib 写图片的后端
    "PIL", "pillow-12.3.0.dist-info",
]

# 通用包, 别的工具也可能要用. 它们在系统 Python 里原地保留, 同时
# 再复制一份进 venv, 因为 matplotlib 和 pandas 缺了它们就用不了.
# 注意是复制不是移动, 这样系统的工具链不会被弄坏.
#
COPY = [
    "packaging", "packaging-26.2.dist-info",
]

DRY = "--apply" not in sys.argv


def main() -> int:
    if not SYS.is_dir():
        print(f"错误：未找到系统 site-packages：{SYS}")
        return 1
    VENV.mkdir(parents=True, exist_ok=True)
    print(f"system : {SYS}")
    print(f"venv   : {VENV}")
    print(f"模式   : {'试运行（加 --apply 才执行）' if DRY else '执行'}")
    print()

    moved = copied = skipped = 0
    for name in MOVE:
        src = SYS / name
        dst = VENV / name
        if not src.exists():
            print(f"  [skip] {name} (absent in system)")
            skipped += 1
            continue
        if dst.exists():
            print(f"  [skip] {name} (already in venv)")
            skipped += 1
            continue
        print(f"  [move] {name}")
        if not DRY:
            if src.is_dir():
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(src, dst)
                src.unlink()
        moved += 1

    for name in COPY:
        src = SYS / name
        dst = VENV / name
        if not src.exists():
            print(f"  [skip] {name} (absent)")
            skipped += 1
            continue
        if dst.exists():
            print(f"  [skip] {name} (already in venv)")
            skipped += 1
            continue
        print(f"  [copy] {name}")
        if not DRY:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        copied += 1

    print()
    print(f"moved={moved} copied={copied} skipped={skipped}"
          + (" (dry-run, nothing changed)" if DRY else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
