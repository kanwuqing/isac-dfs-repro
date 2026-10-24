"""检查仓库源码的标点与注释是否合规.

只检查注释与 docstring, 不检查字符串字面量, 因为打印给人看的提示语
保留中文全角标点更自然.

规则:
1. 注释里不允许出现全角标点;
2. 注释里逗号, 分号, 冒号, 小括号后面必须跟一个空格;
3. 注释与 docstring 必须含中文, 不允许整句英文;
4. 不允许出现连续三个以上空行.

运行: python scripts/check_style.py
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC_DIRS = [HERE]
SKIP = {"check_style.py"}

FULLWIDTH_PUNCT = "\uff0c\uff1b\uff1a\uff08\uff09\uff01\uff1f\u3002\u3001"
CJK = re.compile(r"[\u4e00-\u9fff]")
SPACE_RULES = (",", ";", ":", "(")
NO_SPACE_FOLLOW = set(",;:)}]")


def iter_py_files():
    """遍历要检查的源码文件."""
    for d in SRC_DIRS:
        for f in sorted(d.rglob("*.py")):
            if f.name in SKIP or "__pycache__" in str(f):
                continue
            yield f


def collect_texts(path: Path):
    """取出(行号, 文本)的注释与 docstring 片段."""
    src = path.read_text(encoding="utf-8")
    items = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                items.append((tok.start[0], tok.string))
            elif tok.type == tokenize.STRING and tok.string.startswith(('"""', "'''")):
                items.append((tok.start[0], tok.string))
    except (tokenize.TokenError, IndentationError):
        pass
    return items


def check_file(path: Path) -> list[str]:
    """检查单个文件, 返回问题列表."""
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").split("\n")

    for i in range(2, len(lines)):
        if not lines[i].strip() and not lines[i - 1].strip() and not lines[i - 2].strip():
            errors.append(f"{path.name}:{i + 1}: 连续三个空行")
            break

    for ln, text in collect_texts(path):
        for ch in FULLWIDTH_PUNCT:
            if ch in text:
                errors.append(f"{path.name}:{ln}: 注释含全角标点 {ch!r}")
                break

        if not CJK.search(text) and re.search(r"[A-Za-z]{4,}\s+[A-Za-z]{4,}", text):
            errors.append(f"{path.name}:{ln}: 注释没有中文")

        for j, ch in enumerate(text):
            if ch not in SPACE_RULES:
                continue
            nxt = text[j + 1] if j + 1 < len(text) else ""
            if nxt == "" or nxt in (" ", "\t", "\n"):
                continue
            if nxt in NO_SPACE_FOLLOW:
                continue
            prev = text[j - 1] if j > 0 else ""
            if CJK.search(prev) or CJK.search(nxt):
                # 括号里装的是中文时不套这条规则, 例如 (一致), (脚本名).
                # 这条规则本来是针对半角括号紧跟英文缩写或数字的情形.
                if nxt and CJK.match(nxt):
                    continue
                # 紧贴 ASCII 的括号多半是公式或形状标注, 如 K!^(M-1), (K, K)
                if ch == "(" and nxt.isascii() and (nxt.isalnum() or nxt in "!^"):
                    continue
                errors.append(f"{path.name}:{ln}: {ch!r} 后面缺空格")
                break

    return errors


def main() -> int:
    files = list(iter_py_files())
    total = 0
    for f in files:
        errs = check_file(f)
        if errs:
            total += len(errs)
            print(f"\n{f.relative_to(ROOT)}")
            for e in errs[:20]:
                print(f"  {e}")
            if len(errs) > 20:
                print(f"  ...另有 {len(errs) - 20} 处")

    print()
    if total == 0:
        print(f"检查了 {len(files)} 个文件, 全部合规.")
        return 0
    print(f"检查了 {len(files)} 个文件, 发现 {total} 处不合规.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
