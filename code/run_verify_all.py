"""把几个验证脚本挨个跑一遍, 结果存日志

四个脚本从快到慢排: 前两个几秒到二十几秒, 后两个枚举暴力解,
K=5 那几例单次就要两三秒. --fast 只跑前两个, 现场演示用

    python scripts/run_verify_all.py
    python scripts/run_verify_all.py --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "logs"

# (脚本名, 超时秒数). 后两个吃时间, 给到半小时
TASKS = [
    ("verify_ghost_positive.py", 120),
    ("verify_ghost.py", 600),
    ("verify_enum_strict.py", 1800),
    ("verify_enum.py", 1800),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="只跑前两个")
    args = ap.parse_args()

    tasks = TASKS[:2] if args.fast else TASKS
    LOGS.mkdir(parents=True, exist_ok=True)

    log_path = LOGS / f"verify_all_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = log_path.open("w", encoding="utf-8")

    def say(text: str) -> None:
        log.write(text)
        log.flush()
        sys.stdout.write(text)
        sys.stdout.flush()

    # 子脚本直接继承控制台编码, 中文 Windows 下是 GBK, 日志里会变乱码,
    # 所以强制子进程用 UTF-8 输出
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    say(f"开始 {datetime.now():%Y-%m-%d %H:%M:%S}, "
        f"{'只跑前两个' if args.fast else '四个全跑'}\n")

    rows = []
    try:
        for name, timeout in tasks:
            script = HERE / name
            say(f"\n{'=' * 70}\n{name}\n{'=' * 70}\n")

            t0 = time.time()
            stdout = stderr = ""
            rc = None
            status = "失败"

            # 输出先落临时文件, 再读回来一起打印
            out_f = LOGS / f"_tmp_{name}.out"
            err_f = LOGS / f"_tmp_{name}.err"
            try:
                with out_f.open("w", encoding="utf-8") as fo, \
                        err_f.open("w", encoding="utf-8") as fe:
                    rc = subprocess.run(
                        [sys.executable, str(script)],
                        cwd=str(ROOT), stdout=fo, stderr=fe,
                        env=env, timeout=timeout,
                    ).returncode
                status = "通过" if rc == 0 else "失败"
            except subprocess.TimeoutExpired:
                status, rc = "超时", None
            except Exception as exc:  # noqa: BLE001
                stderr = f"{type(exc).__name__}: {exc}"

            if out_f.exists():
                stdout = out_f.read_text(encoding="utf-8", errors="replace")
                out_f.unlink()
            if err_f.exists():
                stderr += err_f.read_text(encoding="utf-8", errors="replace")
                err_f.unlink()

            if stdout:
                say(stdout if stdout.endswith("\n") else stdout + "\n")
            if stderr.strip():
                say(f"[stderr] {stderr.strip()}\n")

            cost = time.time() - t0
            say(f"-- {status}, 退出码 {rc}, {cost:.1f} 秒\n")
            rows.append((name, status, rc, cost))

        say(f"\n{'=' * 70}\n")
        for name, status, rc, cost in rows:
            say(f"{name:<26} {status:<4} {str(rc):>4} {cost:>8.1f} 秒\n")

        bad = [r for r in rows if r[1] != "通过"]
        if bad:
            say(f"\n{len(bad)} 个没过: {', '.join(r[0] for r in bad)}\n")
        else:
            say(f"\n{len(rows)} 个全过\n")
        say(f"日志 {log_path.relative_to(ROOT)}\n")
        return 0 if not bad else 1
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
