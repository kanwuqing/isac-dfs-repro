"""用暴力置换枚举校验高效鬼影检测器

对应关系: brute_ghost 为参考实现, _has_ghost_efficient 为被测实现,
两者对同一批几何给出相同判定即视为一致
"""

from __future__ import annotations

import sys
import time
from itertools import permutations, product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import (  # noqa: E402
    has_range_collision,
    random_geometry,
    select_bs_subset,
)
from phase1_range import (  # noqa: E402
    C0,
    range_from_tap_index,
    tap_index_from_range,
)
from phase2_localize import (  # noqa: E402
    association_consistent,
    refine_position,
)
from run_experiments import REGION, _has_ghost_efficient, make_cfg  # noqa: E402


def brute_ghost(clean, bs_xy, tol=1e-6):
    """参考实现: 完整置换枚举."""
    m_bs, k = clean.shape
    genuine = []
    for kk in range(k):
        if association_consistent(clean[:, kk], bs_xy, tol=1e-6):
            x, _ = refine_position(clean[:, kk], bs_xy)
            genuine.append(x)
    genuine = np.array(genuine) if genuine else np.zeros((0, 2))

    def is_gen(x):
        if genuine.shape[0] == 0:
            return False
        return bool(np.min(np.linalg.norm(genuine - x[None, :], axis=1)) < 1e-3)

    perms = list(permutations(range(k)))
    for combo in product(perms, repeat=m_bs - 1):
        cand = np.zeros((m_bs, k))
        cand[0, :] = clean[0, :]
        for mi in range(1, m_bs):
            for t in range(k):
                cand[mi, t] = clean[mi, combo[mi - 1][t]]
        for t in range(k):
            if association_consistent(cand[:, t], bs_xy, tol=tol) and not is_gen(cand[:, t]):
                x, _ = refine_position(cand[:, t], bs_xy)
                if not is_gen(x):
                    return True
    return False


# 暴力法的代价随(K!)^(M-1) 增长: M=4 且 K=4 时需 216 组置换, 单次约 35 秒
# 网格取 M=3 的 K=2/3/4 与 M=4 的 K=2/3, 同时覆盖多基站与多目标两个维度
GRID = [(3, 2), (3, 3), (3, 4), (4, 2), (4, 3)]


def main() -> None:
    header = (
        f"{'M':>3} {'K':>3} {'样本数':>6} {'暴力检出':>10} "
        f"{'快速检出':>10} {'一致':>6} {'暴力耗时':>9} {'快速耗时':>9}"
    )
    print(header, flush=True)

    for m_bs, k in GRID:
        cfg = make_cfg(100e6, k, m_bs=m_bs)
        rng = np.random.default_rng(4242 + 10 * m_bs + k)
        n_checked = 0
        nb = nf = 0
        tb = tf = 0.0

        while n_checked < 8:
            got = None
            for _ in range(5000):
                g = random_geometry(m_bs, k, REGION, rng)
                g, _ = select_bs_subset(g, min(3, m_bs))
                d = g.bs_to_target()
                limit = min(cfg.max_paths, cfg.cp_len) * C0 / (2 * cfg.bandwidth)
                if d.max() > limit or d.min() < 1.0:
                    continue
                cl = np.array(
                    [
                        [range_from_tap_index(tap_index_from_range(v, cfg), cfg) for v in row]
                        for row in d
                    ]
                )
                if has_range_collision(cl):
                    continue
                got = (g, cl)
                break
            if got is None:
                break

            g, cl = got
            n_checked += 1
            t0 = time.time()
            b = brute_ghost(cl, g.bs_xy)
            tb += time.time() - t0
            t0 = time.time()
            f = _has_ghost_efficient(cl, g.bs_xy)
            tf += time.time() - t0
            nb += int(b)
            nf += int(f)

        agree = "是" if nb == nf else "否"
        print(
            f"{m_bs:>3} {k:>3} {n_checked:>5} {nb:>12} {nf:>11} "
            f"{agree:>7} {tb:>9.3f} {tf:>8.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
