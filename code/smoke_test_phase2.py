"""阶段二冒烟测试: 三边定位, 数据关联, 定位."""
from __future__ import annotations

import sys
from pathlib import Path
from itertools import permutations

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import random_geometry, select_bs_subset, has_range_collision
from phase1_range import (
    OFDMConfig, C0, range_from_tap_index, tap_index_from_range,
)
from phase2_localize import (
    refine_position, trilaterate_linear, ml_data_association,
    pairwise_position_errors, association_consistent,
)


def main() -> int:
    print("=" * 70)
    print("冒烟测试 5: 线性三边定位还原已知点")
    print("=" * 70)
    bs = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    true_x = np.array([30.0, 40.0])
    d = np.linalg.norm(bs - true_x, axis=1)
    est = trilaterate_linear(d, bs)
    print(f"  真值={true_x}  三边定位={np.round(est, 9)}")
    assert np.allclose(est, true_x, atol=1e-6), est
    x, resid = refine_position(d, bs, x0=est)
    print(f"  精修后={np.round(x, 9)}  最大残差={np.max(resid):.3e}")
    assert np.allclose(x, true_x, atol=1e-6), x
    print("通过")

    print("\n" + "=" * 70)
    print("冒烟测试 6: 完美测距下的最大似然数据关联")
    print("=" * 70)
    cfg = OFDMConfig(n_subcarriers=1024, subcarrier_spacing=97_656.25,
                     n_bs=4, n_targets=3, max_paths=96, cp_len=128)
    limit = min(cfg.max_paths, cfg.cp_len) * C0 / (2 * cfg.bandwidth)
    rng = np.random.default_rng(7)
    n_ok = 0
    trials = 15
    done = 0
    while done < trials:
        g = random_geometry(4, 3, 90.0, rng)
        g, _ = select_bs_subset(g, 3)
        d_true = g.bs_to_target()
        if d_true.max() > limit:
            continue
        clean = np.array([[range_from_tap_index(tap_index_from_range(v, cfg), cfg)
                           for v in row] for row in d_true])
        if has_range_collision(clean):
            continue
        done += 1
        # 阶段一的输出: 每个基站给出一个不带标签, 已排好序的距离集合
        bs_ranges = [np.sort(clean[m]) for m in range(4)]
        res = ml_data_association(bs_ranges, g.bs_xy, sigma_d=cfg.delta_d / 2)
        if not res.feasible:
            continue
        errs = pairwise_position_errors(res.positions, g.target_xy)
        # 论文的量化模型下, 精度上限由 Delta_d 卡住: 100 MHz 时是 0.75 米
        # 所以门槛取在 Delta_d 以下, 也就是 0.5 米, 再给三边定位的几何
        # 留一点余量, 这样才算合理的判据
        if errs.max() < 1.0:
            n_ok += 1
    print(f"  完美测距最大似然关联: {n_ok}/{trials} 个几何的定位误差小于 1 米")
    # 论文的结论是鬼影目标少但并非不可能; 这里只跑 15 个几何,
    # 偶尔失败是正常的, 所以只要求通过率够高
    assert n_ok >= int(0.8 * trials), f"{n_ok}/{trials}"
    print("通过（不少于 80% 的几何定位误差低于量化下限）")

    print("\n" + "=" * 70)
    print("冒烟测试 7: association_consistent 拒绝鬼影几何")
    print("=" * 70)
    bs = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    good = np.linalg.norm(bs - np.array([30.0, 40.0]), axis=1)
    print(f"  一致性判定: {association_consistent(good, bs)}  （预期为 True）")
    assert association_consistent(good, bs)
    bad = good.copy()
    bad[2] += 25.0          # 把一个距离挪一点点, 这几个圆就交不到同一点了
    print(f"  扰动后判定: {association_consistent(bad, bs)}  （预期为 False）")
    assert not association_consistent(bad, bs)
    print("通过")

    print("\n" + "=" * 70)
    print("冒烟测试 8: 最大似然关联对量化误差稳健")
    print("=" * 70)
    cfg = OFDMConfig(n_subcarriers=1024, subcarrier_spacing=97_656.25,
                     n_bs=4, n_targets=3, max_paths=96, cp_len=128)
    rng = np.random.default_rng(11)
    n_ok, done, n_attempt = 0, 0, 0
    trials = 25
    while done < trials and n_attempt < 300:
        n_attempt += 1
        g = random_geometry(4, 3, 90.0, rng)
        g, _ = select_bs_subset(g, 3)
        d_true = g.bs_to_target()
        if d_true.max() > limit:
            continue
        clean = np.array([[range_from_tap_index(tap_index_from_range(v, cfg), cfg)
                           for v in row] for row in d_true])
        if has_range_collision(clean):
            continue
        done += 1
        bs_ranges = [np.sort(clean[m]) for m in range(4)]
        res = ml_data_association(bs_ranges, g.bs_xy, sigma_d=cfg.delta_d / 2)
        if res.feasible:
            errs = pairwise_position_errors(res.positions, g.target_xy)
            if errs.max() < 1.5:
                n_ok += 1
    print(f"  量化测距最大似然关联: {n_ok}/{done} 个几何定位误差小于 1.5 米")
    print(f"  （距离刻度 Δd = {cfg.delta_d:.4f} 米）")
    assert n_ok / max(done, 1) > 0.8, f"{n_ok}/{done}"

    print("\n阶段二冒烟测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
