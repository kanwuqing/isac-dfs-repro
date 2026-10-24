"""快速冒烟测试: 阶段一能否恢复正确的距离单元?

运行: python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import random_geometry, select_bs_subset  # noqa: E402
from phase1_range import (  # noqa: E402
    OFDMConfig,
    C0,
    phase1_range_estimation,
    range_from_tap_index,
    tap_index_from_range,
    g_matrix,
    ls_channel_estimate,
    make_pilots,
)


def main() -> int:
    print("=" * 70)
    print("冒烟测试 1: 波形参数自洽")
    print("=" * 70)
    cfg = OFDMConfig(n_subcarriers=1024, subcarrier_spacing=97_656.25, n_bs=4, n_targets=4)
    print(f"子载波数 N     = {cfg.n_subcarriers}")
    print(f"子载波间隔 df  = {cfg.subcarrier_spacing/1e6:.6f} MHz")
    print(f"带宽 B         = {cfg.bandwidth/1e6:.4f} MHz")
    print(f"距离刻度 Δd    = {cfg.delta_d:.6f} 米   （论文: 100 MHz 时 0.75 米，400 MHz 时 0.1875 米）")
    print(f"最大无模糊距离 = {cfg.max_range:.2f} 米")
    assert cfg.cp_len < cfg.n_subcarriers

    # 用论文给的数核对 Delta_d. 论文写的是取整过的值
    # (0.75 米和 0.1875 米), 所以允许 1% 的偏差
    for bw, expected in [(100e6, 0.75), (400e6, 0.1875)]:
        c = cfg.with_bandwidth(bw)
        rel = abs(c.delta_d - expected) / expected
        print(f"  B={bw/1e6:6.1f} MHz -> Δd={c.delta_d:.5f} 米（论文 {expected}，相对误差 {rel*100:.2f}%）")
        assert rel < 0.01, (c.delta_d, expected)

    print("\n" + "=" * 70)
    print("冒烟测试 2: 无噪单路径下最小二乘估计精确")
    print("=" * 70)
    rng = np.random.default_rng(0)
    pilots = make_pilots(cfg, rng)
    g = g_matrix(cfg.n_subcarriers, cfg.max_paths)
    h_true = np.zeros(cfg.max_paths, dtype=complex)
    h_true[9] = 3.0 + 4.0j
    ybar = np.sqrt(cfg.tx_power) * (pilots[0] * (g @ h_true))
    h_hat = ls_channel_estimate(ybar, pilots[0], cfg)
    err = np.linalg.norm(h_hat - h_true) / np.linalg.norm(h_true)
    print(f"最小二乘相对误差 = {err:.3e}")
    assert err < 1e-8, err
    print("通过: 无噪最小二乘估计精确")

    print("\n" + "=" * 70)
    print("冒烟测试 2b: 快速最小二乘（缓存傅里叶字典）与显式伪逆一致")
    print("=" * 70)
    from phase1_range import ls_channel_estimate_fast  # noqa: E402

    for l_paths in (64, 227, 512):
        c = OFDMConfig(n_subcarriers=1024, subcarrier_spacing=97_656.25,
                       max_paths=l_paths, cp_len=2048)
        hh = np.zeros(l_paths, dtype=complex)
        idx = rng.choice(l_paths, size=min(5, l_paths), replace=False)
        hh[idx] = rng.standard_normal(idx.size) + 1j * rng.standard_normal(idx.size)
        g = g_matrix(c.n_subcarriers, l_paths)
        yy = np.sqrt(c.tx_power) * (pilots[0] * (g @ hh))
        slow = ls_channel_estimate(yy, pilots[0], c)
        fast = ls_channel_estimate_fast(yy, pilots[0], c)
        d = np.max(np.abs(slow - fast))
        print(f"  L={l_paths:4d}: 慢快两版最大差 = {d:.3e}")
        assert d < 1e-12, (l_paths, d)
        assert np.linalg.norm(fast - hh) / np.linalg.norm(hh) < 1e-8
    print("通过: 快速路径数值上完全相同且精确")

    print("\n" + "=" * 70)
    print("冒烟测试 3: 抽头序号与距离的往返换算")
    print("=" * 70)
    print(f"  无模糊距离上限 = {min(cfg.max_paths, cfg.cp_len)*C0/(2*cfg.bandwidth):.2f} 米")
    for d in [10.0, 30.0, 55.0, 90.0]:
        l = tap_index_from_range(d, cfg)
        r = range_from_tap_index(l, cfg)
        print(f"  d={d:8.3f} 米 -> 抽头 l={l:4d} -> dbar={r:8.3f} 米  |误差|={abs(r-d):.4f} 米")
        assert abs(r - d) <= cfg.delta_d + 1e-9

    print("\n" + "=" * 70)
    print("冒烟测试 3b: 超无模糊距离的混叠能被检出")
    print("=" * 70)
    try:
        cfg.validate_unambiguous(200.0)
        raise AssertionError("超范围距离应当抛出 ValueError")
    except ValueError as exc:
        print(f"  已正确拒绝: {exc}")

    # 论文自己的仿真区域是 R=240 米, 也就是 model.m 里的设定. 在 100 MHz 带宽, 
    # N=1024 的波形下, 要覆盖这么大区域需要更长的 CP 和更多抽头, 比 L=64 多
    # 所以这里用一个缩小后的区域, 让几何和波形彼此自洽
    print("\n  注意: 论文的 R=240 米要求无模糊距离不小于约 340 米（按区域对角算）。")
    print("        所以 100 MHz 波形下这里用 R=90 米，或者用更宽的带宽配 R=240 米，")
    print("        两种配置在 README 里都有说明。")

    print("\n" + "=" * 70)
    print("冒烟测试 4: 阶段一端到端恢复真实距离")
    print("=" * 70)
    # 只接受同时满足论文两条假设的几何:
    #  (a) 可分辨: 每个基站的 K 个距离落在不同的距离单元里
    #      论文 Sec. III-A 里说每个 h_{m,m} 有 K 个非零项;
    #  (b) 无模糊: 每个距离都落在无模糊窗口内, 也就是
    #      min(L,Q)*c0/(2B) 以内
    # 这两条正是官方 model.m 在蒙特卡洛里用的剔除规则,
    # 也就是它的 same_range 判据
    from geometry import has_range_collision  # noqa: E402

    cfg2 = OFDMConfig(n_subcarriers=1024, subcarrier_spacing=97_656.25,
                      n_bs=4, n_targets=4, max_paths=96, cp_len=128)
    limit = min(cfg2.max_paths, cfg2.cp_len) * C0 / (2 * cfg2.bandwidth)
    print(f"L={cfg2.max_paths}，无模糊上限={limit:.2f} 米，Δd={cfg2.delta_d:.4f} 米")

    ok = 0
    trials = 20
    accepted = 0
    seed = 0
    while accepted < trials and seed < 5000:
        r = np.random.default_rng(1000 + seed)
        seed += 1
        geom = random_geometry(4, 4, 90.0, r)
        geom, _ = select_bs_subset(geom, 3)
        d_true = geom.bs_to_target()
        if d_true.max() > limit:
            continue
        clean = np.array(
            [[range_from_tap_index(tap_index_from_range(d, cfg2), cfg2) for d in row]
             for row in d_true]
        )
        if has_range_collision(clean):
            continue
        accepted += 1
        est = phase1_range_estimation(cfg2, geom.bs_xy, geom.target_xy, r, pfa=1e-6)
        good = True
        for m in range(4):
            tb = sorted(clean[m].tolist())
            gb = sorted(est[m].ranges.tolist())
            if len(gb) != 4 or not np.allclose(tb, gb, atol=1e-9):
                good = False
                break
        if good:
            ok += 1
    print(f"阶段一精确单元恢复: {ok}/{accepted} 个有效几何（共抽 {seed} 次）")
    assert accepted == trials, f"only {accepted} valid geometries found"
    assert ok == accepted, f"only {ok}/{accepted}"
    print("通过: 每个有效几何下阶段一都能精确恢复全部 K 个距离")

    print("\n全部冒烟测试通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
