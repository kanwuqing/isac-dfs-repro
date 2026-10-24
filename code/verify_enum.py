"""校验剪枝后的枚举与暴力穷举一致, 并测加速比."""
from __future__ import annotations
import sys, time
from pathlib import Path
from itertools import product
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isac_dfs.geometry import random_geometry, select_bs_subset, has_range_collision
from isac_dfs.phase1_range import OFDMConfig, C0, range_from_tap_index, tap_index_from_range
from isac_dfs.phase2_localize import (
    ml_data_association, pairwise_position_errors, triangle_feasible, refine_position,
)

REGION = 240.0
bw = 100e6

def brute(bs_ranges, bs_xy, sigma_d):
    """参考实现: 在三元组上做完整的 K^K 笛卡尔积."""
    k = len(bs_ranges[0]); m_bs = len(bs_ranges)
    delta = 3*sigma_d; budget = m_bs*2*sigma_d*1.5
    triples = []
    for i,j,kk in product(range(k), repeat=3):
        cand = np.array([bs_ranges[0][i], bs_ranges[1][j], bs_ranges[2][kk]])
        if not triangle_feasible(cand, bs_xy, delta): continue
        _, r_ = refine_position(cand, bs_xy)
        if np.sum(r_) > budget: continue
        triples.append(cand)
    if not triples: return None, 0
    target_sorted = np.sort(np.stack(bs_ranges[:3], axis=0), axis=1)
    partials = []
    for choice in product(range(len(triples)), repeat=k):
        cand = np.stack([triples[c] for c in choice], axis=1)
        if np.array_equal(np.sort(cand, axis=1), target_sorted):
            partials.append(cand)
    if not partials: return None, 0
    scored = []
    for part in partials:
        assoc = np.zeros((m_bs,k)); assoc[:3,:] = part
        pos = np.stack([refine_position(part[:,t], bs_xy[:3])[0] for t in range(k)], axis=0)
        for m in range(3, m_bs):
            cost = np.array([[abs(bs_ranges[m][r]-np.linalg.norm(pos[t]-bs_xy[m])) for r in range(k)] for t in range(k)])
            from scipy.optimize import linear_sum_assignment
            row,col = linear_sum_assignment(cost)
            matched = np.zeros(k); matched[row] = bs_ranges[m][col]; assoc[m,:] = matched
            for t in range(k): pos[t],_ = refine_position(assoc[:m+1,t], bs_xy[:m+1])
        tot = 0.0
        for t in range(k):
            x,r_ = refine_position(assoc[:,t], bs_xy); pos[t]=x; tot += float(np.sum(r_))
        scored.append((tot, pos.copy()))
    scored.sort(key=lambda z: z[0])
    return scored[0][1], len(partials)

for k in (3, 4, 5):
    cfg = OFDMConfig(n_subcarriers=1024, subcarrier_spacing=bw/1024, n_bs=4, n_targets=k,
                     max_paths=int(np.ceil(2*REGION*np.sqrt(2)*bw/C0))+2, cp_len=2048)
    limit = min(cfg.max_paths, cfg.cp_len)*C0/(2*bw)
    rng = np.random.default_rng(777+k)
    n_checked = 0
    while n_checked < 5:
        g = random_geometry(4, k, REGION, rng); g,_ = select_bs_subset(g,3)
        d = g.bs_to_target()
        if d.max() > limit or d.min() < 1.0: continue
        clean = np.array([[range_from_tap_index(tap_index_from_range(v,cfg),cfg) for v in row] for row in d])
        if has_range_collision(clean): continue
        n_checked += 1
        br = [np.sort(clean[m]) for m in range(4)]
        t0=time.time(); bpos, npart_b = brute(br, g.bs_xy, cfg.delta_d/2); tb=time.time()-t0
        t0=time.time(); res = ml_data_association(br, g.bs_xy, sigma_d=cfg.delta_d/2); tf=time.time()-t0
        if bpos is None and not res.feasible:
            print(f"K={k} 第{n_checked}例：两者均不可行（暴力 {tb:.2f} 秒，快速 {tf:.3f} 秒）"); continue
        if bpos is None or not res.feasible:
            print(f"K={k} #{n_checked}: MISMATCH feasibility brute={bpos is not None} fast={res.feasible}"); continue
        # 按无序的点集来比对
        from scipy.optimize import linear_sum_assignment
        cost = np.linalg.norm(bpos[:,None,:]-res.positions[None,:,:],axis=-1)
        r_,c_ = linear_sum_assignment(cost)
        dev = cost[r_,c_].max()
        # 候选个数: 暴力法枚举带标签的指派, 其中有 K! 倍的重复;
        # 快速路径把第 0 站的顺序钉死, 所以期望 快速 == 暴力 / K!.
        import math
        ratio = npart_b / max(res.n_candidates, 1)
        flag = "一致" if dev < 1e-6 else "有差异"
        print(f"K={k} 第{n_checked}例：暴力候选={npart_b:5d} 快速候选={res.n_candidates:5d} "
              f"倍数={ratio:8.1f} (K!={math.factorial(k)})  "
              f"最大点偏差={dev:.3e} 米  耗时 暴力={tb:6.2f}秒 快速={tf:6.3f}秒  {flag}")
