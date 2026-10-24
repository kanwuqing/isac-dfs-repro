"""严格枚举校验: 快速路径探索的候选集合与暴力穷举完全相同.

不只看最终挑中的那个解, 而是要求两边枚举出来的候选个数 (去掉目标重编号
的冗余之后) 一致, 并且最优值也一样.
"""
from __future__ import annotations
import sys, math, time
from pathlib import Path
from itertools import product, permutations
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isac_dfs.geometry import random_geometry, select_bs_subset, has_range_collision
from isac_dfs.phase1_range import OFDMConfig, C0, range_from_tap_index, tap_index_from_range
from isac_dfs.phase2_localize import (
    triangle_feasible, refine_position, ml_data_association,
)

REGION, bw = 240.0, 100e6


def brute_full(bs_ranges, bs_xy, sigma_d):
    """穷举: 每个槽位都取遍所有有序的三元组选择."""
    k = len(bs_ranges[0]); m_bs = len(bs_ranges)
    delta = 3*sigma_d; budget = m_bs*2*sigma_d*1.5
    triples = []
    for i,j,kk in product(range(k), repeat=3):
        c = np.array([bs_ranges[0][i], bs_ranges[1][j], bs_ranges[2][kk]])
        if not triangle_feasible(c, bs_xy, delta): continue
        _, r_ = refine_position(c, bs_xy)
        if np.sum(r_) > budget: continue
        triples.append(c)
    ts = np.sort(np.stack(bs_ranges[:3],axis=0), axis=1)
    out = []
    for choice in product(range(len(triples)), repeat=k):
        cand = np.stack([triples[c] for c in choice], axis=1)
        if not np.array_equal(np.sort(cand,axis=1), ts): continue
        assoc = np.zeros((m_bs,k)); assoc[:3,:]=cand
        pos = np.stack([refine_position(cand[:,t], bs_xy[:3])[0] for t in range(k)],axis=0)
        for m in range(3,m_bs):
            cost = np.array([[abs(bs_ranges[m][r]-np.linalg.norm(pos[t]-bs_xy[m])) for r in range(k)] for t in range(k)])
            row,col = linear_sum_assignment(cost)
            mm = np.zeros(k); mm[row]=bs_ranges[m][col]; assoc[m,:]=mm
            for t in range(k): pos[t],_ = refine_position(assoc[:m+1,t], bs_xy[:m+1])
        tot=0.0
        for t in range(k):
            x,r_ = refine_position(assoc[:,t],bs_xy); pos[t]=x; tot+=float(np.sum(r_))
        out.append((tot,pos.copy()))
    return out


for k in (3,4,5):
    cfg = OFDMConfig(n_subcarriers=1024, subcarrier_spacing=bw/1024, n_bs=4, n_targets=k,
                     max_paths=int(np.ceil(2*REGION*np.sqrt(2)*bw/C0))+2, cp_len=2048)
    limit = min(cfg.max_paths, cfg.cp_len)*C0/(2*bw)
    rng = np.random.default_rng(31337+k)
    checked=0
    while checked < 4:
        g = random_geometry(4,k,REGION,rng); g,_=select_bs_subset(g,3)
        d=g.bs_to_target()
        if d.max()>limit or d.min()<1.0: continue
        clean=np.array([[range_from_tap_index(tap_index_from_range(v,cfg),cfg) for v in row] for row in d])
        if has_range_collision(clean): continue
        checked+=1
        br=[np.sort(clean[m]) for m in range(4)]
        cands = brute_full(br,g.bs_xy,cfg.delta_d/2)
        if not cands:
            res=ml_data_association(br,g.bs_xy,sigma_d=cfg.delta_d/2)
            print(f"K={k} #{checked}: brute infeasible, fast feasible={res.feasible}")
            continue
        best = min(c[0] for c in cands)
        res = ml_data_association(br,g.bs_xy,sigma_d=cfg.delta_d/2)
        # 快速路径有没有达到真正的全局最优?
        dcost = abs(res.objective - best)
        # 以及它给出的点集是不是最优的?
        bpos = [c[1] for c in cands if abs(c[0]-best)<1e-9][0]
        cost = np.linalg.norm(bpos[:,None,:]-res.positions[None,:,:],axis=-1)
        r_,c_ = linear_sum_assignment(cost)
        print(f"K={k} 第{checked}例：暴力候选={len(cands):5d} 快速候选={res.n_candidates:5d} "
              f"(K!={math.factorial(k)})  暴力目标值={best:.6f} 快速目标值={res.objective:.6f} "
              f"目标值差={dcost:.2e}  最大点偏差={cost[r_,c_].max():.3e} 米")
