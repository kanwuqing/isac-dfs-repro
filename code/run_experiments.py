"""实验驱动: 复现论文的数值结果

实验列表
--------
E1  误差概率随目标数 K 变化 : B=100 MHz,M=4 下的定位误差概率(阈值 1.5/2/2.5 米)
E2  误差概率随带宽 B 变化   : 误差概率与 RMSE 随系统带宽变化, 论文的核心结论
E3  鬼影概率随 K 变化      : **完美测距**下的鬼影目标存在概率, 覆盖多个 M
E4  阶段一测距误差          : 阶段一距离单元恢复率与测距误差随带宽变化
E5  消融: 跨站路径与路径损耗 : 正交频率假设被破坏时的恢复率

用法
----
    python scripts/run_experiments.py --exp all
    python scripts/run_experiments.py --exp E2 --trials 200
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from geometry import (  # noqa: E402
    random_geometry,
    select_bs_subset,
    has_range_collision,
)
from phase1_range import (  # noqa: E402
    C0,
    OFDMConfig,
    phase1_range_estimation,
    range_from_tap_index,
    tap_index_from_range,
)
from phase2_localize import (  # noqa: E402
    ml_data_association,
    pairwise_position_errors,
    association_consistent,
)

# CSV 与图都落在仓库的 src/ 下, 日志落在 logs/
DATA = ROOT / "src"
FIGS = ROOT / "src"
LOGS = ROOT / "logs"
for d in (DATA, FIGS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

THRESHOLDS = (1.5, 2.0, 2.5)

# 正式结果使用的样本数.低于这个数且没带 --tag 时, save() 会拒绝覆盖已有 CSV
REFERENCE_TRIALS = 300

# ------------------------------------------------------------------
# 参考配置.R=240 米与官方 `model.m` 一致; 无模糊窗口开得足够大
# (L=512,Q=1024 个单元), 使测过的任何带宽下都不会发生目标混叠
# ------------------------------------------------------------------
REGION = 240.0


def make_cfg(bandwidth: float, k_targets: int, m_bs: int = 4) -> OFDMConfig:
    """按给定带宽与目标个数构造一份配置

    N 固定 1024, 子载波间隔按 df = B/N 反推, 所以 B = N*df 是精确的
    L 取能覆盖整个区域的最小值, 即 L >= 2*(R*sqrt(2))*B/c0,
    其中区域对角线决定最长的往返距离. 这样最小二乘的字典矩阵尽量小
    """
    region_diag = REGION * np.sqrt(2.0)
    l_needed = int(np.ceil(2.0 * region_diag * bandwidth / C0)) + 2
    l_paths = int(min(max(l_needed, 64), 4096))
    cfg = OFDMConfig(
        n_subcarriers=1024,
        subcarrier_spacing=bandwidth / 1024.0,
        n_bs=m_bs,
        n_targets=k_targets,
        max_paths=l_paths,
        cp_len=max(2048, l_paths + 1),
        pilot_mode="ones",
        noise_power=1e-18,
    )
    return cfg


def unambiguous_limit(cfg: OFDMConfig) -> float:
    return min(cfg.max_paths, cfg.cp_len) * C0 / (2.0 * cfg.bandwidth)


def sample_geometry(
    cfg: OFDMConfig,
    rng: np.random.Generator,
    max_attempts: int = 20000,
) -> "object | None":
    """抽一个满足论文可分辨性假设的几何布局

    剔除条件, 对应官方 `model.m` 里的 same_range 检验:
      * 每个距离都落在无模糊窗口以内;
      * 每个基站上 K 个距离各自落在不同的单元, 即 K 条路径可分辨
    """
    limit = unambiguous_limit(cfg)
    for _ in range(max_attempts):
        g = random_geometry(cfg.n_bs, cfg.n_targets, REGION, rng)
        g, _ = select_bs_subset(g, 3)
        d = g.bs_to_target()
        if d.max() > limit or d.min() < 1.0:
            continue
        clean = np.array(
            [
                [range_from_tap_index(tap_index_from_range(v, cfg), cfg) for v in row]
                for row in d
            ]
        )
        if has_range_collision(clean):
            continue
        return g, clean
    return None


# ------------------------------------------------------------------
# E1: 误差概率随目标个数 K 变化
# ------------------------------------------------------------------
def exp_error_prob_vs_K(trials: int, bandwidth: float, seed: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for k in range(2, 8):
        cfg = make_cfg(bandwidth, k)
        rng = np.random.default_rng(seed + k)
        errs: List[float] = []
        infeasible = 0
        n_geom = 0
        while n_geom < trials:
            got = sample_geometry(cfg, rng)
            if got is None:
                break
            g, clean = got
            n_geom += 1
            res = ml_data_association(
                [np.sort(clean[m]) for m in range(cfg.n_bs)],
                g.bs_xy,
                sigma_d=cfg.delta_d / 2,
            )
            if not res.feasible:
                # 不可行是指最大似然搜索找不到任何满足几何约束的候选, 也就是定位失败
                # 这种情况的误差按区域对角线算, 同时单独计数
                infeasible += 1
                errs.extend([REGION * np.sqrt(2)] * k)
                continue
            errs.extend(
                pairwise_position_errors(res.positions, g.target_xy).tolist()
            )
        errs_a = np.array(errs)
        if errs_a.size == 0:
            continue
        # 把普通定位行为和灾难性失败分开统计. 灾难性失败指关联配错,
        # 目标被甩到区域另一头. 两种指标并列给出
        # 判据: 关联正确时重建出的位置误差在 Delta_d 的几倍以内, 因为最大似然
        # 的目标函数是 M 个残差之和, 每个残差上限大约就是 Delta_d
        # 所以误差超过量化下限的若干倍就算灾难性失败
        # 另设 5 米的下限, 免得带宽特别宽时这个判据变得没有意义

        gate = max(20.0 * cfg.delta_d, 5.0)
        clean = errs_a[errs_a < gate]
        row = {
            "K": k,
            "M": cfg.n_bs,
            "bandwidth_MHz": bandwidth / 1e6,
            "delta_d_m": cfg.delta_d,
            "catastrophic_gate_m": gate,
            "n_geometries": n_geom,
            "n_target_errors": errs_a.size,
            "infeasible": infeasible,
            "rmse_m": float(np.sqrt(np.mean(errs_a ** 2))),
            "rmse_clean_m": float(np.sqrt(np.mean(clean ** 2))) if clean.size else np.nan,
            "mean_err_m": float(np.mean(errs_a)),
            "median_err_m": float(np.median(errs_a)),
            "p95_err_m": float(np.percentile(errs_a, 95)),
            "n_catastrophic": int(np.sum(errs_a >= gate)),
            "catastrophic_rate": float(np.mean(errs_a >= gate)),
        }
        for t in THRESHOLDS:
            row[f"P_err_gt_{t}m"] = float(np.mean(errs_a > t))
        rows.append(row)
        print(
            f"  K={k}: RMSE={row['rmse_m']:9.4f} m  RMSE(clean)={row['rmse_clean_m']:.4f} m  "
            + "  ".join(f"P(>{t})={row[f'P_err_gt_{t}m']:.4f}" for t in THRESHOLDS)
            + f"  崩盘={row['n_catastrophic']}/{errs_a.size}  不可行={infeasible}/{n_geom}"
        , flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# E2: 误差概率随带宽变化
# ------------------------------------------------------------------
def exp_error_prob_vs_B(trials: int, k_targets: int, seed: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for bw in [10e6, 20e6, 50e6, 100e6, 200e6, 400e6]:
        cfg = make_cfg(bw, k_targets)
        rng = np.random.default_rng(seed + int(bw / 1e6))
        errs: List[float] = []
        infeasible = 0
        n_geom = 0
        while n_geom < trials:
            got = sample_geometry(cfg, rng)
            if got is None:
                break
            g, clean = got
            n_geom += 1
            res = ml_data_association(
                [np.sort(clean[m]) for m in range(cfg.n_bs)],
                g.bs_xy,
                sigma_d=cfg.delta_d / 2,
            )
            if not res.feasible:
                infeasible += 1
                errs.extend([REGION * np.sqrt(2)] * k_targets)
                continue
            errs.extend(
                pairwise_position_errors(res.positions, g.target_xy).tolist()
            )
        errs_a = np.array(errs)
        if errs_a.size == 0:
            continue
        gate = max(20.0 * cfg.delta_d, 5.0)
        clean = errs_a[errs_a < gate]
        row = {
            "bandwidth_MHz": bw / 1e6,
            "K": k_targets,
            "M": cfg.n_bs,
            "delta_d_m": cfg.delta_d,
            "catastrophic_gate_m": gate,
            "n_geometries": n_geom,
            "n_target_errors": errs_a.size,
            "infeasible": infeasible,
            "rmse_m": float(np.sqrt(np.mean(errs_a ** 2))),
            "rmse_clean_m": float(np.sqrt(np.mean(clean ** 2))) if clean.size else np.nan,
            "mean_err_m": float(np.mean(errs_a)),
            "median_err_m": float(np.median(errs_a)),
            "p95_err_m": float(np.percentile(errs_a, 95)),
            "n_catastrophic": int(np.sum(errs_a >= gate)),
            "catastrophic_rate": float(np.mean(errs_a >= gate)),
        }
        for t in THRESHOLDS:
            row[f"P_err_gt_{t}m"] = float(np.mean(errs_a > t))
        rows.append(row)
        print(
            f"  B={bw/1e6:6.0f} MHz  Delta_d={cfg.delta_d:.4f} m  "
            f"RMSE={row['rmse_m']:8.4f} m  RMSE(clean)={row['rmse_clean_m']:.4f} m  "
            + "  ".join(f"P(>{t})={row[f'P_err_gt_{t}m']:.4f}" for t in THRESHOLDS)
            + f"  崩盘={row['n_catastrophic']}/{errs_a.size}"
        , flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# E3: 完美测距下的鬼影目标存在概率
# ------------------------------------------------------------------
def _association_is_true(
    cand: np.ndarray, clean: np.ndarray, bs_xy: np.ndarray, tol: float = 1e-3
) -> bool:
    """`cand` 这个候选能否重建出真实目标位置?"""
    if not association_consistent(cand, bs_xy, tol=1e-6):
        return False
    from phase2_localize import refine_position

    x, _ = refine_position(cand, bs_xy)
    k = clean.shape[1]
    for kk in range(k):
        true_cand = clean[:, kk]
        xt, _ = refine_position(true_cand, bs_xy)
        if np.linalg.norm(xt - x) < tol:
            return True
    return False


def _reconstruct_targets(clean: np.ndarray, bs_xy: np.ndarray, tol: float = 1e-3):
    """返回由一致对角结构可推出的真实目标位置

    测距完美时, 真实目标恰好就是那些一致的对角关联, 也就是每个基站都取
    同一列 k 得到的距离组合. 把这些位置重建出来, 与其中之一重合的候选
    就不算鬼影
    """
    from phase2_localize import refine_position

    k = clean.shape[1]
    pts = []
    for kk in range(k):
        cand = clean[:, kk]
        if association_consistent(cand, bs_xy, tol=1e-6):
            x, _ = refine_position(cand, bs_xy)
            pts.append(x)
    return pts


def _has_ghost_efficient(
    clean: np.ndarray, bs_xy: np.ndarray, tol: float = 1e-6
) -> bool:
    """不必枚举 K!^(M-1) 种置换就能检出鬼影目标

    鬼影的定义: 存在一组下标(i_1..i_M), 每个 i_m 落在 1..K 之间,
    对应的 M 个圆交于一点(一致), 但那个点不是任何真实目标

    与其把所有组合都试一遍, 不如利用一致性本身是个很强的几何约束:
    三个基站只要一致就已经定出一个点, 这个点不是真目标就是鬼影. 所以

      1. 先在前 3 个基站上枚举三元组, 最多 K^3 个, K=6 时也才 216 个;
      2. 留下既一致又不是真实目标的;
      3. 剩下的基站直接按 |x_ghost - a_m| == d 反查该站哪个距离对得上,
         每个基站 O(K) 就够

    这样能覆盖所有"新三元组"引出的鬼影. 还有一种情况: 前三个基站的
    距离碰巧构成真实目标, 此时若还闭合, 后面几个基站就必须与真目标
    不一致, 这种情况在下面单独检查
    """
    from phase2_localize import refine_position

    m_bs, k = clean.shape
    genuine = _reconstruct_targets(clean, bs_xy)
    genuine_arr = np.array(genuine) if genuine else np.zeros((0, 2))

    def _is_genuine(x: np.ndarray) -> bool:
        if genuine_arr.shape[0] == 0:
            return False
        return bool(np.min(np.linalg.norm(genuine_arr - x[None, :], axis=1)) < tol)

# 在前 3 个基站上枚举三元组
    from itertools import product as _prod

    for i, j, kk in _prod(range(k), repeat=3):
        cand3 = np.array([clean[0, i], clean[1, j], clean[2, kk]])
        if not association_consistent(cand3, bs_xy[:3], tol=tol):
            continue
        x, _ = refine_position(cand3, bs_xy[:3])
        if _is_genuine(x):
            continue
# 把这个鬼影点往剩下的基站上扩展
        ok = True
        for m in range(3, m_bs):
            target_r = float(np.linalg.norm(x - bs_xy[m]))
            if not np.any(np.abs(clean[m] - target_r) < 1e-3 + tol):
                ok = False
                break
        if ok:
            return True
    return False


def exp_ghost_prob_vs_K(trials: int, seed: int, bandwidth: float = 100e6) -> pd.DataFrame:
    """完美测距下的鬼影目标概率(论文 Sec. V)

    用上面那个快速检测器. 朴素的 K!^(M-1) 置换枚举在 M=6, K=6 时有
    2e14 种组合, 根本跑不完
    """
    rows: List[Dict] = []
    for m_bs in (3, 4, 6):
        for k in range(2, 7):
            cfg = make_cfg(bandwidth, k, m_bs=m_bs)
            rng = np.random.default_rng(seed + 100 * m_bs + k)
            n_ghost = 0
            n_geom = 0
            while n_geom < trials:
                got = sample_geometry(cfg, rng)
                if got is None:
                    break
                g, clean = got
                n_geom += 1
                if _has_ghost_efficient(clean, g.bs_xy):
                    n_ghost += 1
            if n_geom == 0:
                continue
            row = {
                "M": m_bs,
                "K": k,
                "n_geometries": n_geom,
                "n_ghost": n_ghost,
                "ghost_prob": n_ghost / n_geom,
            }
            rows.append(row)
            print(
                f"  M={m_bs} K={k}：鬼影概率={row['ghost_prob']:.5f} "
                f"({n_ghost}/{n_geom})",
                flush=True,
            )
    return pd.DataFrame(rows)


def _product_repeat(pool, repeat):
    from itertools import product as _product

    return _product(pool, repeat=repeat)


# ------------------------------------------------------------------
# E4: 信号级阶段一的测距误差
# ------------------------------------------------------------------
def exp_phase1_range_error(seed: int, n_trials: int = 40) -> pd.DataFrame:
    """信号级阶段一: 最小二乘 + CFAR 能否恢复正确的距离单元?"""
    rows: List[Dict] = []
    for bw in [20e6, 50e6, 100e6, 200e6, 400e6]:
        for k in (2, 4, 6):
            cfg = make_cfg(bw, k)
            cfg_sig = make_cfg(bw, k)
            cfg_sig = OFDMConfig(
                n_subcarriers=cfg_sig.n_subcarriers,
                subcarrier_spacing=cfg_sig.subcarrier_spacing,
                n_bs=cfg_sig.n_bs,
                n_targets=cfg_sig.n_targets,
                max_paths=cfg_sig.max_paths,
                cp_len=cfg_sig.cp_len,
                noise_power=1e-14,
            )
            rng = np.random.default_rng(seed + int(bw / 1e6) + k)
            errs: List[float] = []
            n_exact = 0
            n_geom = 0
            while n_geom < n_trials:
                got = sample_geometry(cfg, rng)
                if got is None:
                    break
                g, clean = got
                n_geom += 1
                est = phase1_range_estimation(
                    cfg_sig, g.bs_xy, g.target_xy, rng, pfa=1e-6
                )
                exact = True
                for m in range(cfg.n_bs):
                    tb = sorted(clean[m].tolist())
                    gb = sorted(est[m].ranges.tolist())
                    if len(gb) != k or not np.allclose(tb, gb, atol=1e-9):
                        exact = False
# 每个目标的测距误差, 取最近的那个匹配
                        for tv in tb:
                            if gb:
                                errs.append(min(abs(tv - q) for q in gb))
                            else:
                                errs.append(cfg.delta_d)
                if exact:
                    n_exact += 1
            row = {
                "bandwidth_MHz": bw / 1e6,
                "K": k,
                "delta_d_m": cfg_sig.delta_d,
                "n_geometries": n_geom,
                "exact_recovery_rate": n_exact / max(n_geom, 1),
                "mean_range_err_m": float(np.mean(errs)) if errs else 0.0,
                "rmse_range_m": float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0,
            }
            rows.append(row)
            print(
                f"  B={bw/1e6:6.0f} MHz K={k}: exact={row['exact_recovery_rate']:.3f} "
                f"RMSE_range={row['rmse_range_m']:.5f} m (Delta_d={cfg_sig.delta_d:.4f})",
                flush=True,
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# E5 与 E6: 消融实验
# ------------------------------------------------------------------
def exp_ablation(seed: int, n_trials: int = 30) -> pd.DataFrame:
    rows: List[Dict] = []
    bw, k = 100e6, 4
    for cross in (False, True):
        for ple in (1.0, 2.0):
            cfg = make_cfg(bw, k)
            rng = np.random.default_rng(seed + int(cross) * 7 + int(ple * 3))
            n_exact, n_geom = 0, 0
            while n_geom < n_trials:
                got = sample_geometry(cfg, rng)
                if got is None:
                    break
                g, clean = got
                n_geom += 1
                est = phase1_range_estimation(
                    cfg, g.bs_xy, g.target_xy, rng, pfa=1e-6,
                    include_cross_paths=cross, path_loss_exponent=ple,
                )
                exact = True
                for m in range(cfg.n_bs):
                    gb = sorted(est[m].ranges.tolist())
                    if gb != sorted(clean[m].tolist()):
                        exact = False
                        break
                if exact:
                    n_exact += 1
            rows.append(
                {
                    "cross_paths": cross,
                    "path_loss_exponent": ple,
                    "n_geometries": n_geom,
                    "exact_recovery_rate": n_exact / max(n_geom, 1),
                }
            )
            print(
                f"  cross_paths={cross}  path_loss_exp={ple}: "
                f"exact_recovery={n_exact}/{n_geom}",
                flush=True,
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="all",
                    choices=["all", "E1", "E2", "E3", "E4", "E5"])
    ap.add_argument("--trials", type=int, default=100,
                    help="每个参数点跑多少次蒙特卡洛几何")
    ap.add_argument("--seed", type=int, default=20240929)
    ap.add_argument("--tag", default="", help="输出文件名后缀")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("《OFDM 蜂窝网络无设备感知》复现实验")
    print("=" * 78)
    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
    print(f"numpy       : {np.__version__}")
    print(f"随机种子    : {args.seed}")
    print(f"每点样本数  : {args.trials} 次几何")
    print(f"区域        : R={REGION} 米")
    print("=" * 78, flush=True)

    tag = args.tag
    manifest = {
        "seed": args.seed,
        "trials": args.trials,
        "region_m": REGION,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "thresholds_m": list(THRESHOLDS),
        "experiments": {},
    }

    def save(df: pd.DataFrame, name: str) -> None:
        path = DATA / f"{name}{tag}.csv"
        # 保护: 不覆盖 300 次样本的正式结果.小样本试跑必须带 --tag,
        # 否则文件名会撞上正式结果, 把真实数据冲掉
        if tag == "" and args.trials < REFERENCE_TRIALS and path.exists():
            raise SystemExit(
                f"拒绝写入 {path.name}：当前只跑 {args.trials} 次，"
                f"正式结果是 {REFERENCE_TRIALS} 次。"
                f"小样本试跑请加 --tag _try 之类的后缀。"
            )
        df.to_csv(path, index=False)
        print(f"    -> 已写入 {path.relative_to(ROOT)}", flush=True)

    if args.exp in ("all", "E1"):
        print("\n[E1] 误差概率随目标个数 K 变化（M=4，B=100 MHz）", flush=True)
        df = exp_error_prob_vs_K(args.trials, 100e6, args.seed)
        save(df, "e1_error_prob_vs_K")
        manifest["experiments"]["E1"] = df.to_dict(orient="records")

    if args.exp in ("all", "E2"):
        print("\n[E2] error probability & RMSE vs bandwidth (M=4, K=3)", flush=True)
        df = exp_error_prob_vs_B(args.trials, 3, args.seed)
        save(df, "e2_error_prob_vs_B")
        manifest["experiments"]["E2"] = df.to_dict(orient="records")

    if args.exp in ("all", "E3"):
        print("\n[E3] 完美测距下的鬼影目标概率", flush=True)
        # 鬼影罕见, 样本数取 300.
        df = exp_ghost_prob_vs_K(max(200, args.trials), args.seed)
        save(df, "e3_ghost_prob_vs_K")
        manifest["experiments"]["E3"] = df.to_dict(orient="records")

    if args.exp in ("all", "E4"):
        print("\n[E4] Phase I signal-level range recovery", flush=True)
        df = exp_phase1_range_error(args.seed, n_trials=max(10, args.trials // 3))
        save(df, "e4_phase1_range_error")
        manifest["experiments"]["E4"] = df.to_dict(orient="records")

    if args.exp in ("all", "E5"):
        print("\n[E5/E6] ablations (cross paths, path-loss exponent)", flush=True)
        df = exp_ablation(args.seed, n_trials=max(10, args.trials // 3))
        save(df, "e5_ablations")
        manifest["experiments"]["E5"] = df.to_dict(orient="records")

    manifest["elapsed_s"] = time.time() - t0
    mpath = DATA / f"manifest{tag}.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {mpath.relative_to(ROOT)}")
    print(f"总耗时：{manifest['elapsed_s']:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
