"""由 src/ 下的 CSV 重新生成全部图

运行: python code/make_figures.py
所有图写入 src/,PNG 为 300 dpi, 同时导出 PDF
图中所有文字均为中文
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "src"
FIGS = ROOT / "src"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        # 中文优先, 负号正常显示
        "font.sans-serif": [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans SC",
            "DengXian",
            "Arial",
        ],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "figure.autolayout": True,
    }
)

THRESHOLDS = (1.5, 2.0, 2.5)
COLORS = {1.5: "#1f77b4", 2.0: "#ff7f0e", 2.5: "#2ca02c"}


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  已写入 src/{name}.png、src/{name}.pdf")


def _load(name: str) -> pd.DataFrame | None:
    p = DATA / f"{name}.csv"
    if not p.exists():
        print(f"  [跳过] 未找到 {p.name}")
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
def fig_error_prob_vs_K(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for t in THRESHOLDS:
        col = f"P_err_gt_{t}m"
        if col in df:
            ax.semilogy(df["K"], df[col].replace(0, np.nan), "o-",
                        color=COLORS[t], label=f"误差 > {t:g} 米", markersize=5)
    ax.set_xlabel("目标个数 K")
    ax.set_ylabel("定位误差概率")
    ax.set_title(f"(a) 误差概率随目标个数变化"
                 f"（M={int(df['M'].iloc[0])}，B={df['bandwidth_MHz'].iloc[0]:g} MHz）")
    ax.set_xticks(df["K"])
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=1e-4)

    ax = axes[1]
    ax.plot(df["K"], df["rmse_clean_m"], "s-", color="#1f77b4",
            label="RMSE（正常样本）")
    ax.plot(df["K"], df["rmse_m"], "^--", color="#d62728",
            label="RMSE（含崩盘样本）")
    ax.axhline(df["delta_d_m"].iloc[0], color="grey", ls=":",
               label="Δd（距离量化下限）")
    ax.set_xlabel("目标个数 K")
    ax.set_ylabel("定位 RMSE（米）")
    ax.set_yscale("log")
    ax.set_title("(b) 定位 RMSE 随目标个数变化")
    ax.set_xticks(df["K"])
    ax.legend(fontsize=8)

    _save(fig, "fig1_error_prob_vs_K")


def fig_error_prob_vs_B(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for t in THRESHOLDS:
        col = f"P_err_gt_{t}m"
        if col in df:
            ax.loglog(df["bandwidth_MHz"], df[col].replace(0, np.nan), "o-",
                      color=COLORS[t], label=f"误差 > {t:g} 米", markersize=5)
    ax.set_xlabel("系统带宽 B（MHz）")
    ax.set_ylabel("定位误差概率")
    ax.set_title(f"(a) 误差概率随带宽变化（K={int(df['K'].iloc[0])}）")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.loglog(df["bandwidth_MHz"], df["rmse_clean_m"], "s-", color="#1f77b4",
              label="RMSE（正常样本）")
    ax.loglog(df["bandwidth_MHz"], df["rmse_m"], "^--", color="#d62728",
              label="RMSE（含崩盘样本）")
    ax.loglog(df["bandwidth_MHz"], df["delta_d_m"], "k:", label="Δd = c/(4B)")
    ax.set_xlabel("系统带宽 B（MHz）")
    ax.set_ylabel("RMSE（米）")
    ax.set_title("(b) RMSE 随带宽变化")
    ax.legend(fontsize=8)

    _save(fig, "fig2_error_prob_vs_bandwidth")


def fig_ghost(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m in sorted(df["M"].unique()):
        sub = df[df["M"] == m].sort_values("K")
        y = sub["ghost_prob"].to_numpy(dtype=float)
        n = sub["n_geometries"].to_numpy(dtype=float)
        # 观测到 0 次时, 画 95% 上界(三倍法则 3/n)
        y_plot = np.where(y > 0, y, 3.0 / n)
        ax.semilogy(sub["K"], y_plot, "o-", label=f"M={m}")
    ax.set_xlabel("目标个数 K")
    ax.set_ylabel("鬼影目标存在概率")
    ax.set_title("完美测距下的鬼影目标\n（值为 0 的点画的是 95% 上界 3/n）")
    ax.legend()
    _save(fig, "fig3_ghost_prob_vs_K")


def fig_phase1(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for k in sorted(df["K"].unique()):
        sub = df[df["K"] == k].sort_values("bandwidth_MHz")
        ax.semilogx(sub["bandwidth_MHz"], sub["exact_recovery_rate"], "o-",
                    label=f"K={k}")
    ax.set_xlabel("系统带宽 B（MHz）")
    ax.set_ylabel("距离单元精确恢复率")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(a) 阶段一抽头检测准确度")
    ax.legend(fontsize=8)

    ax = axes[1]
    sub = df[df["K"] == df["K"].min()].sort_values("bandwidth_MHz")
    ax.loglog(sub["bandwidth_MHz"], sub["rmse_range_m"], "s-",
              label="实测测距 RMSE")
    ax.loglog(sub["bandwidth_MHz"], sub["delta_d_m"], "k:", label="Δd = c/(4B)")
    ax.set_xlabel("系统带宽 B（MHz）")
    ax.set_ylabel("测距误差（米）")
    ax.set_title("(b) 阶段一测距误差随带宽变化")
    ax.legend(fontsize=8)

    _save(fig, "fig4_phase1_range_error")


def fig_ablation(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = []
    vals = []
    for _, r in df.iterrows():
        cross_txt = "跨站混入" if r["cross_paths"] else "无跨站"
        labels.append(f"{cross_txt}\n路损指数 {r['path_loss_exponent']:g}")
        vals.append(r["exact_recovery_rate"])
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("距离单元精确恢复率")
    ax.set_ylim(0, 1.05)
    ax.set_title("消融实验：正交频率假设与路径损耗假设")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=8)
    _save(fig, "fig5_ablations")


def fig_summary_scatter() -> None:
    """定性定位散点图: 真实位置与估计位置对比."""
    sys.path.insert(0, str(HERE))
    from geometry import random_geometry, select_bs_subset, has_range_collision
    from phase1_range import (
        C0, range_from_tap_index, tap_index_from_range,
    )
    from phase2_localize import ml_data_association
    from run_experiments import make_cfg, REGION

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=False)
    k = 4
    for ax, bw in zip(axes, (10e6, 50e6, 400e6)):
        cfg = make_cfg(bw, k)
        rng = np.random.default_rng(2024)
        limit = min(cfg.max_paths, cfg.cp_len) * C0 / (2 * cfg.bandwidth)
        got = None
        for _ in range(20000):
            g = random_geometry(4, k, REGION, rng)
            g, _ = select_bs_subset(g, 3)
            d = g.bs_to_target()
            if d.max() > limit or d.min() < 1.0:
                continue
            cl = np.array([[range_from_tap_index(tap_index_from_range(v, cfg), cfg)
                            for v in row] for row in d])
            if has_range_collision(cl):
                continue
            got = (g, cl)
            break
        if got is None:
            continue
        g, cl = got
        res = ml_data_association([np.sort(cl[m]) for m in range(4)], g.bs_xy,
                                  sigma_d=cfg.delta_d / 2)
        ax.scatter(g.bs_xy[:, 0], g.bs_xy[:, 1], marker="^", s=90, c="black",
                   label="基站", zorder=5)
        ax.scatter(g.target_xy[:, 0], g.target_xy[:, 1], marker="o", s=80,
                   facecolors="none", edgecolors="#1f77b4", linewidths=1.8,
                   label="真实目标", zorder=4)
        if res.feasible:
            ax.scatter(res.positions[:, 0], res.positions[:, 1], marker="x", s=70,
                       c="#d62728", linewidths=2, label="估计位置", zorder=6)
        ax.set_title(f"B={bw/1e6:g} MHz，Δd={cfg.delta_d:.2f} 米")
        ax.set_xlabel("x（米）")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("y（米）")
    _save(fig, "fig6_localisation_scatter")


def fig_rmse_vs_deltad() -> None:
    """RMSE 与量化下限 Δd 的关系, 展示线性趋势."""
    df = _load("e2_error_prob_vs_B")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.loglog(df["delta_d_m"], df["rmse_clean_m"], "o-", color="#1f77b4",
              label="RMSE（正常样本）")
    lim = [df["delta_d_m"].min() * 0.5, df["delta_d_m"].max() * 2]
    ax.loglog(lim, lim, "k--", lw=1, label="RMSE = Δd")
    ax.set_xlabel("量化下限 Δd = c/(4B)（米）")
    ax.set_ylabel("RMSE（米）")
    ax.set_title("定位 RMSE 紧贴距离量化下限")
    ax.legend(fontsize=8)
    _save(fig, "fig7_rmse_vs_deltad")


def main() -> int:
    print("正在由 src/ 生成图像 ...")
    d1 = _load("e1_error_prob_vs_K")
    if d1 is not None:
        fig_error_prob_vs_K(d1)
    d2 = _load("e2_error_prob_vs_B")
    if d2 is not None:
        fig_error_prob_vs_B(d2)
        fig_rmse_vs_deltad()
    d3 = _load("e3_ghost_prob_vs_K")
    if d3 is not None:
        fig_ghost(d3)
    d4 = _load("e4_phase1_range_error")
    if d4 is not None:
        fig_phase1(d4)
    d5 = _load("e5_ablations")
    if d5 is not None:
        fig_ablation(d5)
    fig_summary_scatter()
    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
