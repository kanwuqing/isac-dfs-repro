"""无设备感知仿真所需的几何工具和共用函数."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Geometry:
    """2D 区域内基站和目标的坐标."""

    bs_xy: np.ndarray  # (M, 2) 基站坐标
    target_xy: np.ndarray  # (K, 2) 目标坐标
    region_size: float  # R: 正方形区域为 [-R/2, R/2]^2

    @property
    def n_bs(self) -> int:
        """基站个数."""
        return self.bs_xy.shape[0]

    @property
    def n_targets(self) -> int:
        """目标个数."""
        return self.target_xy.shape[0]

    def bs_to_target(self) -> np.ndarray:
        """(M, K) 矩阵: 基站到目标的距离 d_{m,k}, 见论文 Sec. II-A."""
        return np.linalg.norm(self.bs_xy[:, None, :] - self.target_xy[None, :, :], axis=-1)

    def bs_to_bs(self) -> np.ndarray:
        """(M, M) 矩阵: 基站到基站的距离 d^BS_{m,u}."""
        return np.linalg.norm(self.bs_xy[:, None, :] - self.bs_xy[None, :, :], axis=-1)


def random_geometry(
    m_bs: int,
    k_targets: int,
    region_size: float,
    rng: np.random.Generator,
) -> Geometry:
    """在 [-R/2, R/2]^2 内均匀撒基站和目标, 见论文 Sec. VI-B."""
    bs = region_size * (rng.random((m_bs, 2)) - 0.5)  # 基站位置, 平移到以原点为中心
    tg = region_size * (rng.random((k_targets, 2)) - 0.5)  # 目标位置
    return Geometry(bs_xy=bs, target_xy=tg, region_size=region_size)


def select_bs_subset(geom: Geometry, n_select: int = 3) -> Tuple[Geometry, np.ndarray]:
    """重排基站, 让前 n_select 个是条件数最好的

    官方 MATLAB 的 model.m 做法是选三个两两距离最接近的基站,
    也就是最小化三角形边长的离散程度, 这样三边定位数值稳定; 这里复现该启发式
    """
    d = geom.bs_to_bs()  # 站间距离表
    m = geom.n_bs
    best_idx, best_spread = None, np.inf
    from itertools import combinations

    for combo in combinations(range(m), n_select):
        vals = [d[i, j] for i, j in combinations(combo, 2)]  # 三条边
        spread = sum(abs(vals[0] - v) for v in vals)  # 边长差得越多越不稳
        if spread < best_spread:
            best_spread, best_idx = spread, combo
    rest = [i for i in range(m) if i not in best_idx]  # 剩下的基站排在后面
    perm = np.array(list(best_idx) + rest)
    return (
        Geometry(geom.bs_xy[perm], geom.target_xy, geom.region_size),
        perm,
    )


def has_range_collision(estimated_ranges: np.ndarray, tol: float = 0.0) -> bool:
    """若任一基站看到两个目标落在同一个距离单元里, 返回 True

    论文假设每个基站的 K 条路径可分辨, 也就是两个不同目标的距离差要大于
    2*Delta_d; 违反该假设的几何在蒙特卡洛循环里会被丢弃, 和 model.m 一致
    """
    m = estimated_ranges.shape[0]
    for i in range(m):
        row = np.sort(estimated_ranges[i])  # 本站的 K 个距离排好
        if np.any(np.diff(row) <= tol):
            return True  # 有任意两个挤在同一格
    return False


def add_laplace_or_uniform_errors(
    clean_ranges: np.ndarray,
    rng: np.random.Generator,
    delta_d: float,
    model: str = "quantization",
) -> np.ndarray:
    """给干净的真实距离套一个误差模型

    论文名义上的阶段一误差是式(7) 的量化误差, 确定性且上界为
    Delta_d = c0/(4*N*df); 带宽扫描时精确复现该模型, uniform 作为鲁棒性检查
    """
    if model == "quantization":
        return clean_ranges  # 量化误差已经在距离里了, 不用再加
    if model == "uniform":
        return clean_ranges + rng.uniform(-delta_d, delta_d, size=clean_ranges.shape)
    raise ValueError(f"未知的距离误差模型 {model!r}")


def trilaterate_linear(ranges: np.ndarray, bs_xy: np.ndarray) -> np.ndarray:
    """线性最小二乘三边定位, 对应论文的 initial_est.m

    求解使 || |x - a_i|^2 - r_i^2 || 最小的点 x, 先线性化成 A x = R 的形式;
    用前三个基站定义 A(与 MATLAB 代码一致), 基站数大于 3 时其余站都参与贡献
    退化时回退到只在前 3 个基站上做纯线性求解, 与官方实现一致
    """
    a1, b1 = bs_xy[0]  # 第一个基站坐标
    a2, b2 = bs_xy[1]  # 第二个
    a3, b3 = bs_xy[2]  # 第三个
    r1, r2, r3 = ranges[0], ranges[1], ranges[2]  # 对应的三个距离
    A = np.array([[a1 - a2, b1 - b2], [a1 - a3, b1 - b3]], dtype=float)  # 两两相减消掉二次项
    R = np.array(
        [
            r2 ** 2 - r1 ** 2 + a1 ** 2 + b1 ** 2 - a2 ** 2 - b2 ** 2,
            r3 ** 2 - r1 ** 2 + a1 ** 2 + b1 ** 2 - a3 ** 2 - b3 ** 2,
        ],
        dtype=float,
    ) / 2.0  # 右端项, 二次项全搬到右边
    try:
        return np.linalg.solve(A, R)  # 一般情况直接解
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, R, rcond=None)[0]  # 退化时用最小二乘兜底


__all__ = [
    "Geometry",
    "random_geometry",
    "select_bs_subset",
    "has_range_collision",
    "add_laplace_or_uniform_errors",
    "trilaterate_linear",
]
