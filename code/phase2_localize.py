"""阶段二: 无设备感知的定位与数据关联.

复现论文 Sec. V 与 Sec. VI. 实现了两种情形:

1. 完美测距( Sec. V). 每个基站 m 提供一个集合 D_m, 不带目标标签; 如果某个候选
   关联算出的 M 个圆不能交于同一点, 这个候选就是鬼影目标.
2. 不完美测距( Sec. VI). 论文的最大似然算法: 枚举前 3 个基站的可行三元组,
   用匈牙利算法向其余基站扩展, 最后按残差总和取最小.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import trilaterate_linear


def refine_position(
    ranges: np.ndarray,
    bs_xy: np.ndarray,
    x0: Optional[np.ndarray] = None,
    lam: float = 0.2,
    max_iter: int = 50,
    tol: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """用阻尼高斯-牛顿法精修目标位置.

    对应官方 coorest.m: 从线性三边定位结果出发反复迭代, 直到步长范数小于 tol.
    J^T J 只有 2x2, 所以正规方程用显式闭式求逆来解( 几何接近降秩时加阻尼);
    对 2x2 系统这与伪逆等价, 但快好几倍, 而最大似然算法会调用本函数很多次.

    返回位置和残差向量, 残差元素为 | |x-a_m| - d_m |.
    """
    ranges = np.asarray(ranges, dtype=float).ravel()
    if x0 is None:
        x0 = trilaterate_linear(ranges, bs_xy)  # 没有初值就先线性解一把
    beta = np.asarray(x0, dtype=float).ravel()[:2].copy()  # 只要二维坐标
    l_d = ranges.size

    if not np.all(np.isfinite(beta)):
        beta = np.zeros(2)  # 初值坏掉就从头来过

    bs = bs_xy[:l_d]
    for _ in range(max_iter):
        diff = beta[None, :] - bs  # (L, 2) 当前点到各基站的方向
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))  # 当前距离
        dist = np.maximum(dist, 1e-12)  # 防止和基站重合时除零
        jac = diff / dist[:, None]  # (L, 2) 雅可比, 就是单位方向
        resid = dist - ranges  # (L,) 当前残差

        jtj = jac.T @ jac  # 2x2 正规方程左端
        a, b, d = jtj[0, 0], jtj[0, 1], jtj[1, 1]
        det = a * d - b * b  # 行列式
        if abs(det) < 1e-12:
            a += 1e-6  # 几何退化成一条线时加点阻尼, 让它还能算下去
            d += 1e-6
            det = a * d - b * b
            if abs(det) < 1e-18:
                break
        grad = jac.T @ resid  # (2,) 梯度
        step0 = (d * grad[0] - b * grad[1]) / det  # 2x2 求逆的闭式写法, 比调伪逆快
        step1 = (a * grad[1] - b * grad[0]) / det

        beta_next = beta - lam * np.array([step0, step1])  # 阻尼步长
        if not np.all(np.isfinite(beta_next)):
            break
        if np.linalg.norm(beta_next - beta) < tol:
            beta = beta_next  # 步长够小, 收工
            break
        beta = beta_next

    diff = beta[None, :] - bs
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    residuals = np.abs(dist - ranges)  # 最终残差
    return beta, residuals


def position_cost(
    ranges: np.ndarray, bs_xy: np.ndarray, x0: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """目标函数值 sum_m | |x - a_m| - d_m | 以及对应位置."""
    x, resid = refine_position(ranges, bs_xy, x0=x0)
    return float(np.sum(resid)), x


def association_consistent(
    ranges: np.ndarray,
    bs_xy: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """这 M 个圆是否交于同一点, 见论文 Sec. V-A.

    一个候选关联是一致的, 当且仅当存在一点 x 使所有 m 都满足 |x - a_m| = d_m;
    这里用最小二乘解出位置再检查残差来判断.
    """
    if len(ranges) < 3:
        return True  # 少于三个圆谈不上交不交
    x, resid = refine_position(ranges, bs_xy, tol=tol)
    return bool(np.max(resid) <= max(tol, 1e-9))


def ghost_target_exists(
    bs_ranges: Sequence[np.ndarray],
    bs_xy: np.ndarray,
    k_targets: int,
    tol: float = 1e-6,
) -> Tuple[bool, Optional[np.ndarray]]:
    """搜索除真实关联之外是否还存在一致的关联.

    在完美测距下鬼影目标存在, 当且仅当某个关联( 每个基站取一个距离, 且都来自
    不同目标)也一致, 但对应到另一个物理点. 返回是否存在和一个示例关联.
    """
    m_bs = len(bs_ranges)
    # 对每个基站, 枚举它 K 个距离的所有排列
    perms_per_bs = [list(permutations(range(k_targets))) for _ in range(m_bs)]
    for combo in product(*perms_per_bs[1:]):  # 第 0 个基站固定按自然顺序取, 因为目标无标签, 重命名是免费的
        assoc = np.zeros((m_bs, k_targets), dtype=int)
        for k in range(k_targets):
            assoc[0, k] = k  # 第 0 站的对应关系钉死
        for m in range(1, m_bs):
            for k in range(k_targets):
                assoc[m, k] = combo[m - 1][k]  # 其余站按当前排列取
        for k in range(k_targets):
            cand = np.array([bs_ranges[m][assoc[m, k]] for m in range(m_bs)])
            if association_consistent(cand, bs_xy, tol=tol):
                x, _ = refine_position(cand, bs_xy)
                if not _matches_known_target(x, bs_ranges, bs_xy, tol=1e-3):  # 这个点是不是某个别的目标的已知位置?
                    return True, cand
    return False, None


def _matches_known_target(
    x: np.ndarray,
    bs_ranges: Sequence[np.ndarray],
    bs_xy: np.ndarray,
    tol: float = 1e-3,
) -> bool:
    """判断 x 是不是 K 个真实目标位置之一.

    真实目标位置可以从一致的对角关联重建出来; 因为测距是完美的, 真目标恰好就是
    那些一致的关联, 所以按位置比对就能认出已经见过的点.
    """
    k = len(bs_ranges[0])
    for kk in range(k):
        cand = np.array([bs_ranges[m][kk] for m in range(len(bs_ranges))])
        if association_consistent(cand, bs_xy, tol=1e-6):
            xk, _ = refine_position(cand, bs_xy)
            if np.linalg.norm(xk - x) < tol:
                return True  # 就是它, 不算新鬼影
    return False


@dataclass
class AssociationResult:
    """最大似然数据关联算法的返回结果."""

    positions: np.ndarray  # (K, 2) 估计出来的目标位置
    assignment: np.ndarray  # (M, K) 每个基站给每个目标配的距离下标
    objective: float  # 最终的残差总和
    feasible: bool  # 有没有候选通过筛子
    n_candidates: int = 0  # 打过分的完整候选数


def triangle_feasible(
    cand: np.ndarray,
    bs_xy: np.ndarray,
    delta: float,
) -> bool:
    """带容差 delta 的三角不等式检验, 见论文 Sec. VI-A.

    对第 i, j 两个基站来说, 两个距离之差不能超过站间距离加容差, 之和也不能
    小于站间距离减容差, 否则这三个距离根本围不成三角形.
    """
    for i in range(3):
        for j in range(i + 1, 3):
            dij = float(np.linalg.norm(bs_xy[i] - bs_xy[j]))  # 两站之间的距离
            si, sj = cand[i], cand[j]  # 这两个站各取的距离
            if abs(si - sj) > dij + delta:
                return False  # 差太多, 三角形闭不上
            if abs(si + sj) < dij - delta:
                return False  # 和太小, 也闭不上
    return True


def ml_data_association(
    bs_ranges: Sequence[np.ndarray],
    bs_xy: np.ndarray,
    sigma_d: float,
    region_size: Optional[float] = None,
    feasibility_factor: float = 1.5,
    delta_factor: float = 3.0,
    max_error: Optional[float] = None,
) -> AssociationResult:
    """最大似然联合数据关联与定位, 见论文 Sec. VI-A 与 model.m.

    bs_ranges 是各基站的距离集合( 不带标签), sigma_d 是测距误差的标准差,
    论文的量化模型里误差上界是 Delta_d, 所以 sigma_d 取 Delta_d / 2.
    feasibility_factor 决定残差预算, delta_factor 决定三角不等式容差,
    max_error 给了就直接当绝对预算用.
    """
    m_bs = len(bs_ranges)
    k_tg = len(bs_ranges[0])
    delta = delta_factor * sigma_d  # 三角不等式容差

    if max_error is None:
        # 量化误差上界为 Delta_d, 预算取 M * Delta_d 与误差量级匹配.
        budget = m_bs * 2.0 * sigma_d * feasibility_factor
    else:
        budget = float(max_error)

    d0, d1, d2 = bs_ranges[0], bs_ranges[1], bs_ranges[2]  # 第一步: 在前 3 个基站上枚举可行三元组
    triples: List[np.ndarray] = []
    for i, j, k in product(range(k_tg), repeat=3):
        cand = np.array([d0[i], d1[j], d2[k]])  # 一个候选三元组
        if not triangle_feasible(cand, bs_xy, delta):
            continue  # 围不成三角形就不要
        _, resid = refine_position(cand, bs_xy)
        if np.sum(resid) > budget:
            continue  # 残差太大也不要
        triples.append(cand)

    if not triples:
        # 一个可行三元组都没有, 只能报失败, 这会被算进定位误差
        return AssociationResult(
            positions=np.zeros((k_tg, 2)),
            assignment=np.zeros((m_bs, k_tg), dtype=int),
            objective=np.inf,
            feasible=False,
            n_candidates=0,
        )

    # 第二步: 枚举前 3 个基站上一致的部分匹配
    # 每个槽位选一个三元组, 且每个基站上用掉的 K 个距离必须刚好等于它的集合
    target_sorted = np.sort(np.stack(bs_ranges[:3], axis=0), axis=1)  # 三站各自的排序结果

    triples_arr = np.asarray(triples, dtype=float)  # (T, 3) 所有候选堆一块
    d1_sorted = target_sorted[0]
    groups: List[np.ndarray] = []
    for t in range(k_tg):
        # 目标无标签, 约定第 t 个槽位用 D_1 里第 t 小的距离, 这是无损的重编号
        sel = np.flatnonzero(np.isclose(triples_arr[:, 0], d1_sorted[t]))
        groups.append(triples_arr[sel])

    partials: List[np.ndarray] = []

    def _extend(slot: int, chosen: List[np.ndarray]) -> None:
        """逐槽位往里填三元组, 填满了检查是否正好用完各站的距离."""
        if slot == k_tg:
            cand = np.stack(chosen, axis=1)  # (3, K) 拼成一张表
            # 每个基站的 K 个距离必须各用一次, 所以排序后要逐元素对得上
            if np.array_equal(np.sort(cand, axis=1), target_sorted):
                partials.append(cand)
            return
        for tr in groups[slot]:
            ok = True  # 剪枝: 当前用掉的距离必须仍是各站集合的子多重集
            for m in range(3):
                used = [c[m] for c in chosen] + [tr[m]]  # 目前已用
                avail = list(target_sorted[m])  # 本站还有哪些能用
                for u in used:
                    hit = None
                    for ai, av in enumerate(avail):
                        if np.isclose(u, av):
                            hit = ai
                            break
                    if hit is None:
                        ok = False  # 用超了, 这个分支废了
                        break
                    avail.pop(hit)
                if not ok:
                    break
            if ok:
                chosen.append(tr)
                _extend(slot + 1, chosen)
                chosen.pop()

    _extend(0, [])

    if not partials:
        return AssociationResult(
            positions=np.zeros((k_tg, 2)),
            assignment=np.zeros((m_bs, k_tg), dtype=int),
            objective=np.inf,
            feasible=False,
            n_candidates=0,
        )

    # 第三步: 用匈牙利算法把每个部分匹配扩展到第 4 到第 M 个基站
    # 固定部分匹配时各目标互不影响, 所以每个目标维护一份运行状态,
    scored: List[Tuple[float, np.ndarray, np.ndarray]] = []  # 只在它的距离集合变大时才重新精修, 省掉大量重复拟合
    for part in partials:
        assoc = np.zeros((m_bs, k_tg), dtype=float)
        assoc[:3, :] = part  # 前三个基站先填进去

        pos = np.zeros((k_tg, 2))
        for t in range(k_tg):
            pos[t], _ = refine_position(part[:, t], bs_xy[:3])  # 只用三站先定个位

        for m in range(3, m_bs):
            # 代价表: 第 t 个目标用第 r 个距离差多少
            d_to_bs = np.linalg.norm(pos - bs_xy[m][None, :], axis=1)  # (K,)
            cost = np.abs(d_to_bs[:, None] - bs_ranges[m][None, :])  # (K, K)
            row, col = linear_sum_assignment(cost)  # 匈牙利算法求最优指派
            matched = np.zeros(k_tg)
            matched[row] = bs_ranges[m][col]
            assoc[m, :] = matched
            for t in range(k_tg):
                pos[t], _ = refine_position(assoc[: m + 1, t], bs_xy[: m + 1])  # 目标多了新距离, 只重新精修它自己

        total = 0.0
        for t in range(k_tg):
            x, resid = refine_position(assoc[:, t], bs_xy)  # 全部 M 站一起精修
            pos[t] = x
            total += float(np.sum(resid))
        scored.append((total, pos.copy(), assoc.copy()))

    scored.sort(key=lambda z: z[0])  # 按残差总和从小到大排
    best_cost, best_pos, best_assoc = scored[0]  # 最小的那个就是最大似然解

    return AssociationResult(
        positions=best_pos,
        assignment=_to_index_assignment(best_assoc, bs_ranges),
        objective=best_cost,
        feasible=True,
        n_candidates=len(scored),
    )


def _to_index_assignment(
    matched_ranges: np.ndarray, bs_ranges: Sequence[np.ndarray]
) -> np.ndarray:
    """把匹配到的距离数值还原成它在各自集合里的下标."""
    m_bs, k_tg = matched_ranges.shape
    idx = np.zeros((m_bs, k_tg), dtype=int)
    for m in range(m_bs):
        for t in range(k_tg):
            # 数值上相等就算命中, 容差给到 1e-9
            hits = np.flatnonzero(np.isclose(bs_ranges[m], matched_ranges[m, t], atol=1e-9))
            idx[m, t] = hits[0] if hits.size else -1
    return idx


def pairwise_position_errors(est: np.ndarray, true: np.ndarray) -> np.ndarray:
    """用最优( 匈牙利)配对把估计位置与真实位置对上, 返回误差.

    目标本身无标签, 所以按最小代价指派来打分, 与官方 diffdist.m 的做法一致.
    """
    k = true.shape[0]
    cost = np.linalg.norm(est[:, None, :] - true[None, :, :], axis=-1)  # 两两距离表
    row, col = linear_sum_assignment(cost)
    return cost[row, col]  # 只取配对上的那几个


def error_probability(errors: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """定位误差超过各阈值的目标所占比例."""
    return np.array([np.mean(errors > t) for t in thresholds], dtype=float)


__all__ = [
    "refine_position",
    "position_cost",
    "association_consistent",
    "ghost_target_exists",
    "AssociationResult",
    "ml_data_association",
    "triangle_feasible",
    "pairwise_position_errors",
    "error_probability",
]
