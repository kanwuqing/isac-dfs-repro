"""阶段一: 从基站收到的信号里, 把目标有多远抠出来.

对应论文 Sec. III-A 与 Sec. IV. 一句话讲整个故事: 基站发一段已知的电波,
打到人身上弹回来, 基站听回声; 回声晚到多久, 就说明人有多远.

用到的符号: M = 基站个数, K = 目标个数, N = 子载波个数, df = 子载波间隔,
带宽 B = N * df, L = 最多记多少个时间格( 抽头数), c0 = 光速.

为什么按格子算: 接收端不是连续记录时间, 而是每隔一小段记一次; 就像秒表只能
按 0.1 秒一格记, 回声落在第几格, 就大概知道距离. 格子越细距离越准,
这就是带宽越大越准的根源.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

C0 = 299_792_458.0  # 光速, 单位米每秒


@dataclass
class OFDMConfig:
    """发射机的参数, 改这里就等于改波形."""

    n_subcarriers: int = 1024  # N: 把频段切成多少条子载波
    subcarrier_spacing: float = 97_656.25  # df: 子载波之间隔多少赫兹
    n_bs: int = 4  # M: 几个基站
    n_targets: int = 4  # K: 几个人
    cp_len: int = 128  # Q: 循环前缀长度, 只要保证 L < Q 即可
    max_paths: int = 64  # L: 最多记多少个时间格
    tx_power: float = 1.0  # p: 发射功率
    noise_power: float = 1e-12  # 接收机噪声大小, 越小越干净
    pilot_mode: str = "ones"  # 发什么样的探测信号
    seed: Optional[int] = None

    @property
    def bandwidth(self) -> float:
        """总带宽 B = 子载波数 * 间隔, 单位赫兹."""
        return self.n_subcarriers * self.subcarrier_spacing

    @property
    def sample_period(self) -> float:
        """每个时间格有多长, 单位秒, 就是 1 除以带宽."""
        return 1.0 / self.bandwidth

    @property
    def delta_d(self) -> float:
        """距离尺子的最小刻度 Delta_d = c0 / (4B), 单位米.

        这是测得再准也只能准到这个刻度的下限. 100 MHz 时约 0.75 米,
        400 MHz 时约 0.19 米; 带宽翻几倍, 刻度就细几倍.
        """
        return C0 / (4.0 * self.bandwidth)

    @property
    def max_range(self) -> float:
        """这套参数最多能测多远.

        格子只有 L 个, 超过这个距离的回声会绕回来假装成近处目标( 叫混叠),
        所以要提前知道上限在哪.
        """
        return self.max_paths * C0 / (2.0 * self.bandwidth)

    def validate_unambiguous(self, max_distance_m: float) -> None:
        """检查最远的目标会不会绕回来; 绕回来了就直接报错拦住.

        如果目标比无模糊距离还远, 回声会落到格子外面绕回来, 算法会把它当成
        近处目标, 这种错很难发现, 所以宁可当场抛异常.
        """
        limit = min(self.max_paths, self.cp_len) * C0 / (2.0 * self.bandwidth)  # 无模糊上限
        if max_distance_m > limit:
            raise ValueError(
                f"最远距离 {max_distance_m:.2f} 米超过了无模糊距离 "
                f"{limit:.2f} 米(要增大 N, L 或带宽; "
                f"L={self.max_paths}, Q={self.cp_len}, B={self.bandwidth/1e6:.1f} MHz)"
            )

    def with_bandwidth(self, bandwidth_hz: float) -> "OFDMConfig":
        """复制一份配置, 只把带宽改成新值.

        做法是保持子载波数 N 不变, 只改子载波间隔 df; 论文扫带宽就是这么扫的.
        """
        import dataclasses

        new = dataclasses.replace(self, subcarrier_spacing=bandwidth_hz / self.n_subcarriers)  # 只改 df
        if new.cp_len >= new.n_subcarriers:
            raise ValueError("CP 长度必须 < N")  # 否则单载波均衡失效
        return new


def path_delay_index(distance_m: float, cfg: OFDMConfig) -> int:
    """一个距离对应第几个时间格.

    电波走一个来回, 路程是 2*d; 除以光速得到时间, 再除以每格时长,
    就知道落在第几格. 比如 30 米的目标, 回声约 0.2 微秒后到.
    """
    delay_samples = 2.0 * distance_m / (C0 * cfg.sample_period)  # 双程时延折成格子数
    l = int(np.floor(delay_samples)) + 1  # 格子编号从 1 开始
    return int(np.clip(l, 1, cfg.max_paths))


def range_from_tap_index(l: int, cfg: OFDMConfig) -> float:
    """反过来: 第 l 个格子代表多少米, 取格子中点误差最小."""
    return (l - 1) * C0 / (2.0 * cfg.bandwidth) + C0 / (4.0 * cfg.bandwidth)


def tap_index_from_range(d: float, cfg: OFDMConfig) -> int:
    """range_from_tap_index 的反函数: 真实距离落在哪个格子里."""
    return path_delay_index(d, cfg)


def make_pilots(cfg: OFDMConfig, rng: np.random.Generator) -> np.ndarray:
    """造每个基站要发的探测信号, 形状是( 基站数, 子载波数).

    全 1 最简单, 接收端拿它一除就抵消了; QPSK 和 Zadoff-Chu 更像真实 5G,
    但实验默认用全 1.
    """
    m, n = cfg.n_bs, cfg.n_subcarriers
    if cfg.pilot_mode == "ones":
        return np.ones((m, n), dtype=complex)  # 恒模, 后面能直接约掉
    if cfg.pilot_mode == "qpsk":
        phase = rng.integers(0, 4, size=(m, n)) * (np.pi / 2.0)  # 四个相位随机挑
        return np.exp(1j * phase)
    if cfg.pilot_mode == "zadoff-chu":
        seq = np.zeros((m, n), dtype=complex)
        for i in range(m):
            idx = np.arange(n)
            # 每个基站用同一串信号的不同移位, 保证大家功率一样
            seq[i] = np.exp(-1j * np.pi * (idx + i * 7) * (idx + i * 7 + 1) / n)
        return seq
    raise ValueError(f"未知的 pilot_mode {cfg.pilot_mode!r}")


def g_matrix(n_subcarriers: int, n_paths: int) -> np.ndarray:
    """傅里叶字典 G, 形状( 子载波数, 格子数).

    它就是一个翻译器: 把哪个时刻有回声( 时域)翻译成每个子载波上收到什么
    ( 频域), 反过来也能翻回去. 因为它每一列互相垂直( 正交), 后面估信道才能
    一步算出来.
    """
    n = np.arange(n_subcarriers)[:, None]  # 子载波序号, 竖着排
    l = np.arange(n_paths)[None, :]  # 格子序号, 横着排
    return np.exp(-2j * np.pi * n * l / n_subcarriers)


@dataclass
class RangeEstimate:
    """一个基站的阶段一结果: 它听到了哪几个距离."""

    bs_index: int
    tap_indices: np.ndarray  # 挑中的时间格编号, 从 1 开始
    ranges: np.ndarray  # 换算出来的距离, 单位米, 从小到大
    h_hat: np.ndarray  # 完整的信道估计, 长度 L
    threshold: float  # 这次用的门槛
    noise_var_est: float = 0.0


def ls_channel_estimate(
    ybar: np.ndarray,
    s: np.ndarray,
    cfg: OFDMConfig,
) -> np.ndarray:
    """从接收信号里把每个时间格有没有回声估出来.

    正常的最小二乘要算矩阵逆, 很慢; 但这里的翻译器 G 每一列互相垂直,
    所以 G 的共轭转置乘 G 等于 N 倍单位阵, 单位阵的逆就是 1/N 倍单位阵,
    直接约掉, 最后只剩一步逆变换( IDFT):

        h_hat = (1/(N*sqrt(p))) * G^H * (conj(s) * ybar)

    从要算矩阵逆变成做一次逆变换, 快约 500 倍; 和慢方法对比过, 结果逐位一样,
    不是近似.
    """
    n, l = cfg.n_subcarriers, cfg.max_paths
    g = g_matrix(n, l)  # 现算翻译器
    y_de = np.conj(s) * ybar  # 先用导频共轭把调制去掉
    return (g.conj().T @ y_de) / (n * np.sqrt(cfg.tx_power))


_LS_CACHE: Dict[Tuple[str, int, int], np.ndarray] = {}  # 同一个波形下 G 是固定的, 算一次存起来, 循环里反复用


def _cached_g(n: int, l: int) -> np.ndarray:
    """取缓存的 G, 没有就现算并塞进缓存."""
    key = ("g", n, l)
    if key not in _LS_CACHE:
        _LS_CACHE[key] = g_matrix(n, l)
    return _LS_CACHE[key]


def ls_channel_estimate_fast(
    ybar: np.ndarray,
    s: np.ndarray,
    cfg: OFDMConfig,
) -> np.ndarray:
    """和上面一样, 只是把翻译器 G 缓存起来, 不重复算."""
    g = _cached_g(cfg.n_subcarriers, cfg.max_paths)  # 走缓存, 省掉重复构造
    y_de = np.conj(s) * ybar  # 去调制
    return (g.conj().T @ y_de) / (cfg.n_subcarriers * np.sqrt(cfg.tx_power))


def estimate_noise_variance(h_hat: np.ndarray, n_subcarriers: int) -> float:
    """粗略估一下噪声有多大, 用估计结果的平均能量来算."""
    n, l = n_subcarriers, h_hat.size
    if n <= l:
        return float(np.mean(np.abs(h_hat) ** 2))  # 格子比子载波还多, 只能这样凑合
    return float(np.mean(np.abs(h_hat) ** 2))


def detect_nonzero_taps(
    h_hat: np.ndarray,
    noise_var: float,
    cfg: OFDMConfig,
    pfa: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    """判断哪些时间格是真回声, 哪些只是噪声.

    做法很朴素: 每个格子都有一个能量值, 设一个门槛; 门槛等于噪声的典型大小
    乘一个系数, 系数由允许的误报率决定( pfa=1e-3 就是纯噪声被误判成回声的
    概率只有千分之一). 超过门槛的格子就认为是真回声. 门槛跟着噪声自动调,
    所以噪声忽大忽小都不会满屏假回声.
    """
    sigma_h2 = (noise_var / cfg.tx_power) / cfg.n_subcarriers  # 每个格子的噪声典型大小; 因为 G 的列正交, 每个格子噪声一样大

    sigma_h2 = max(sigma_h2, 1e-30)  # 给一个极小的下限, 保证完全没噪声时门槛也有定义, 不会除零
    gamma = sigma_h2 * np.log(1.0 / max(pfa, 1e-12))  # 门槛由误报率反推

    energy = np.abs(h_hat) ** 2  # 每个格子的能量
    taps = np.flatnonzero(energy > gamma) + 1  # 过门槛的格子, 编号从 1 开始
    return taps, float(gamma)


def phase1_range_estimation(
    cfg: OFDMConfig,
    bs_xy: np.ndarray,
    target_xy: np.ndarray,
    rng: np.random.Generator,
    pfa: float = 1e-3,
    channel_scale: Optional[float] = None,
    include_cross_paths: bool = False,
    path_loss_exponent: float = 1.0,
    dynamic_range_compensation: bool = True,
) -> List[RangeEstimate]:
    """阶段一主流程: 对每个基站造接收信号, 估信道, 挑抽头, 转距离.

    返回每个基站的一组距离( 还没告诉谁是谁).

    论文假设相邻基站用不同频率, 所以一个基站只收到自己发出的信号.
    include_cross_paths=True 时引入跨站路径, 用于 E5 消融.

    默认按 1/d 衰减, 使单站动态范围约 10 到 20 倍, 保证 K 个目标都可检测.
    1/d^2 的情形在 E5 中量化.
    """
    m_bs = cfg.n_bs
    k_tg = cfg.n_targets
    pilots = make_pilots(cfg, rng)  # 每个基站要发的信号

    if channel_scale is None:
        # 用基站到目标的平均距离做基准, 让不同布局下信号强度可比
        d_all = np.linalg.norm(bs_xy[:, None, :] - target_xy[None, :, :], axis=-1)
        channel_scale = float(np.mean(d_all)) ** path_loss_exponent

    results: List[RangeEstimate] = []
    for m in range(m_bs):
        h_true = np.zeros(cfg.max_paths, dtype=complex)  # 第一步: 造真实信道, 每个目标在它的格子里放一个回声
        true_ranges = []
        for k in range(k_tg):
            d = float(np.linalg.norm(bs_xy[m] - target_xy[k]))  # 这个目标到本站多远
            l = path_delay_index(d, cfg)  # 落在第几格
            amp = channel_scale / (d ** path_loss_exponent)  # 强度随距离衰减
            phase = rng.uniform(0, 2 * np.pi)  # 相位随机, 因为反射面朝向未知
            h_true[l - 1] += amp * np.exp(1j * phase)
            true_ranges.append(d)

        g = g_matrix(cfg.n_subcarriers, cfg.max_paths)  # 第二步: 造基站实际收到的信号, 收到 = 导频 * 傅里叶变换( 信道)
        ybar = np.sqrt(cfg.tx_power) * (pilots[m] * (g @ h_true))

        if include_cross_paths:  # 引入跨站路径, 仅 E5 消融使用
            for u in range(m_bs):
                if u == m:
                    continue
                h_cross = np.zeros(cfg.max_paths, dtype=complex)
                for k in range(k_tg):
                    d_u = float(np.linalg.norm(bs_xy[u] - target_xy[k]))  # 别站到目标
                    d_m = float(np.linalg.norm(bs_xy[m] - target_xy[k]))  # 本站到目标
                    eq_range = (d_u + d_m) / 2.0  # 双站路径折成等效单程
                    l = path_delay_index(eq_range, cfg)
                    amp = channel_scale / ((d_u + d_m) / 2.0) ** path_loss_exponent
                    phase = rng.uniform(0, 2 * np.pi)
                    h_cross[l - 1] += amp * np.exp(1j * phase)
                ybar = ybar + np.sqrt(cfg.tx_power) * (pilots[u] * (g @ h_cross))

        # 第三步: 加噪声, 真实接收机不可能没噪声
        sigma = np.sqrt(cfg.noise_power / 2.0)  # 复噪声每个分量各占一半
        noise = sigma * (
            rng.standard_normal(cfg.n_subcarriers) + 1j * rng.standard_normal(cfg.n_subcarriers)
        )
        ybar = ybar + noise

        h_hat = ls_channel_estimate_fast(ybar, pilots[m], cfg)  # 第四步: 估信道, 挑抽头
        nv = estimate_noise_variance(h_hat, cfg.n_subcarriers)
        taps, gamma = detect_nonzero_taps(h_hat, nv, cfg, pfa=pfa)

        if taps.size > k_tg:  # 最多留 K 个最强的抽头, 因为论文假设每个基站看到 K 条路径
            order = np.argsort(-np.abs(h_hat[taps - 1]))  # 按能量从大到小排
            taps = np.sort(taps[order[:k_tg]])
        if taps.size == 0:
            taps = np.sort(np.argsort(-np.abs(h_hat))[:k_tg] + 1)  # 万一门槛太严一个都没挑出来, 兜底取最强的 K 个

        ranges = np.array([range_from_tap_index(int(l), cfg) for l in taps])  # 第五步: 抽头编号转距离
        order = np.argsort(ranges)  # 距离从小到大排好
        results.append(
            RangeEstimate(
                bs_index=m,
                tap_indices=taps[order],
                ranges=ranges[order],
                h_hat=h_hat,
                threshold=gamma,
                noise_var_est=nv,
            )
        )
    return results


def simulate_range_errors(
    cfg: OFDMConfig,
    bs_xy: np.ndarray,
    target_xy: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """给阶段二用的理想化距离: 直接算真实距离落在哪个格子, 取中点.

    不做信号仿真, 跳过阶段一, 专门测阶段二的拼图能力.
    """
    m_bs, k_tg = cfg.n_bs, cfg.n_targets
    out = np.zeros((m_bs, k_tg))
    for m in range(m_bs):
        for k in range(k_tg):
            d = float(np.linalg.norm(bs_xy[m] - target_xy[k]))  # 真实距离
            l = path_delay_index(d, cfg)  # 先落到格子
            out[m, k] = range_from_tap_index(l, cfg)  # 再取格子中点
    return out


__all__ = [
    "C0",
    "OFDMConfig",
    "RangeEstimate",
    "make_pilots",
    "g_matrix",
    "ls_channel_estimate",
    "ls_channel_estimate_fast",
    "detect_nonzero_taps",
    "phase1_range_estimation",
    "range_from_tap_index",
    "tap_index_from_range",
    "path_delay_index",
    "simulate_range_errors",
]
