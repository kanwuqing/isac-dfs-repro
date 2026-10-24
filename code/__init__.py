"""OFDM 蜂窝网络无设备感知 —— 复现代码包.

复现的论文:
    Q. Shi, L. Liu, S. Zhang, and S. Cui, "Device-Free Sensing in OFDM
    Cellular Network," IEEE Journal on Selected Areas in Communications,
    vol. 40, no. 6, pp. 1838-1853, Jun. 2022.  arXiv:2108.09177.

包含的模块:
phase1_range      阶段一, 从 OFDM 信号估到各基站的距离
phase2_localize   阶段二, 鬼影目标, 最大似然数据关联, 定位
geometry          两个阶段共用的几何工具
"""

from .phase1_range import OFDMConfig, phase1_range_estimation
from .phase2_localize import ml_data_association, pairwise_position_errors
from .geometry import Geometry, random_geometry

__version__ = "1.0.0"

__all__ = [
    "OFDMConfig",
    "phase1_range_estimation",
    "ml_data_association",
    "pairwise_position_errors",
    "Geometry",
    "random_geometry",
    "__version__",
]
