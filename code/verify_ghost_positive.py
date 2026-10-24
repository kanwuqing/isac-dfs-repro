"""正控制: 在真的存在鬼影时, 检测器能否真的找到它?

论文说鬼影很少见, 所以一个永远返回 False 的检测器放到随机几何上看起来
也"对". 为了说明检测器不是摆设, 这里手工造几个一定存在鬼影的布局,
看暴力法和快速法是不是都能报出来.
"""
from __future__ import annotations
import sys
from pathlib import Path
from itertools import permutations, product
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from isac_dfs.phase2_localize import association_consistent, refine_position
from run_experiments import _has_ghost_efficient


def brute_ghost(clean, bs_xy, tol=1e-6):
    m_bs, k = clean.shape
    genuine = []
    for kk in range(k):
        if association_consistent(clean[:, kk], bs_xy, tol=1e-6):
            x, _ = refine_position(clean[:, kk], bs_xy)
            genuine.append(x)
    genuine = np.array(genuine) if genuine else np.zeros((0, 2))

    def is_gen(x):
        if genuine.shape[0] == 0:
            return False
        return bool(np.min(np.linalg.norm(genuine - x[None, :], axis=1)) < 1e-3)

    for combo in product(list(permutations(range(k))), repeat=m_bs - 1):
        cand = np.zeros((m_bs, k)); cand[0, :] = clean[0, :]
        for mi in range(1, m_bs):
            for t in range(k):
                cand[mi, t] = clean[mi, combo[mi - 1][t]]
        for t in range(k):
            if association_consistent(cand[:, t], bs_xy, tol=tol):
                x, _ = refine_position(cand[:, t], bs_xy)
                if not is_gen(x):
                    return True, x
    return False, None


print("=" * 72)
print("正控制 1：对称的四基站方形布局，两个目标摆成让交叉关联也闭合")
print("（这是经典的鬼影目标构造方式）")
print("=" * 72)
bs = np.array([[0.0, 0.0], [200.0, 0.0], [0.0, 200.0], [200.0, 200.0]])
# 两个目标关于方形中心镜像摆放: 这时候交换着配一遍也能闭合,
# 于是生成镜像点, 这就是一个真的鬼影.
t1 = np.array([60.0, 70.0])
t2 = np.array([140.0, 130.0])
targets = np.array([t1, t2])
# 精确构造距离集合: 第 m 个基站看到的是它到 t1 和 t2 的距离.
clean = np.zeros((4, 2))
for m in range(4):
    r = np.linalg.norm(bs[m] - targets, axis=1)
    clean[m] = np.sort(r)
print("基站坐标：\n", bs)
print("目标坐标：\n", targets)
print("各基站距离集合（M x K）：\n", np.round(clean, 6))

b, bx = brute_ghost(clean, bs)
f = _has_ghost_efficient(clean, bs)
print(f"暴力判定={b}  （交点 {None if bx is None else np.round(bx,4)}）")
print(f"快速判定={f}")
print("两者一致" if b == f else "*** 两者不一致 ***")

print()
print("=" * 72)
print("正控制 2：显式镜像构造（保证存在鬼影）")
print("=" * 72)
# 两个目标关于原点对称摆, 三个基站摆在以原点为中心的圆周上.
# 这样有 r_1(t) == r_2(-t), 所以在每个基站上把两个距离对调,
# 得到的是镜像的那一对, 它本身也是一个自洽解, 于是就是鬼影.
ang = np.array([0.0, 2*np.pi/3, 4*np.pi/3])
bs2 = 150.0 * np.column_stack([np.cos(ang), np.sin(ang)])
tA = np.array([40.0, 25.0])
tB = -tA
tg2 = np.array([tA, tB])
clean2 = np.zeros((3, 2))
for m in range(3):
    clean2[m] = np.sort(np.linalg.norm(bs2[m] - tg2, axis=1))
print("基站坐标：\n", np.round(bs2, 3))
print("目标坐标：\n", tg2)
print("各基站距离集合：\n", np.round(clean2, 6))
b2, bx2 = brute_ghost(clean2, bs2)
f2 = _has_ghost_efficient(clean2, bs2)
print(f"暴力判定={b2}  （交点 {None if bx2 is None else np.round(bx2,4)}）")
print(f"快速判定={f2}")
print("两者一致" if b2 == f2 else "*** 两者不一致 ***")

print()
print("=" * 72)
print("负控制：普通非对称几何不应出现鬼影")
print("=" * 72)
rng = np.random.default_rng(9)
n_agree = 0
for _ in range(30):
    bsx = rng.uniform(-120, 120, (4, 2))
    tx = rng.uniform(-60, 60, (3, 2))
    cl = np.zeros((4, 3))
    for m in range(4):
        cl[m] = np.sort(np.linalg.norm(bsx[m] - tx, axis=1))
    if brute_ghost(cl, bsx)[0] == _has_ghost_efficient(cl, bsx):
        n_agree += 1
print(f"30 个随机普通几何上的一致数：{n_agree}/30")
