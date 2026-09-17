# 25 个锐角等腰三角形锁定组的构造与拆分搜索

本文尝试为当前 25 个半径

$$
\lambda=\frac1{\sqrt6+\sqrt3}
       =0.239146311738100\ldots
$$

的覆盖圆各构造一个三点组，使每组三点的唯一最小包围圆正好是对应的当前圆。随后不附加任何“组不可拆分”假设，对全部 75 点求真正的最小圆覆盖数。

实现、测试和数据文件为：

- [`src/triangle_groups.py`](../src/triangle_groups.py)；
- [`tests/test_triangle_groups.py`](../tests/test_triangle_groups.py)；
- [`data/triangle_trials_round1.json`](../data/triangle_trials_round1.json)；
- [`data/triangle_trials_strict_round2.json`](../data/triangle_trials_strict_round2.json)；
- [`data/triangle_strict_15_cover.json`](../data/triangle_strict_15_cover.json)。

## 1. 锐角等腰三角形锁定圆

考虑一个圆心为 $c$、半径为 $\lambda$ 的圆。选定一条参考轴，在圆周上取法向角

$$
-\beta,\qquad +\beta,\qquad \pi.
$$

三点为

$$
p_-=c+\lambda(\cos(-\beta),\sin(-\beta)),
$$

$$
p_+=c+\lambda(\cos\beta,\sin\beta),
$$

以及

$$
p_i=c+\lambda(-1,0).
$$

前两点关于参考轴对称，因此三角形是等腰三角形。三个相邻圆周间隔为

$$
\pi-\beta,\qquad \pi-\beta,\qquad 2\beta.
$$

当

$$
0<\beta<\frac\pi2
$$

时，三个间隔都严格小于 $\pi$，圆心严格位于三角形内部，所以三角形是锐角三角形。锐角三角形的最小包围圆唯一等于其外接圆，因此

$$
\operatorname{MEC}\{p_-,p_+,p_i\}=B(c,\lambda).
$$

这实现了“用三个点唯一确定一个需要的圆”。

## 2. 25 组点的坐标

令 $\phi_i$ 为当前圆心相对于原点的极角。对非中心圆，将参考轴取为径向向外方向，则三点统一写成

$$
p_{i,\theta}
=c_i+\lambda
 \bigl(\cos(\phi_i+\theta),\sin(\phi_i+\theta)\bigr),
\qquad
\theta\in\{-\beta_i,+\beta_i,\pi\}.
$$

对于中心圆 $O$，用一个自由相位 $\phi_O$ 代替径向角。

25 个圆心仍为

$$
O=(0,0),
$$

$$
A_k=\lambda\mathcal R_k
 \left(1+\frac{\sqrt2}{2},\frac{\sqrt2}{2}\right),
$$

$$
B_k=\lambda\mathcal R_k(2+\sqrt2,0),
$$

$$
C_k=\lambda\mathcal R_k(2+\sqrt2,\sqrt2),
\qquad k=0,\ldots,7.
$$

其中 $\mathcal R_k$ 表示旋转 $k\pi/4$。

## 3. 保证三点位于目标单位圆盘内

$O$ 圆和全部 $A_k$ 圆完整位于目标单位圆盘内，所以只需要

$$
0<\beta_O,\beta_A<\frac\pi2.
$$

对 $B_k$，小圆周上的点位于单位圆盘内当且仅当其相对径向法向角 $\theta$ 满足

$$
\cos\theta\le\cos\frac\pi4.
$$

因此三角形参数应满足

$$
\frac\pi4\le\beta_B<\frac\pi2.
$$

对 $C_k$，对应条件是

$$
\cos\theta\le\cos\frac{3\pi}{8},
$$

所以

$$
\frac{3\pi}{8}\le\beta_C<\frac\pi2.
$$

程序对每个参数实例重新检查：

1. 75 点全部位于闭单位圆盘；
2. 每组三角形严格为锐角；
3. 每组 MEC 半径与 $\lambda$ 的误差小于 $2\times10^{-10}$；
4. MEC 圆心与对应当前圆心的误差小于 $2\times10^{-10}$。

## 4. 第一轮：在半径 $\lambda$ 下搜索 100 组参数

每个 seed 随机选择

$$
\begin{aligned}
42^\circ&\le\beta_O\le87^\circ,\\
35^\circ&\le\beta_A\le87^\circ,\\
45.5^\circ&\le\beta_B\le88.5^\circ,\\
68^\circ&\le\beta_C\le88.5^\circ,\\
0^\circ&\le\phi_O\le45^\circ.
\end{aligned}
$$

对每个 75 点实例，使用两点 locking-center 完备枚举所有最大 $\lambda$-可覆盖子集，再用整数集合覆盖求最少圆数。

结果为：

| 最少圆数 | 参数实例数 |
|---:|---:|
| 12 | 60 |
| 13 | 40 |

最佳值只有 13。第一轮最佳 seed 为 1，其参数为

$$
\begin{aligned}
\beta_O&=65.0319731^\circ,\\
\beta_A&=84.4241122^\circ,\\
\beta_B&=51.6988633^\circ,\\
\beta_C&=87.4473137^\circ,\\
\phi_O&=14.0324153^\circ.
\end{aligned}
$$

该实例有 104 个最大可覆盖模式，集合覆盖最优值为 13。

## 5. 第二轮：半径严格小于 $\lambda$

三角锁定技巧在严格半径下会排除每个原始三点组：取

$$
\rho=0.999999\lambda
=0.2391460725917885\ldots<\lambda,
$$

则任何一个半径 $\rho$ 的圆都不能完整覆盖某个原始锐角三角形。

重新对全部 100 组参数求解，结果为：

| 最少圆数 | 参数实例数 |
|---:|---:|
| 14 | 81 |
| 15 | 19 |

严格半径下的最佳值仍只有 15。最佳 seed 为 94：

$$
\begin{aligned}
\beta_O&=52.7851916^\circ,\\
\beta_A&=76.0120383^\circ,\\
\beta_B&=76.5479157^\circ,\\
\beta_C&=68.9384489^\circ,\\
\phi_O&=14.8255618^\circ.
\end{aligned}
$$

## 6. 显式的 15 圆拆分反例

[`data/triangle_strict_15_cover.json`](../data/triangle_strict_15_cover.json) 保存了 seed 94 在半径 $0.999999\lambda$ 下的完整 15 圆覆盖。点标签中的 `-`、`+` 和 `i` 分别对应法向角 $-\beta,+\beta,\pi$。

| 子集 | 承担的三角顶点 | MEC 半径 |
|---:|---|---:|
| 1 | `O+ Oi A0i A1i A2i A3i A4i A5i A6i A7i` | 0.2284406350 |
| 2 | `O- A0- A6+ A7- A7+ B0i B7i C7i` | 0.2358477641 |
| 3 | `A1+ A2- A2+ A3- B2i B3i C2i` | 0.2320547949 |
| 4 | `A4+ A5- A5+ A6- B5i B6i C5i` | 0.2320547949 |
| 5 | `A0+ A1- B1- B1i C0+ C0i` | 0.2352731622 |
| 6 | `A3+ A4- B4+ B4i C4- C4i` | 0.2352731622 |
| 7 | `A6+ A7- B7- B7i C6+ C6i` | 0.2352731622 |
| 8 | `B0+ B1- C0- C0i` | 0.2023921884 |
| 9 | `B1+ B2- C1- C1i` | 0.2023921884 |
| 10 | `B2- B2+ C1+ C2-` | 0.2325852867 |
| 11 | `B3- B3i C2+ C3-` | 0.2368516593 |
| 12 | `B3+ B4- C3+ C3i` | 0.2023921884 |
| 13 | `B5- B5+ C4+ C5-` | 0.2325852867 |
| 14 | `B6- B6+ C5+ C6-` | 0.2325852867 |
| 15 | `B0- B7+ C7- C7+` | 0.2231701195 |

其中最大 MEC 半径为

$$
0.236851659268716
<0.239146072591789
=0.999999\lambda.
$$

所以这不是浮点边界造成的假反例，而是带有约 $0.00229$ 半径裕量的真实跨组三角顶点重组。

## 7. 与内外层证书合并

还测试了以下并集：

- 75 个最佳三角锁定点；
- 已证明需要 16 圆的外层 56 点；
- 独立需要 9 圆的内层 65 点。

去重后共 180 点。完整几何集合覆盖结果为：

| 覆盖半径 | 最少圆数 |
|---:|---:|
| $\lambda$ | 19 |
| $0.999999\lambda$ | 19 |

因此独立的“外层至少 16”和“内层至少 9”不能直接相加，三角点也没有阻止跨层圆同时承担两个证书中的点。

## 8. 结论

本次构造成功证明了以下局部事实：

> 对当前 25 个圆中的每一个，都存在一个位于单位圆盘内的三点组，其唯一最小包围圆正好是该当前圆，半径为 $\lambda$。

但它不推出全局覆盖必须使用这 25 个圆。缺失的命题是：

> 任意竞争覆盖中，每个三点组的三个顶点必须由同一个圆承担。

实际的 15 圆反例表明该命题不成立。竞争覆盖可以把每个三角形拆开，再把不同三角形的顶点组合到同一个更小圆中。

因此，“锐角等腰三角形唯一确定外接圆”是有效的**局部锁定器**，但不是独立的 **must-link gadget**。若继续沿此方向，需要在每个三角形周围增加连接/守卫点，并通过完整集合覆盖验证这些守卫点确实强制三顶点不可拆分；不能把组不可拆分作为搜索的先验假设。

## 9. 复现命令

```bash
uv run pytest -q tests/test_triangle_groups.py tests/test_partition_search.py

uv run python src/triangle_groups.py \
  --count 100 \
  --start-seed 0 \
  --time-limit 20 \
  --output data/triangle_trials_round1.json \
  --best-points data/triangle_best_points_round1.npy
```

严格半径第二轮的完整参数、模式哈希和最优值位于
[`data/triangle_trials_strict_round2.json`](../data/triangle_trials_strict_round2.json)。
