# 外层圆周子问题：56 点至少需要 16 个半径 $\lambda$ 的圆

本文研究 [`proof_cover.py`](../src/proof_cover.py) 的 25 圆构造中的外层子问题。目标圆盘已经归一化为单位圆盘，覆盖圆半径为

$$
\lambda=\frac1{\sqrt6+\sqrt3}
       =\frac{\sqrt6-\sqrt3}{3}
       =0.239146311738100\ldots.
$$

结论是：下面构造的 56 点集可以被 16 个半径 $\lambda$ 的圆覆盖，但不能被 15 个这样的圆覆盖。因此其最小拆分组数恰为 16。

对应的可执行验证器和完整 JSON 证书分别是：

- [`src/outer_partition.py`](../src/outer_partition.py)；
- [`data/outer_partition_certificate.json`](../data/outer_partition_certificate.json)。

配套矢量图：[`outer_partition_40_points_25_circles.svg`](outer_partition_40_points_25_circles.svg)。图中采用单位圆盘尺度，绘出完整的 25 个覆盖圆，并突出显示第 7 节前 8 个 $\mathcal B_k$ 组中的 40 个点；在浏览器中悬停点可查看名称和坐标。

## 1. 外层的 16 个已知覆盖圆

沿用交点文档中的记号。令

$$
q=\sqrt2,\qquad
a=\sqrt{2+\sqrt2},\qquad
\eta=\sqrt{2-\sqrt2},
$$

以及旋转矩阵

$$
\mathcal R_k=
\begin{pmatrix}
\cos(k\pi/4)&-\sin(k\pi/4)\\
\sin(k\pi/4)& \cos(k\pi/4)
\end{pmatrix},
\qquad k\in\mathbb Z/8\mathbb Z.
$$

与单位圆周相交的 16 个覆盖圆是 $B_k,C_k$ 两层。归一化后的圆心为

$$
\widehat B_k=\lambda\mathcal R_k(2+q,0),
\qquad
\widehat C_k=\lambda\mathcal R_k(2+q,q),
$$

半径均为 $\lambda$。

## 2. 点集的精确定义

### 2.1 单位圆周上的 16 个原有交点

定义

$$
\begin{aligned}
X_k^-&=\lambda\mathcal R_k
 \left(2+\frac{3q}{2},\frac q2\right),\\
X_k^+&=\lambda\mathcal R_k(2+q,1+q),
\end{aligned}
\qquad k=0,\ldots,7.
$$

它们满足 $|X_k^\pm|=1$。其中

$$
X_k^-\in B_k\cap C_k,
\qquad
X_k^+\in B_{k+1}\cap C_k.
$$

### 2.2 24 个半步相位的均匀圆周点

只使用上述 16 个交点和径向点还不够。加入

$$
Q_j=\left(
\cos\frac{(2j+1)\pi}{24},
\sin\frac{(2j+1)\pi}{24}
\right),
\qquad j=0,\ldots,23.
$$

这些点的角度为

$$
7.5^\circ,22.5^\circ,37.5^\circ,\ldots,352.5^\circ.
$$

相对于间隔 $15^\circ$ 的普通 24 边形，它们采用半步相位。这个相位很重要：不旋转的 24 个均匀点只给出 12 组下界，而半步相位版本给出 16。

### 2.3 外层圆的 16 个径向最内点

对于圆心 $c\ne0$、半径为 $\lambda$ 的圆，其圆周上到目标单位圆周距离最远的点就是径向最内点

$$
y=c-\lambda\frac{c}{|c|}.
$$

这是该小圆周上模长最小的唯一点。对 $B_k,C_k$ 分别得到

$$
Y_k^B
=\lambda\mathcal R_k(1+q,0)
$$

和

$$
Y_k^C
=\lambda\mathcal R_k
\left(2+q-\frac a2,
      q-\frac\eta2\right).
$$

等价地，$Y_k^C$ 位于角 $\pi/8+k\pi/4$ 的射线上，模长为

$$
|Y_k^C|=\lambda(2a-1).
$$

### 2.4 最终点集

最终使用

$$
P=
\{X_k^-,X_k^+:0\le k<8\}
\cup\{Q_j:0\le j<24\}
\cup\{Y_k^B,Y_k^C:0\le k<8\}.
$$

三部分互不重合，所以

$$
|P|=16+24+16=56.
$$

## 3. 为什么原始的 32 点不够

最自然的第一版是

$$
P_{32}=\{X_k^-,X_k^+,Y_k^B,Y_k^C:0\le k<8\}.
$$

完整枚举显示它只有 40 个最大可装三点模式，而且存在 11 组覆盖。因此“16 个边界交点 + 每个外层圆的一个径向最内点”本身不能证明 16 圆下界。

进一步加入 16 个当前短弧中点得到的 48 点集也只需要 14 组。真正起作用的是上一节的 24 个半步相位均匀圆周点；它们排除了跨相邻 $B/C$ 圆合并点组的替代方案。

## 4. 从几何拆分化为有限图覆盖

定义 56 个点的兼容图 $G$：不同顶点 $p_i,p_j$ 相邻，当且仅当

$$
|p_i-p_j|\le 2\lambda.
$$

若一个点组 $S\subseteq P$ 能被某个半径 $\lambda$ 的圆覆盖，则任意 $p_i,p_j\in S$ 都满足

$$
|p_i-p_j|\le|p_i-c|+|p_j-c|\le2\lambda.
$$

所以任何可覆盖点组都必定是 $G$ 的一个团。注意反方向一般不成立，但这里不需要反方向：允许使用所有图团只会放宽原几何拆分问题。若连这个放宽问题都不能用 15 个团覆盖，那么原问题当然也不能。

每个团都包含在某个最大团中，因此只需枚举最大团。Bron--Kerbosch 完整枚举得到：

| 最大团大小 | 个数 |
|---:|---:|
| 3 | 8 |
| 4 | 40 |
| 5 | 8 |
| **合计** | **56** |

## 5. 全部 56 个最大团的对称表达

所有下标分别按模 $8$ 或模 $24$ 解释。56 个最大团恰好由下面 7 类、每类 8 个组成。

### 5.1 五点团（8 个）

$$
\boxed{
\{X_k^-,X_{k-1}^+,Q_{3k-1},Q_{3k},Y_k^B\}
}
\qquad(k=0,\ldots,7).
$$

### 5.2 四点团（共 40 个）

对每个 $k=0,\ldots,7$，有以下五个：

$$
\begin{aligned}
&\{X_k^-,X_k^+,Q_{3k+1},Y_k^C\},\\
&\{X_k^-,Q_{3k},Q_{3k+1},Y_k^C\},\\
&\{X_k^+,Q_{3k+1},Q_{3k+2},Y_k^C\},\\
&\{X_k^-,Q_{3k},Y_k^B,Y_k^C\},\\
&\{X_k^+,Q_{3k+2},Y_{k+1}^B,Y_k^C\}.
\end{aligned}
$$

### 5.3 三点团（8 个）

$$
\boxed{
\{Y_k^B,Y_{k+1}^B,Y_k^C\}
}
\qquad(k=0,\ldots,7).
$$

因此搜索输入不是无限多个圆心，而只是上述 56 个有限模式。

## 6. 15 组排除搜索

令每个最大团对应一个布尔选择变量。搜索问题是：是否能选择至多 15 个最大团，使其并集包含全部 56 个点？

验证器使用 Python 整数 bitmask，并采用以下剪枝：

1. **最大新增覆盖数下界**：若剩余 $m$ 点，而任一模式最多新覆盖 $s$ 点，则还至少需要 $\lceil m/s\rceil$ 组。
2. **冲突点 packing 下界**：在距离大于 $2\lambda$ 的点中贪心寻找两两冲突集；其中每个点必须属于不同组。
3. **最受限点分支**：选择可用模式最少的未覆盖点，只枚举包含该点的最大团。
4. **状态内支配删除**：若一个分支在当前未覆盖点集上的新增覆盖严格包含另一个分支，则删除较弱分支。
5. **状态记忆化**：相同未覆盖 bitmask 若曾在不少于当前剩余组数下失败，则直接剪枝。

确定性运行结果为：

| 项目 | 结果 |
|---|---:|
| 顶点数 | 56 |
| 最大团数 | 56 |
| 根节点下界 | 12 |
| 贪心上界 | 16 |
| 判定目标 | $\le15$ 组 |
| DFS 节点数 | 2142 |
| 结果 | **不可行** |

因此

$$
\operatorname{cover}_{\lambda}(P)\ge16.
$$

## 7. 显式的 16 组上界

下面给出一个互不重叠、覆盖全部 56 点的 16 组拆分。前 8 组由已知 $B_k$ 圆覆盖：

$$
\mathcal B_k=
\{X_k^-,X_{k-1}^+,Q_{3k-1},Q_{3k},Y_k^B\},
\qquad k=0,\ldots,7.
$$

后 8 组由已知 $C_k$ 圆覆盖：

$$
\mathcal C_k=\{Q_{3k+1},Y_k^C\},
\qquad k=0,\ldots,7.
$$

$\mathcal B_k$ 共包含 40 点，$\mathcal C_k$ 共包含其余 16 点，并且各组两两不交。于是

$$
\operatorname{cover}_{\lambda}(P)\le16.
$$

结合上一节，得到

$$
\boxed{\operatorname{cover}_{\lambda}(P)=16}.
$$

## 8. 数值稳定性与可复核数据

使用 80 位 `mpmath` 独立重算全部两点距离后，所得兼容图与双精度版本完全一致。所有两点距离与阈值 $2\lambda$ 之间的最小绝对裕量为

$$
0.0015601212182043683488859063767991\ldots,
$$

远大于代码采用的 $2\times10^{-12}$ 容差。

关键哈希为：

```text
compatibility graph SHA-256:
50ef1ab52123408fb9e2e8cfd7b3fbc8a3659e8975f6035b4e580b3d0a3610a0

maximal clique list SHA-256:
21961374943bbe2233410306669aeb2ddf6d74142146aab43e75eeffe80321d9
```

完整点坐标、56 个最大团和显式 16 组拆分均保存在
[`data/outer_partition_certificate.json`](../data/outer_partition_certificate.json)。

复现命令：

```bash
uv run pytest -q tests/test_partition_search.py tests/test_outer_partition.py
uv run python src/outer_partition.py \
  --output data/outer_partition_certificate.json
```

当前验证将精确代数坐标以双精度构图，再用 80 位计算和显著阈值裕量作独立复核。若需要形式证明级别的证书，下一步可以把 1540 个两点距离比较改成代数数符号判号，并把 2142 节点搜索导出为独立 proof trace；这不会改变有限图或搜索结论。
