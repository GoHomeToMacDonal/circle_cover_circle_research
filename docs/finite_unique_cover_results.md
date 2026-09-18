# 有限私有顶点覆盖计划：计算与反证结果

本文记录对 [`finite_unique_cover_plan.md`](finite_unique_cover_plan.md) 的本地、可复现研究。结论是明确的否定结果：由原构型私有区域顶点得到的最自然同心收缩族虽然满足全部局部最小包围圆条件，但在整个靠近 `rho=1` 的尾区间都能被 **18 个半径严格小于目标半径的圆**覆盖，因此不能建立计划所需的全局命题。这个结论否定的是本文明确构造的 128 点族，不是否定一切可能的有限点证书族。

## 1. 私有区域组合结构

采用 `proof_cover_intersections.md` 的原尺度（覆盖圆半径 1，目标圆盘半径 `R=sqrt(6)+sqrt(3)`）。每个私有区域的顶点循环如下，下标模 8：

- `O`：`I_0,...,I_7`，8 个顶点；第 `k` 条边界弧由 `A_k` 圆承载。
- `A_k`：`I_k,I_{k+1},M_{k+1},T_k^+,T_k^-,M_k`，依次由 `O,A_{k+1},B_{k+1},C_k,B_k,A_{k-1}` 承载。
- `B_k`：`M_k,T_k^-,E_k^-,E_{k-1}^+,T_{k-1}^+`，依次由 `A_k,C_k,S_R,C_{k-1},A_{k-1}` 承载。
- `C_k`：`T_k^-,T_k^+,E_k^+,E_k^-`，依次由 `A_k,B_{k+1},S_R,B_k` 承载。

这里 `S_R` 是目标圆盘边界。代码表示位于 `src/private_region_family.py` 的 `owner_vertices()` 和 `boundary_arcs()`。总 owner-tagged 顶点数为

```text
8 + 8*6 + 8*5 + 8*4 = 128.
```

当 `rho=1` 时，共享副本合并为交点文档中的 48 个不同几何点；当 `rho<1` 时 128 个带 owner 的副本均保留。

## 2. 参数族与局部临界性

对 owner 圆心 `c_i` 和它的边界顶点 `p` 定义

```text
y_{i,p}(rho) = c_i + rho (p-c_i),       0 < rho <= 1.
```

归一化到单位目标圆盘后取 `q=lambda*y`，其中 `lambda=1/R`。由于每个关联满足 `|p-c_i|=1`，故

```text
|y_{i,p}(rho)-c_i| = rho.
```

每组方向不位于任何开半圆内；程序计算的最小角裕量恒为 `pi/4`。因此 owner 圆心位于该组凸包中，每组的最小包围圆恰为 owner 圆、半径恰为 `rho`。具体地，O 组由八重对称锁定；A 组包含反向点；B、C 组的最大相邻方向间隔均至多 `3*pi/4`。

所有 `y(rho)` 都是 owner 圆心与原紧点的凸组合；二者都在 `disk(0,R)` 中，所以归一化点始终在单位盘内。接近 1 的数值检查为：

| rho | 归一化最大模长 | 单位盘裕量 |
|---:|---:|---:|
| 0.99 | 0.9986211971246594 | 1.3788e-3 |
| 0.999 | 0.9998619478788274 | 1.3805e-4 |
| 0.9999 | 0.9999861930718934 | 1.3807e-5 |
| 0.999999 | 0.9999998619288316 | 1.3807e-7 |

这完成了计划的私有边界列举、参数化、单位盘包含性和局部 MEB 检查，但仍不排除跨组重组。

## 3. 可控时限的有限圆覆盖器

`src/finite_cover.py` 实现对任意有限二维点集和固定半径 `r` 的连续圆心完备离散化：

1. 加入以每个输入点为圆心的候选圆；
2. 对每对距离 `d<2r` 的点加入两个半径 `r` 圆周交点；
3. 对 `d=2r` 的相切点对加入唯一中点（旧 `certificate.py` 漏掉了这个闭边界情形）；
4. 计算每个候选圆的点集 bitmask，去重并删除被包含的非极大 mask；
5. 用贪心覆盖给上界，用点数界和两两冲突 packing 给下界，再以“最受限未覆盖点”分支做穷尽 DFS。

完备性的 locking 论证如下。任取一个非极大的可行圆，平移圆心直到至少一个点接触边界；若只有一个接触点，沿其切向继续移动，直到第二个点接触或原接触点成为直径方向的停点。前者是两个半径 `r` 圆周的交点，后者包含于点中心候选。故每个可行点子集包含在枚举出的候选 mask 中；只保留极大 mask 不改变集合覆盖最优值。

所有阶段共享显式预算：`--time-limit`、`--max-nodes`、`--max-candidates`。候选生成和 DFS 均定期输出 JSON 进度。任何预算耗尽只返回 `status=unknown` 和合法上下界，绝不误报不可行。CLI 示例：

```bash
uv run python src/finite_cover.py \
  --qrho 0.99 --time-limit 10 --max-nodes 200000 \
  --progress-interval 0.1 --output data/qrho_099_cover.json
```

几何层当前明确标记为 `arithmetic=float`。正裕量的可行 witness 可独立复核；若要把浮点超图的“不可行”升级为形式证明，还必须对全部临界 incidence 做精确代数或区间判号。

## 4. 分阶段实验

探索阶段在原尺度、目标半径 `r=rho` 下得到：

| rho | 极大模式数 | 贪心圆数 | 各组重新求 MEC 后的最大半径 | 半径比 |
|---:|---:|---:|---:|---:|
| 0.99 | 129 | 18 | 0.933118327836400 | 0.942543765491 |
| 0.999 | 129 | 18 | 0.924803412043798 | 0.925729141185 |
| 0.9999 | 129 | 18 | 0.923971920464538 | 0.924064326897 |
| 0.999999 | 201 | 18 | 0.923880456390819 | 0.923881380272 |

对 `rho=0.99`，穷尽 bitmask 搜索找到 16 圆覆盖并以 11,462 个 DFS 节点排除了 15 圆覆盖，因此浮点超图的组合最优值为 16。由于若干 incidence 正好处在边界，这个下界实验不作为最终形式证明；最终否定结论只使用下一节的精确代数可行证书。

## 5. 整个尾区间的精确反证

令

```text
h = cos(pi/8) = sqrt(2+sqrt(2))/2 = 0.9238795325112867...
rho0 = (1+h)/2 = 0.9619397662556434...
```

`src/verify_qrho_counterexample.py` 给出 18 个位于 `Q(sqrt(2))` 中的固定中心。它们由两个单独中心和四个四重旋转轨道组成；完整表达式保存在 `data/qrho_countercertificate.json`。SymPy 精确判号验证：

- 48 个极限几何点全部位于这 18 个半径 `h` 的圆内；
- 128 条 owner—顶点关联全部严格满足距离平方等于 1；
- `h^2=(2+sqrt(2))/4<1`。

对任意 owner-tagged 点，写 `p=y(1)`。若固定中心 `z` 在极限证书中覆盖 `p`，则三角不等式给出

```text
|y(rho)-z|
 <= |p-z| + |y(rho)-p|
 <= h + (1-rho)|c_i-p|
 = h + 1-rho.
```

并且

```text
rho - (h+1-rho) = 2(rho-rho0) > 0
```

对每个 `rho in (rho0,1)` 成立。因此同一组 18 个固定中心、半径 `h+1-rho` 已覆盖全部 128 点，而且该半径严格小于 `rho`。补入 7 个任意冗余圆后仍是一个 25 圆覆盖，于是

```text
tau_25(Q_rho) <= h+1-rho < rho.
```

缩放到计划的单位目标圆盘尺度后：

```text
tau_25(lambda Q_rho) < rho*lambda
for every rho in (rho0,1).
```

这与计划要求的核心等式方向相反，且反例覆盖完整尾区间，不是有限采样现象。

## 6. 结论边界与复现

已严格完成并否定的是由现有交点/私有区域直接得到的同心仿射收缩族。失败原因完全是全局跨组重组；局部共圆、凸包/MEB 和单位盘包含性均成立。本文不声称证明原始 `r(25)` 问题，也不排除另行设计带额外守卫点或非仿射变形的有限点族。

复现命令：

```bash
uv run pytest -q tests/test_private_region_family.py tests/test_finite_cover.py \
  tests/test_qrho_counterexample.py
uv run python src/private_region_family.py --rho 0.99 0.999 0.9999 0.999999 --json
uv run python src/finite_cover.py --qrho 0.99 --time-limit 10 --max-nodes 200000
uv run python src/verify_qrho_counterexample.py --output data/qrho_countercertificate.json
```

## 7. 后续：边界圆弧加密实验

在每个 owner 圆的目标盘内可行圆弧上做嵌套二分加点后，点数从 128 增至 304、592、1168。加点提高了有限覆盖数，但 level 3 的 1168 点在半径缩小 1% 后仍找到 21–22 圆覆盖；该 witness 会被 level 4 新点击穿，因而不能外推到连续圆弧。算法、预算、完整表格和复现命令见 [`boundary_density_experiment.md`](boundary_density_experiment.md)。

## 8. 后续：圆心同形微缩图形

对每个 outer 点 `c+rho*u` 在圆心附近加入 `c+epsilon*rho*u` 后，小 core 只略微提高覆盖数：outer level 0、严格半径 `0.99rho` 下，`epsilon=0.02/0.05/0.1` 的浮点超图最优值均为 17（无 core 为 16）；outer level 2 加 5% core 的 1184 点仍有 22 圆覆盖。所有代表 witness 中每个竞争圆仍同时混合 outer/core 和多个 owner，故单一同心微缩副本没有形成 must-link。详见 [`core_shape_experiment.md`](core_shape_experiment.md)。

## 9. 后续：多同心层与径向辐条

沿每个 owner-tagged 外方向加入多个径向 scale 后，约束明显增强：`small3/geometric6/uniform9` 的 512/921/1305 点在半径 `0.99rho` 下分别找到 17/23/25 圆覆盖；但 `rho=0.999999` 的 uniform9 又找到 23 圆，且所有竞争圆仍混合多个 owner 和层。uniform9 witness 在少数连续辐条上利用层间缺口；自适应补 8 个缺口中点后仍有 25 圆覆盖并把缺口转移。详见 [`spoke_experiment.md`](spoke_experiment.md)。
