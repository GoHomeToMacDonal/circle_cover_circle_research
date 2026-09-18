# 多同心层与径向辐条实验

## 构造

对每个 owner 的外点方向

```text
p = c_i + rho*u_j
```

和一组径向 scale `S subset [0,1)`，加入

```text
p(s) = c_i + s*rho*u_j,    s in S.
```

外层 `s=1` 始终保留。同一 `s` 的点形成与外边界同形的同心层；固定方向 `u_j` 上的各层点形成径向辐条。`s=0` 退化时每个 owner 只加入一个圆心点，避免按方向重复。

所有新增点都是圆心与外点的凸组合，因此仍在目标盘内。已知 owner 圆覆盖全部层，故目标半径 `rho` 下始终有 25 圆上界。

实现为 `boundary_with_radial_layers_family()`。

## 三种日程

| 名称 | 内层 scales | 点数 |
|---|---|---:|
| `small3` | 0.02, 0.05, 0.10 | 512 |
| `geometric6` | 0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70 | 921 |
| `uniform9` | 0, 0.10, 0.20, ..., 0.90 | 1,305 |

点数均采用 outer level 0 的 128 个 owner-tagged 方向。`geometric6` 包含 `small3` 的全部点，并将辐条延伸到中外层；`uniform9` 以步长 0.1 采样整条辐条。

## 预算和判读

`src/spoke_experiment.py` 使用完整 locking-centre 候选枚举和 bitmask DFS。每案有独立 wall-clock、候选和节点硬预算，并持续输出 JSON 进度。以下主实验均取

```text
rho = 0.99,       covering radius = 0.99*rho = 0.9801.
```

因此任何不超过 25 圆的 witness 都足以说明该离散点集仍不满足 `tau_25=rho`。所有 case 因继续尝试改善上界而最终超时，故表中是预算内界，不是精确最优值。

## 主结果

| 日程 | 点数 | 点对数 | 候选圆 | 极大 mask | 候选生成 | 覆盖数界 | 已找到圆数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `small3` | 512 | 130,816 | 56,136 | 3,080 | 3.55 s | 12–17 | 17 |
| `geometric6` | 921 | 423,660 | 165,273 | 9,384 | 15.23 s | 16–23 | 23 |
| `uniform9` | 1,305 | 850,860 | 314,305 | 20,952 | 40.53 s | 16–25 | 25 |

多层辐条比单一小 core 明显更强：small3 仍只有 17 圆，延伸到中外层后上升到 23，均匀整段采样则第一次把当前找到的上界推到 25。

但 `uniform9` 的 25 圆 witness 使用半径 `0.9801<0.99`，所以即使其最优值最终真是 25，它仍然直接否定所需临界等式，而不是支持它。当前搜索也没有证明 24 圆不可行；可靠结论只是 `16 <= optimum <= 25`。

## rho 趋近 1

取

```text
rho = 0.999999, radius = 0.99*rho = 0.98999901
```

对相同的 1,305 点 `uniform9` 日程，90 秒内找到 **23 圆**覆盖，界为 15–23。因而“上界达到 25”不是稳定的尾区间现象；更靠近 `rho=1` 时仍出现更强重组。

所有四份 witness 都已由保存圆心重新生成点集并重放，未覆盖点数均为 0；到目标半径 `rho` 的裕量约为 0.0099 或 0.01。

## 25 圆仍不是 owner 配置

`rho=0.99` 的 `uniform9` 25 圆 witness 具有：

- 25/25 个竞争圆都是 multi-owner；
- 25/25 个竞争圆都同时覆盖多个径向层；
- 单圆最多服务 6 个 owner、覆盖全部 11 类层（中心、0.1–0.9、outer）；
- 竞争圆心到最近已知 owner 圆心的距离最大约 0.7856。

`rho=0.999999` 的 23 圆 witness 同样全部 multi-owner、multi-layer，单圆最多 5 个 owner。可见辐条增加了所需圆数，却仍未把竞争圆锁回预定 owner。

## 连续辐条缺口与自适应 guard

`src/spoke_refinement.py` 对每条参数线段

```text
c_i + t*(p-c_i),    0 <= t <= 1
```

解析求每个竞争圆覆盖的 `t` 区间，并合并这些区间。`rho=0.99` 的 uniform9 witness：

- 120/128 条连续辐条已完全覆盖；
- 8 条存在层间缺口；
- 最大缺口宽度为 0.0833463，小于离散层间距 0.1。

因此该 witness 利用了相邻离散层之间的空隙。只在这 8 个缺口中点增加 guard，得到 1,313 点，再运行 90 秒：仍找到 25 圆覆盖，界为 16–25。新 witness 把缺口转移到 12 条其他辐条，最大宽度仍约 0.08335。

这说明自适应径向加点确实能击穿当前 witness，但优化器可以通过新的全局重组移动缺口；一次 guard 迭代没有提高 25 圆上界。

## 结论

1. 多同心层和跨尺度辐条明显强于单一微型 core：覆盖上界从 17 逐步上升到 23、25。
2. 但离散 `uniform9` 仍有 25 个半径严格小于 `rho` 的圆覆盖，因此尚不能给出临界性。
3. 在 `rho=0.999999` 时甚至仍只需 23 圆，说明当前固定日程不具尾区间稳定性。
4. 所有竞争圆仍混合多个 owner 和多个层，must-link 仍未形成。
5. 均匀层 witness 利用层间缺口；自适应 guard 会让缺口迁移。下一步应把“连续辐条覆盖”直接作为约束，或迭代加入 guard 并采用列生成，避免每轮重新枚举约 85 万点对。

复现：

```bash
uv run python src/spoke_experiment.py \
  --rho 0.99 --level 0 \
  --schedule small3 geometric6 uniform9 \
  --radius-factor 0.99 --time-limit 90 \
  --max-nodes 500000 --max-candidates 1000000 \
  --output data/spoke_scan.json

uv run python src/spoke_refinement.py data/spoke_uniform9_rho099.json \
  --time-limit 90 --output data/spoke_uniform9_refined1.json
```

数据文件：

- `data/spoke_small3_rho099.json`
- `data/spoke_geometric6_rho099.json`
- `data/spoke_uniform9_rho099.json`
- `data/spoke_uniform9_tail.json`
- `data/spoke_uniform9_refined1.json`
