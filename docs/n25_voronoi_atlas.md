# n=25 clipped Voronoi/Delaunay atlas

> 状态：**exploration**；P0 语义验收：**PASS**。此产物不是 fixed-radius certificate，也不是 global optimality proof。

## 复现

```text
python src/n25_voronoi_atlas.py
```
JSON 确定性 SHA-256（排除运行时遥测）：`cda22f3719de1e20eb100ae011667fbe25952137b82be77c74e8b5726fc92257`；候选集合 SHA-256：`90ab466ea213d79085143e99d045634d5299854e82a3384ae1946c812c43b815`。

## 图谱计数

候选穷举：非平行三站点等距线 2216/2300；目标圆与垂直平分线候选根 600（300 对）；严格保留 48 个顶点，精确重复 1223 个。排除分类：{"exact_duplicate": 1223, "negative_boundary_radicand": 0, "not_nearest_site": 2480, "outside_target": 288}。
25 个有序圆心；48 个顶点，其中内部 32、目标边界 16；56 条内部 Voronoi 边、16 条目标边界弧；56 条 clipped Delaunay 边、32 个三角面。
Euler：V=48，E=72（内部 56 + 边界 16），F=25+外部面，V-E+F=2（通过：True）。边界 edge ID 全局固定为 56..71。

运行时遥测：总耗时 95.226s；判号调用 12978；sign cache {"currsize": 641, "hits": 12337, "maxsize": null, "misses": 641}；coordinate cache {"entries": 1593, "hits": 1223, "raw_duplicates": 1223}。

## 验证状态

- 候选集合：complete exact enumeration; no float filtering or deduplication。
- 坐标与判号：exact-algebraic expressions with principal-root rational intervals；exact active equalities and strictly positive rational interval margins。
- Cell cycle：exact polar comparator: y-sign/x-axis half-plane, exact cross sign, exact collinear radius/coordinate tie-break; duplicate points fail。
- 内部边：full segment certified by affine nearest-site differences at both endpoints。
- 全局边界 cycle：all 16 certified target-boundary vertices sorted by exact global CCW polar comparator。
- 边界弧：complete 300 bisector-pair / 600-root enumeration, global-cycle adjacency, and endpoint ownership certify each short directed arc with no omitted responsibility switch。
- 径向层：exact interval radial min/max certification。
- 浮点仅用于 discovery/display 与 D8 展示对象匹配；证书拓扑排序使用 exact polar comparator。

## 已证与未证边界

候选顶点集合、坐标、nearest-site、完整内部线段和目标圆边界弧责任、四层穿越均已用精确表达式与有理区间记录；浮点不参与候选集合、筛选、去重或严格状态。对象仍标为 `exploration`：没有枚举竞争覆盖构型、没有 Stage B 分支穷尽，也没有固定半径刚性或全局最优性证明。
