# n=25 P1 fixed-radius search

> 状态：**exploration**。这是无对称浮点发现实验，不是 fixed-radius certificate，也不是 global optimality proof；未进入 Stage B。

## 复现

```text
.venv/bin/python src/n25_p1_search.py --smoke
.venv/bin/python -m py_compile src/hole.py src/n25_p1_search.py
```

共 16 runs（四个 sigma、两 seed、每档两 restart），每 run 12 轮、n_r=16、正常路径含最终 best_C 复评估共 38 次 H 评估，硬预算 250，bounded poll=2。若 stop_reason=max_hole_evals，则分类为 unresolved；wall time 约 15.087s，且未进入 manifest hash。
参考构型 H=1.000000000000002，geom.covering_radius=4.181540550352000，参考 topology hash=85a4572853d01422db89f97443cac2c29f62fec55e924a6594149705fb1523b0。
最终 H 范围=[1.000001031962350, 1.032058766013259]；分类={"alternate-numerical-candidate": 0, "not-covering": 16, "reference-like": 0, "unresolved": 0}。best congruence RMS=1.21263937e-06，max displacement=1.99515503e-06。

## 判据与比较

只有 H <= 1+strict_tol 且独立 geom.covering_radius >= R25-strict_tol 才通过覆盖门；预算耗尽的 run 不论数值门结果如何均分类为 unresolved。其他正常结束 run 再用 rotation/reflection + Hungarian/Procrustes 数值 congruence 分类。topology 比较先按 candidate-site 到 reference-site 的 permutation 重标，再比较规范化组合结构；raw boundary candidate switches 仅是 floating-point fingerprint，不参与 topology hash 或比较。
alternate-numerical-candidate found: **False**；本次预算内未发现替代覆盖候选。该结论即使为真也只表示浮点候选，不能表示精确等号覆盖。

## Active/topology 轨迹

轨迹 topology hash 种类数=1，跨轮 topology 变化=0；active 数值 hash 种类数=189，跨轮 active 数值变化=192；active kind/owners hash 种类数=48，跨轮组合变化=184。最终 topology hash 计数={"85a4572853d01422db89f97443cac2c29f62fec55e924a6594149705fb1523b0": 16}，最终 hash 与 P0 基线相同的 runs=16/16，字段比较={"boundary_owner_cycle_equal": 16, "counts_equal": 16, "delaunay_edges_equal": 16, "delaunay_faces_equal": 16}。每轮完整 active holes、active kind/owners、分离的 active hash、raw fingerprint、topology signature 和 signature_change 均保存在 JSON；候选来源全部标为浮点。

## 限制

候选枚举使用浮点 circumcenter、boundary bisector intersection 和 boundary stationary points；geom.covering_radius 是独立数值交叉检查而非区间证明。有限的 12 轮、250 次 H 评估和两次/轮随机 poll 不能排除未搜索的拓扑或连续候选，也不证明全局最优性。

确定性 manifest hash（排除 wall time）=`06fefe31636bfca216834aceeb3121a6962a6a05950ced62252cd2520108e433`。
