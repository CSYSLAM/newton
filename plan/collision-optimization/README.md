# Newton 碰撞加速优化 — 完整实现文档

> 覆盖 Phase 1 (LBVH Broadphase)、Phase 2 (AABB 缓存)、Phase 3 (Hill-climbing Support Mapping)
> 基于 commit `541fa97a` (Phase 1+2) 及工作区未提交修改 (Phase 3)

---

## 目录

- [01-overview.md](01-overview.md) — 总览：架构、数据流、修改文件清单
- [02-phase1-bvh-broadphase.md](02-phase1-bvh-broadphase.md) — Phase 1: LBVH Broadphase 完整实现
- [03-phase2-aabb-cache.md](03-phase2-aabb-cache.md) — Phase 2: AABB 缓存 (mesh_vs_convex midphase)
- [04-phase3-hill-climbing.md](04-phase3-hill-climbing.md) — Phase 3: Hill-climbing Support Mapping 完整实现
- [05-data-structures.md](05-data-structures.md) — 数据结构详解：CSR 格式、SupportMapDataProvider、Model 新字段
- [06-call-chain.md](06-call-chain.md) — 邻接数组贯穿调用链的完整路径
- [07-warp-caveats.md](07-warp-caveats.md) — Warp 开发踩坑记录
- [08-demos-and-tests.md](08-demos-and-tests.md) — Demo 与测试清单