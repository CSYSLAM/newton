# Upstream Newton base

This repository is a private copy of the `coupled-high-accuracy` branch, developed on top of
[Newton](https://github.com/newton-physics/newton). Use the base commit below to find which
upstream changes are missing when syncing.

| Item | Value |
| --- | --- |
| Upstream repository | https://github.com/newton-physics/newton |
| Upstream branch | `main` |
| Base commit (merge base with upstream `main`) | `f39d8199f18d5117f9dd8073c1eb71c9a196f020` |
| Base commit date | 2026-11-21 19:26:59 UTC |
| Base commit title | Differentiable ANYmal-D standing backflip — fixes four Featherstone adjoint defects (#5260) |
| Newton version at the base | `1.9.0.dev0` |
| Previous base | `8f266ff31` (#5021), merged up to `f39d8199f` on 2026-11-22 |
| Recorded on | 2026-11-22 |

Update this table whenever upstream changes are merged in.

## Syncing upstream changes

```bash
git remote add upstream https://github.com/newton-physics/newton.git   # once
git fetch upstream main

# Upstream commits not yet included here:
git log --oneline f39d8199f18d5117f9dd8073c1eb71c9a196f020..upstream/main

# Merge them (or rebase onto upstream/main), then resolve conflicts and run the tests:
git merge upstream/main
uv run --extra dev -m newton.tests
```

After merging, `git merge-base HEAD upstream/main` gives the new base commit; record it above.

Upstream removals can break branch-only code without a merge conflict. The 2026-11-22 merge, for example, removed
`Model.collide()`, `Model.contacts()` and `CollisionPipeline(soft_contact_margin=...)`, which CHA examples and demos
used. After each merge, search the branch's own files for removed names (see the merged commits' `.removed.md`
changelog fragments) and run the CHA tests and examples.
