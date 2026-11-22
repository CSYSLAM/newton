# Upstream Newton base

This repository is a private copy of the `coupled-high-accuracy` branch, developed on top of
[Newton](https://github.com/newton-physics/newton). Use the base commit below to find which
upstream changes are missing when syncing.

| Item | Value |
| --- | --- |
| Upstream repository | https://github.com/newton-physics/newton |
| Upstream branch | `main` |
| Base commit (merge base with upstream `main`) | `8f266ff31b3c1c6caac945304a4fd46deae54150` |
| Base commit date | 2026-11-20 06:59:50 UTC |
| Base commit title | Improve implicit MPM volume and boundary preservation (#5021) |
| Newton version at the base | `1.9.0.dev0` |
| Branch head when this file was written | `ffd5f6104` (Skip rejected full-surface records in detection) |
| Recorded on | 2026-11-22 |

Update this table whenever upstream changes are merged in.

## Syncing upstream changes

```bash
git remote add upstream https://github.com/newton-physics/newton.git   # once
git fetch upstream main

# Upstream commits not yet included here:
git log --oneline 8f266ff31b3c1c6caac945304a4fd46deae54150..upstream/main

# Merge them (or rebase onto upstream/main), then resolve conflicts and run the tests:
git merge upstream/main
uv run --extra dev -m newton.tests
```

After merging, `git merge-base HEAD upstream/main` gives the new base commit; record it above.
