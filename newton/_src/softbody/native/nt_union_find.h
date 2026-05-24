// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Union-Find data structure, ported from PhysX ExtTetUnionFind.
// Algorithm is identical; PxArray replaced with std::vector.

#pragma once

#include "nt_types.h"
#include <vector>

namespace nt {

class UnionFind {
public:
    UnionFind() {}
    UnionFind(Pi32 numSets) { init(numSets); }

    void init(Pi32 numSets);
    Pi32 find(Pi32 x);
    void makeSet(Pi32 x, Pi32 y);

    Pi32 computeSetNrs();
    Pi32 getSetNr(Pi32 x);

private:
    struct Entry {
        Pi32 parent, rank;
        Pi32 setNr;
    };

    std::vector<Entry> mEntries;
};

} // namespace nt