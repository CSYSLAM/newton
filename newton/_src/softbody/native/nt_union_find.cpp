// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Union-Find implementation, ported from PhysX ExtTetUnionFind.cpp.

#include "nt_union_find.h"

namespace nt {

void UnionFind::init(Pi32 numSets)
{
    mEntries.resize(numSets);
    for (Pi32 i = 0; i < numSets; i++) {
        Entry& e = mEntries[i];
        e.parent = i;
        e.rank = 0;
        e.setNr = i;
    }
}

Pi32 UnionFind::find(Pi32 x)
{
    if (mEntries[x].parent != x)
        mEntries[x].parent = find(mEntries[x].parent);
    return mEntries[x].parent;
}

void UnionFind::makeSet(Pi32 x, Pi32 y)
{
    Pi32 xroot = find(x);
    Pi32 yroot = find(y);
    if (xroot == yroot)
        return;
    if (mEntries[xroot].rank < mEntries[yroot].rank)
        mEntries[xroot].parent = yroot;
    else if (mEntries[xroot].rank > mEntries[yroot].rank)
        mEntries[yroot].parent = xroot;
    else {
        mEntries[yroot].parent = xroot;
        mEntries[xroot].rank++;
    }
}

Pi32 UnionFind::computeSetNrs()
{
    std::vector<Pi32> oldToNew(mEntries.size(), -1);
    Pi32 numSets = 0;

    for (Pi32 i = 0; i < Pi32(mEntries.size()); i++) {
        Pi32 nr = find(i);
        if (oldToNew[nr] < 0)
            oldToNew[nr] = numSets++;
        mEntries[i].setNr = oldToNew[nr];
    }
    return numSets;
}

Pi32 UnionFind::getSetNr(Pi32 x)
{
    return mEntries[x].setNr;
}

} // namespace nt