// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// MultiList: multiple linked lists in a common array with a free list.
// Ported from PhysX ExtMultiList.h. Algorithm identical; PxArray -> std::vector.

#pragma once

#include "nt_types.h"
#include <vector>

namespace nt {

template <class T>
class MultiList {
public:
    MultiList(Pi32 maxId = 0) {
        firstFree = -1;
        if (maxId > 0)
            first.reserve(maxId + 1);
    }

    void reserve(Pi32 maxId) { first.reserve(maxId + 1); }

    void clear() {
        first.clear();
        next.clear();
        items.clear();
        queryItems.clear();
        firstFree = -1;
    }

    Pi32 add(Pi32 id, const T& item);
    bool addUnique(Pi32 id, const T& item);
    bool exists(Pi32 id, const T& item) const;
    void remove(Pi32 id, const T& item);

    // used by VoxelTetrahedralizer::createTets for edge dedup
    mutable std::vector<T> queryItems;

private:
    std::vector<Pi32> first;
    std::vector<T> items;
    std::vector<Pi32> next;
    Pi32 firstFree;
};

// ---------- implementation ----------

template <class T>
Pi32 MultiList<T>::add(Pi32 id, const T& item)
{
    if (id >= Pi32(first.size()))
        first.resize(id + 1, -1);
    Pi32 pos = firstFree;
    if (pos >= 0)
        firstFree = next[firstFree];
    else {
        pos = Pi32(items.size());
        items.resize(items.size() + 1);
        next.resize(items.size() + 1);
    }
    next[pos] = first[id];
    first[id] = pos;
    items[pos] = item;
    return pos;
}

template <class T>
bool MultiList<T>::addUnique(Pi32 id, const T& item)
{
    if (exists(id, item))
        return false;
    add(id, item);
    return true;
}

template <class T>
bool MultiList<T>::exists(Pi32 id, const T& item) const
{
    if (id < 0 || id >= Pi32(first.size()))
        return false;
    Pi32 nr = first[id];
    while (nr >= 0) {
        if (items[nr] == item)
            return true;
        nr = next[nr];
    }
    return false;
}

template <class T>
void MultiList<T>::remove(Pi32 id, const T& itemNr)
{
    Pi32 nr = first[id];
    Pi32 prev = -1;
    while (nr >= 0 && items[nr] != itemNr) {
        prev = nr;
        nr = next[nr];
    }
    if (nr < 0)
        return;
    if (prev >= 0)
        next[prev] = next[nr];
    else
        first[id] = next[nr];
    next[nr] = firstFree;
    firstFree = nr;
}

} // namespace nt