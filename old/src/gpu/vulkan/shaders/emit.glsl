// src/gpu/vulkan/shaders/emit.glsl: what the node kernels do with one lane's children
// (mbgpu.cu: emitLane, flushStage). Included after common.glsl and expand_push.glsl, with
// LEAF defined 0 (children go to the next level) or 1 (children are leaves).

#if LEAF
shared uint hist[256];
#else
// Children are staged per subgroup and written out in contiguous runs, one atomic per
// flush. Each subgroup owns 5 x (its size) slots, one lane's children at most.
shared uint64_t stB1[BLOCK * 5u];
shared uint64_t stB2[BLOCK * 5u];
shared uint stPk[BLOCK * 5u];

uint stageBase() {
    return gl_SubgroupID * gl_SubgroupSize * 5u;
}

// Writes the subgroup's `staged` children to the next level (every invocation calls it).
void flushStage(Counters C, uint staged) {
    uint base = 0u;
    if (subgroupElect()) {
        base = atomicAdd(C.next, staged);
    }
    base = subgroupBroadcastFirst(base);
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    U64Ref o1 = U64Ref(pc.outB1);
    U64Ref o2 = U64Ref(pc.outB2);
    U32Ref op = U32Ref(pc.outPk);
    uint sb = stageBase();
    for (uint j = gl_SubgroupInvocationID; j < staged; j += gl_SubgroupSize) {
        uint at = base + j;
        if (at < pc.outCap) {
            o1.v[at] = stB1[sb + j];
            o2.v[at] = stB2[sb + j];
            op.v[at] = stPk[sb + j];
        }
    }
    subgroupMemoryBarrierShared();
    subgroupBarrier();
}
#endif

// One lane's children: mask bit k = amount k + 1, boards in cb1[k], cb2[k], move code code + k.
// Every invocation of the subgroup calls it (subgroup prefixes inside).
void emitLane(Globals g, Counters C, uint mask, uint64_t cb1[5], uint64_t cb2[5], uint code, uint seed,
              inout uint staged, inout uint leaves, inout uint inRange) {
#if !LEAF
    uint mine = bitCount(mask);
    uint offset = subgroupExclusiveAdd(mine);
    uint total = subgroupAdd(mine);
    if (staged + total > gl_SubgroupSize * 5u) {
        flushStage(C, staged);
        staged = 0u;
    }
    uint at = stageBase() + staged + offset;
    for (uint k = 0u; k < 5u; ++k) {
        if ((mask & (1u << k)) != 0u) {
            stB1[at] = cb1[k];
            stB2[at] = cb2[k];
            stPk[at] = ((code + k) << 24) | seed;
            ++at;
        }
    }
    staged += total;
#else
    uint64_t key[5];
    uint inMask = 0u;
    for (uint k = 0u; k < 5u; ++k) {
        key[k] = 0ul;
        if ((mask & (1u << k)) != 0u) {
            key[k] = mixKey(stateHash(cb1[k], cb2[k]));
            uint p = uint(key[k] >> 56);
            if (p >= g.pBegin && p < g.pEnd) {
                inMask |= 1u << k;
            }
        }
    }
    leaves += bitCount(mask);
    if (g.mode == LEAF_COUNTS) {
        for (uint k = 0u; k < 5u; ++k) {
            if ((mask & (1u << k)) != 0u) {
                atomicAdd(hist[uint(key[k] >> 56)], 1u);
            }
        }
    } else if (g.mode == LEAF_CHECKSUM) {
        uint s = pc.seedBase + seed;
        for (uint k = 0u; k < 5u; ++k) {
            if ((mask & (1u << k)) != 0u) {
                atomicAdd64(g.cnt, s, 1ul);
                atomicAdd64(g.sum, s, key[k]);
                atomicXor64(g.xr, s, key[k]);
            }
        }
    } else if (g.mode == LEAF_KEYS) {
        uint mine = bitCount(inMask);
        uint offset = subgroupExclusiveAdd(mine);
        uint total = subgroupAdd(mine);
        uint base = 0u;
        if (subgroupElect() && total != 0u) {
            base = atomicAdd(C.passed, total);
        }
        base = subgroupBroadcastFirst(base);
        uint at = base + offset;
        U64Ref ok = U64Ref(g.okey);
        for (uint k = 0u; k < 5u; ++k) {
            if ((inMask & (1u << k)) != 0u) {
                if (at < g.oCap) {
                    ok.v[at] = key[k];
                } else {
                    C.overflow = 1u;
                }
                ++at;
            }
        }
    } else {   // LEAF_FILTER
        inRange += bitCount(inMask);
        for (uint k = 0u; k < 5u; ++k) {
            if ((inMask & (1u << k)) != 0u && leafPasses(g, key[k])) {
                uint at = atomicAdd(C.passed, 1u);
                if (at < g.oCap) {
                    U64Ref(g.ob1).v[at] = cb1[k];
                    U64Ref(g.ob2).v[at] = cb2[k];
                    U64Ref(g.okey).v[at] = key[k];
                    U32Ref(g.oseed).v[at] = pc.seedBase + seed;
                }
            }
        }
    }
#endif
}

// The end of a node kernel: counters, and the partition histogram (LEAF_COUNTS).
void finishNodes(Globals g, Counters C, uint staged, uint leaves, uint inRange) {
#if LEAF
    SUBGROUP_ADD_TO(C.leaves, leaves)
    SUBGROUP_ADD_TO(C.inRange, inRange)
    if (g.mode == LEAF_COUNTS) {
        memoryBarrierShared();
        barrier();
        uint t = gl_LocalInvocationID.x;
        if (hist[t] != 0u) {
            atomicAdd64(g.partCounts, t, uint64_t(hist[t]));
        }
    }
#else
    if (staged != 0u) {
        flushStage(C, staged);
    }
#endif
}

void startNodes(Globals g) {
#if LEAF
    if (g.mode == LEAF_COUNTS) {
        hist[gl_LocalInvocationID.x] = 0u;
    }
#endif
}
