// src/gpu/mbgpu.cu
//
// The GPU module (interface and contract: src/gpu/mbgpu.h): the left stream, the
// audit of its bound, and the goal side's child keys for the index build.
// Normal boards only so far.
//
// Layout and rules are the CPU's, re-implemented without its headers (the
// solver is built with MinGW, this file with nvcc + MSVC):
// - B1B2: rows 0-2 in b1, rows 3-5 in b2; cell (row r, column c) of a half at
//   bit 51 - 18 * (r % 3) - 3 * c (lane_rotations.hpp). A lane read as 18 bits
//   has its first cell in the top 3 bits; shifting it forward by n cells is a
//   right rotation by 3n.
// - Moves are coded dir * 30 + lane * 5 + (amount - 1), dir 0 = rows, 1 = columns
//   (perm_stream_detail::decodePrevMove).
// - stream_lanes_dfs (ASCENDING, CHECK_CROSS, CHECK_SIM) and forEachFirstMove: after a
//   move on (dir, lane), same-direction moves only on later lanes; a lane stops at the
//   first amount that gives the board back (its period); after a column move, a row
//   move that commutes with it (Board::doActISColMatchBatched) is skipped.
// - Rotation lane bound (lane_bound.hpp, Kind::Rotation): h = max over rows of
//   d(row) + max over columns of d(column), d = fewest cells that differ from the
//   target lane over its 6 rotations; a child with r moves still below it is
//   dropped when h > limit + r, the leaves included (bounds::LanePruneLeaves).
//
// Work is level by level: one thread per node tries its 60 moves and appends the
// children it keeps to the next level's buffer (one atomic per warp and lane). The
// bound is evaluated incrementally from packed per-rotation mismatch counts (see
// expandNodes). A level is processed in chunks sized from the measured children per
// node, so their children fit the next buffer; a chunk that overflows it is run again
// smaller (nothing is consumed before the check). Every chunk's subtree is finished
// before the next chunk starts, so each level needs one fixed buffer and every
// kernel launch is short (WDDM resets a GPU whose kernel runs for ~2 s). The audit
// keeps the first kernel (expandKernel: one thread per (node, lane), h from its
// definition), so the audit also cross-checks the incremental bound.
// What happens to the leaves depends on the call: Bloom test + output (the
// probe), per-seed checksums (tests and the audit), keys in a partition range
// (index build) or per-partition counts.

#define MBGPU_BUILD_DLL
#include "src/gpu/mbgpu.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

namespace {

    constexpr u64 LANE_BITS = 0x3FFFFULL;
    constexpr u32 NO_PREV = 255;
    constexpr u32 THREADS_PER_NODE = 12;   // 6 rows + 6 columns
    constexpr u64 MAX_CHILDREN = 60;
    constexpr u64 FAT_BOOL_BIT = 1ULL << 57;
    constexpr u32 BLOCK = 256;             // the per-partition histogram needs exactly 256 threads

    // ---------------- board helpers (host and device) ----------------

    __host__ __device__ __forceinline__ u64 rotateLane(const u64 lane, const u32 cells) {
        const u32 s = 3 * cells;
        return ((lane >> s) | (lane << (18 - s))) & LANE_BITS;
    }

    __host__ __device__ __forceinline__ u64 rowLane(const u64 b1, const u64 b2, const u32 r) {
        return ((r < 3 ? b1 : b2) >> (36 - 18 * (r % 3))) & LANE_BITS;
    }

    __host__ __device__ __forceinline__ u64 colLane(const u64 b1, const u64 b2, const u32 x) {
        u64 lane = 0;
        for (u32 r = 0; r < 6; ++r) {
            const u64 half = r < 3 ? b1 : b2;
            lane |= ((half >> (51 - 18 * (r % 3) - 3 * x)) & 7ULL) << (15 - 3 * r);
        }
        return lane;
    }

    __host__ __device__ __forceinline__ u32 colorAt(const u64 b1, const u64 b2, const u32 x, const u32 y) {
        return static_cast<u32>(((y < 3 ? b1 : b2) >> (51 - 18 * (y % 3) - 3 * x)) & 7ULL);
    }

    __host__ __device__ __forceinline__ u64 columnMask(const u32 x) {
        return (7ULL << (51 - 3 * x)) | (7ULL << (33 - 3 * x)) | (7ULL << (15 - 3 * x));
    }

    // the board with lane (dir, lane) shifted by `n` (1..5); L = that lane, as read above
    __device__ __forceinline__ void shiftLane(const u64 b1, const u64 b2, const u32 dir, const u32 lane, const u64 L,
                                              const u32 n, u64& o1, u64& o2) {
        const u64 moved = rotateLane(L, n);
        if (dir == 0) {
            const u32 sh = 36 - 18 * (lane % 3);
            if (lane < 3) {
                o1 = (b1 & ~(LANE_BITS << sh)) | (moved << sh);
                o2 = b2;
            } else {
                o1 = b1;
                o2 = (b2 & ~(LANE_BITS << sh)) | (moved << sh);
            }
        } else {
            const u64 m = columnMask(lane);
            u64 n1 = b1 & ~m, n2 = b2 & ~m;
            for (u32 r = 0; r < 3; ++r) {
                n1 |= ((moved >> (15 - 3 * r)) & 7ULL) << (51 - 18 * r - 3 * lane);   // rows 0-2
                n2 |= ((moved >> (6 - 3 * r)) & 7ULL) << (51 - 18 * r - 3 * lane);    // rows 3-5
            }
            o1 = n1;
            o2 = n2;
        }
    }

    __host__ __device__ __forceinline__ u64 stateHash(const u64 x, const u64 y) {
        return x ^ ((y << 1) | (y >> 63)) ^ (x >> 17) ^ (y << 13);
    }

    __host__ __device__ __forceinline__ u64 mixKey(u64 x) {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    // ---------------- bound ----------------

    __constant__ u64 c_rowTarget[6];
    __constant__ u64 c_colTarget[6];
    __constant__ int c_limit;   // < 0: no pruning
    __constant__ u32 c_salt[8];
    // Packed mismatch counts (expandNodes): c_mc[(column c * 6 + row r) * 8 + colour v] has
    // nibble k = 1 when colour v at row r differs from column c's target cell at rotation k,
    // i.e. from target cell (r + k) % 6; c_mr the same for rows (row y, column x).
    __constant__ u32 c_mc[288];
    __constant__ u32 c_mr[288];

    // A lane's distance, in 32-bit arithmetic (a lane is 18 bits; 64-bit integer ops cost
    // the GPU two instructions each).
    __device__ __forceinline__ int laneDistance(const u32 lane, const u32 target) {
        int best = 6;
#pragma unroll
        for (u32 k = 0; k < 6; ++k) {
            const u32 s = 3 * k;
            u32 x = (((lane >> s) | (lane << (18 - s))) & 0x3FFFFu) ^ target;
            x = (x | (x >> 1) | (x >> 2)) & 0x9249u;
            best = min(best, __popc(x));
        }
        return best;
    }

    __device__ __forceinline__ u32 colLane32(const u64 b1, const u64 b2, const u32 x) {
        return ((static_cast<u32>(b1 >> (51 - 3 * x)) & 7u) << 15) | ((static_cast<u32>(b1 >> (33 - 3 * x)) & 7u) << 12) |
               ((static_cast<u32>(b1 >> (15 - 3 * x)) & 7u) << 9) | ((static_cast<u32>(b2 >> (51 - 3 * x)) & 7u) << 6) |
               ((static_cast<u32>(b2 >> (33 - 3 * x)) & 7u) << 3) | (static_cast<u32>(b2 >> (15 - 3 * x)) & 7u);
    }

    // The row (or column) half of h, stopping as soon as it exceeds `stopAbove`
    // (the caller then only needs to know that it does).
    __device__ __forceinline__ int rowHalf(const u64 b1, const u64 b2, const int stopAbove = 99) {
        int r = 0;
        for (u32 i = 0; i < 6 && r <= stopAbove; ++i) {
            r = max(r, laneDistance(static_cast<u32>(rowLane(b1, b2, i)), static_cast<u32>(c_rowTarget[i])));
        }
        return r;
    }

    __device__ __forceinline__ int colHalf(const u64 b1, const u64 b2, const int stopAbove = 99) {
        int c = 0;
        for (u32 i = 0; i < 6 && c <= stopAbove; ++i) {
            c = max(c, laneDistance(colLane32(b1, b2, i), static_cast<u32>(c_colTarget[i])));
        }
        return c;
    }

    // ---------------- buffers ----------------

    struct Nodes {
        u64* b1;
        u64* b2;
        u8* prev;
        u32* seed;
    };

    enum LeafMode : int { LEAF_FILTER = 0, LEAF_CHECKSUM = 1, LEAF_KEYS = 2, LEAF_COUNTS = 3 };

    struct LeafOut {
        int mode;
        unsigned long long* cnt;         // LEAF_CHECKSUM: per seed
        unsigned long long* sum;
        unsigned long long* xr;
        const u32* bloom;                // LEAF_FILTER; nullptr: every leaf in range passes
        const u64* match;                // LEAF_FILTER: if set, only leaves whose key is in this sorted list
        u64 matchCount;
        u64 blocksPerPartition;
        u32 pBegin, pEnd;                // LEAF_FILTER, LEAF_KEYS: partition range
        u64* ob1;                        // LEAF_FILTER output
        u64* ob2;
        u64* okey;                       // LEAF_FILTER and LEAF_KEYS output
        u32* oseed;
        u64 oCap;
        unsigned long long* partCounts;  // LEAF_COUNTS: 256 counters
    };

    struct Counters {
        unsigned long long next;         // nodes appended to the next level
        unsigned long long leaves;
        unsigned long long inRange;
        unsigned long long passed;       // leaves (or keys) written to the output buffer
        unsigned long long edges;        // audit
        unsigned long long violations;
        unsigned long long overLimit;
        int overflow;
    };

    struct Step {
        Nodes in;
        u64 n;
        Nodes out;
        u64 outCap;
        int rem;      // moves still below a child
        int leaf;     // the children are leaves
        int prune;    // apply the bound (when c_limit >= 0)
        int audit;    // check |dh| <= 1 on every edge; checksums count only leaves with h <= limit
    };

    __device__ __forceinline__ bool bloomPasses(const LeafOut& lo, const u64 key) {
        const u64 h = mixKey(key ^ 0x5851f42d4c957f2dULL);
        const u64 part = (key >> 56) - lo.pBegin;
        const u64 block = part * lo.blocksPerPartition + (((h >> 32) * lo.blocksPerPartition) >> 32);
        const u32* w = lo.bloom + block * 8;
        const u32 low = static_cast<u32>(h);
        u32 missing = 0;
#pragma unroll
        for (u32 i = 0; i < 8; ++i) {
            missing |= ~__ldg(&w[i]) & (1U << ((low * c_salt[i]) >> 27));
        }
        return missing == 0;
    }

    // LEAF_FILTER's test: an exact key list (mbgpu_match_children), else the Bloom filter, else all
    __device__ __forceinline__ bool leafPasses(const LeafOut& lo, const u64 key) {
        if (lo.match != nullptr) {
            u64 a = 0, b = lo.matchCount;
            while (a < b) {
                const u64 mid = (a + b) / 2;
                if (__ldg(&lo.match[mid]) < key) {
                    a = mid + 1;
                } else {
                    b = mid;
                }
            }
            return a < lo.matchCount && __ldg(&lo.match[a]) == key;
        }
        return lo.bloom == nullptr || bloomPasses(lo, key);
    }

    // Warp-wide exclusive prefix of `v` and the warp's total (every lane of the warp calls it).
    __device__ __forceinline__ u32 warpPrefix(const u32 v, u32& total) {
        const u32 lane = threadIdx.x & 31;
        u32 x = v;
#pragma unroll
        for (u32 d = 1; d < 32; d <<= 1) {
            const u32 y = __shfl_up_sync(0xFFFFFFFFu, x, d);
            if (lane >= d) {
                x += y;
            }
        }
        total = __shfl_sync(0xFFFFFFFFu, x, 31);
        return x - v;
    }

    // Adds `v` of every lane of the warp to *counter (one atomic per warp).
    __device__ __forceinline__ void warpAdd(unsigned long long* counter, const u32 v) {
        u32 total = 0;
        warpPrefix(v, total);
        if ((threadIdx.x & 31) == 0 && total != 0) {
            atomicAdd(counter, static_cast<unsigned long long>(total));
        }
    }

    // One thread per (node, lane).
    __global__ void expandKernel(const Step s, const LeafOut lo, Counters* ctr) {
        const u64 t = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
        const u64 i = t / THREADS_PER_NODE;

        u64 cb1[5], cb2[5];
        u32 amount[5];
        int hChild[5];
        u32 keep = 0, edges = 0, violations = 0;
        u32 codeBase = 0, seed = 0;

        if (i < s.n) {
            const u32 dl = static_cast<u32>(t % THREADS_PER_NODE);
            const u32 dir = dl / 6, lane = dl % 6;
            const u64 b1 = s.in.b1[i], b2 = s.in.b2[i];
            const u32 prev = s.in.prev[i];
            seed = s.in.seed[i];
            const int prevDir = prev == NO_PREV ? -1 : static_cast<int>(prev / 30);
            const u32 prevLane = (prev % 30) / 5, prevAmount = prev % 5 + 1;
            if (!(static_cast<int>(dir) == prevDir && lane <= prevLane)) {
                // after a column move, the row moves that commute with it
                u32 crossSkip = 0;
                if (prevDir == 1 && dir == 0) {
                    const u32 x1 = prevLane, y1 = lane;
                    const u32 pivot = colorAt(b1, b2, x1, y1);
                    if (colorAt(b1, b2, x1, (y1 + prevAmount) % 6) == pivot) {
                        for (u32 k = 0; k < 5; ++k) {
                            if (colorAt(b1, b2, (x1 + 5 - k) % 6, y1) == pivot) {
                                crossSkip |= 1U << k;
                            }
                        }
                    }
                }
                // The moved lane's half of h is the parent's: a lane's distance is the same
                // for all its rotations, and the other lanes of its kind do not move.
                const int limit = c_limit;
                const bool pruneOn = s.prune && limit >= 0;
                int fixedHalf = 0, hParent = 0;
                bool allFar = false;
                if (pruneOn) {
                    fixedHalf = dir == 0 ? rowHalf(b1, b2) : colHalf(b1, b2);
                    allFar = fixedHalf > limit + s.rem;
                }
                if (s.audit) {
                    hParent = rowHalf(b1, b2) + colHalf(b1, b2);
                }
                codeBase = dir * 30 + lane * 5;
                if (!allFar) {
                    const u64 L = dir == 0 ? rowLane(b1, b2, lane) : colLane(b1, b2, lane);
                    for (u32 k = 0; k < 5; ++k) {
                        u64 o1, o2;
                        shiftLane(b1, b2, dir, lane, L, k + 1, o1, o2);
                        if (o1 == b1 && o2 == b2) {
                            break;   // the lane's period: larger amounts repeat smaller ones
                        }
                        if (crossSkip & (1U << k)) {
                            continue;
                        }
                        if (pruneOn) {
                            const int room = limit + s.rem - fixedHalf;   // the other half may be at most this
                            if ((dir == 0 ? colHalf(o1, o2, room) : rowHalf(o1, o2, room)) > room) {
                                continue;
                            }
                        }
                        int hc = 0;
                        if (s.audit) {
                            hc = rowHalf(o1, o2) + colHalf(o1, o2);
                            ++edges;
                            violations += (hc > hParent + 1 || hc < hParent - 1) ? 1u : 0u;
                        }
                        cb1[keep] = o1;
                        cb2[keep] = o2;
                        amount[keep] = k;
                        hChild[keep] = hc;
                        ++keep;
                    }
                }
            }
        }
        if (s.audit) {
            warpAdd(&ctr->edges, edges);
            warpAdd(&ctr->violations, violations);
        }

        if (!s.leaf) {
            u32 total = 0;
            const u32 offset = warpPrefix(keep, total);
            unsigned long long base = 0;
            if ((threadIdx.x & 31) == 0 && total != 0) {
                base = atomicAdd(&ctr->next, static_cast<unsigned long long>(total));
            }
            base = __shfl_sync(0xFFFFFFFFu, base, 0);
            for (u32 j = 0; j < keep; ++j) {
                const u64 at = base + offset + j;
                if (at >= s.outCap) {
                    ctr->overflow = 1;
                    break;
                }
                s.out.b1[at] = cb1[j];
                s.out.b2[at] = cb2[j];
                s.out.prev[at] = static_cast<u8>(codeBase + amount[j]);
                s.out.seed[at] = seed;
            }
            return;
        }

        warpAdd(&ctr->leaves, keep);
        u64 key[5];
        for (u32 j = 0; j < keep; ++j) {
            key[j] = mixKey(stateHash(cb1[j], cb2[j]));
        }

        if (lo.mode == LEAF_COUNTS) {
            __shared__ u32 hist[256];
            hist[threadIdx.x] = 0;
            __syncthreads();
            for (u32 j = 0; j < keep; ++j) {
                atomicAdd(&hist[key[j] >> 56], 1u);
            }
            __syncthreads();
            if (hist[threadIdx.x] != 0) {
                atomicAdd(&lo.partCounts[threadIdx.x], static_cast<unsigned long long>(hist[threadIdx.x]));
            }
            return;
        }

        if (lo.mode == LEAF_CHECKSUM) {
            u32 over = 0;
            for (u32 j = 0; j < keep; ++j) {
                if (s.audit && hChild[j] > c_limit) {
                    ++over;
                    continue;   // the audit's checksums cover only the wanted leaves
                }
                atomicAdd(&lo.cnt[seed], 1ULL);
                atomicAdd(&lo.sum[seed], static_cast<unsigned long long>(key[j]));
                atomicXor(&lo.xr[seed], static_cast<unsigned long long>(key[j]));
            }
            if (s.audit) {
                warpAdd(&ctr->overLimit, over);
            }
            return;
        }

        if (lo.mode == LEAF_KEYS) {
            u64 mine[5];
            u32 m = 0;
            for (u32 j = 0; j < keep; ++j) {
                const u32 p = static_cast<u32>(key[j] >> 56);
                if (p >= lo.pBegin && p < lo.pEnd) {
                    mine[m++] = key[j];
                }
            }
            u32 total = 0;
            const u32 offset = warpPrefix(m, total);
            unsigned long long base = 0;
            if ((threadIdx.x & 31) == 0 && total != 0) {
                base = atomicAdd(&ctr->passed, static_cast<unsigned long long>(total));
            }
            base = __shfl_sync(0xFFFFFFFFu, base, 0);
            for (u32 j = 0; j < m; ++j) {
                const u64 at = base + offset + j;
                if (at >= lo.oCap) {
                    ctr->overflow = 1;
                    break;
                }
                lo.okey[at] = mine[j];
            }
            return;
        }

        // LEAF_FILTER
        u32 inRange = 0;
        for (u32 j = 0; j < keep; ++j) {
            const u32 p = static_cast<u32>(key[j] >> 56);
            if (p < lo.pBegin || p >= lo.pEnd) {
                continue;
            }
            ++inRange;
            if (lo.bloom != nullptr && !bloomPasses(lo, key[j])) {
                continue;
            }
            const unsigned long long at = atomicAdd(&ctr->passed, 1ULL);
            if (at >= lo.oCap) {
                ctr->overflow = 1;
                continue;
            }
            lo.ob1[at] = cb1[j];
            lo.ob2[at] = cb2[j];
            lo.okey[at] = key[j];
            lo.oseed[at] = seed;
        }
        warpAdd(&ctr->inRange, inRange);
    }

    // ---------------- the node kernel ----------------
    //
    // One thread per node: it tries all 12 lanes x 5 amounts, so the threads of a warp run
    // the same loop (the first kernel, one thread per (node, lane), kept ~10 of 32 threads
    // busy). The bound is incremental: a lane's count word has, in nibble k, how many of its
    // cells differ from its target lane at rotation k, and the lane's distance is its
    // smallest nibble. A row move changes one cell of every column, so a child's column
    // words are the parent's minus the old cell's mismatches plus the new cell's: one
    // lookup per column in c_mc (copied to shared memory), where the first kernel rotated
    // and compared every column six times. The moved row's own distance does not change.

    __device__ __forceinline__ u32 cellOf(const u32 lane, const u32 p) {
        return (lane >> (15 - 3 * p)) & 7u;
    }

    __device__ __forceinline__ u32 rot32(const u32 lane, const u32 cells) {
        const u32 s = 3 * cells;
        return ((lane >> s) | (lane << (18 - s))) & 0x3FFFFu;
    }

    // the smallest of the 6 nibbles of a count word
    __device__ __forceinline__ int minNibble(const u32 p) {
        const u32 a = (p & 0x0F0F0Fu) | 0xFF000000u;
        const u32 b = ((p >> 4) & 0x0F0F0Fu) | 0xFF000000u;
        u32 m = __vminu4(a, b);
        m = __vminu4(m, m >> 8);
        return static_cast<int>(min(m & 0xFFu, (m >> 8) & 0xFFu));
    }

    // The children of moving lane `lane` (DIR 0 = row, 1 = column; L = that lane), as a mask
    // of amounts (bit k = amount k + 1) with the boards in cb1[k], cb2[k]. `cross` = the count
    // words of the lanes the move crosses (columns for a row move), `tab` their table. A
    // child is dropped when a crossed lane's distance exceeds `room` (room >= 6: no test).
    template<u32 DIR>
    __device__ __forceinline__ u32 laneChildren(const u64 b1, const u64 b2, const u32 lane, const u32 L,
                                                const u32 (&cross)[6], const u32* tab, const int room,
                                                const u32 crossSkip, u64 (&cb1)[5], u64 (&cb2)[5]) {
        const bool test = room < 6;
        const u32 addv = test ? static_cast<u32>(7 - room) * 0x111111u : 0u;   // nibble > room <=> bit 3 set
        // per crossed lane c: its table row and its count word without the moving cell, plus addv
        const u32* row[6];
        u32 excl[6], v[6];
#pragma unroll
        for (u32 c = 0; c < 6; ++c) {
            v[c] = cellOf(L, c);
            row[c] = tab + (c * 6 + lane) * 8;
            excl[c] = cross[c] - row[c][v[c]] + addv;
        }
        u32 mask = 0;
        bool period = false;
#pragma unroll
        for (u32 k = 0; k < 5; ++k) {
            const u32 moved = rot32(L, k + 1);
            period = period || moved == L;   // the lane's period: larger amounts repeat smaller ones
            if (period || (crossSkip & (1u << k))) {
                continue;
            }
            if (test) {
                // after the shift by k + 1, crossed lane c gets the cell from position c - k - 1
                u32 tooFar = 0;
#pragma unroll
                for (u32 c = 0; c < 6; ++c) {
                    const u32 cnt = excl[c] + row[c][v[(c + 5 - k) % 6]];
                    tooFar |= (cnt & 0x888888u) == 0x888888u ? 1u : 0u;
                }
                if (tooFar) {
                    continue;
                }
            }
            u64 o1, o2;
            if (DIR == 0) {
                const u32 sh = 36 - 18 * (lane % 3);
                const u64 m64 = static_cast<u64>(moved) << sh;
                if (lane < 3) {
                    o1 = (b1 & ~(LANE_BITS << sh)) | m64;
                    o2 = b2;
                } else {
                    o1 = b1;
                    o2 = (b2 & ~(LANE_BITS << sh)) | m64;
                }
            } else {
                const u64 m = columnMask(lane);
                u64 n1 = b1 & ~m, n2 = b2 & ~m;
#pragma unroll
                for (u32 r = 0; r < 3; ++r) {
                    n1 |= static_cast<u64>((moved >> (15 - 3 * r)) & 7u) << (51 - 18 * r - 3 * lane);
                    n2 |= static_cast<u64>((moved >> (6 - 3 * r)) & 7u) << (51 - 18 * r - 3 * lane);
                }
                o1 = n1;
                o2 = n2;
            }
            cb1[k] = o1;
            cb2[k] = o2;
            mask |= 1u << k;
        }
        return mask;
    }

    // Children are staged per warp in shared memory and written out in contiguous runs
    // (written straight from registers, each thread's children land scattered and a 32-byte
    // sector carried ~7 useful bytes), with one atomic per flush.
    constexpr u32 STAGE = 160;   // children per warp: one lane's children at most (32 threads x 5)
    constexpr u32 WARPS = BLOCK / 32;

    struct Stage {
        u64 b1[WARPS][STAGE];
        u64 b2[WARPS][STAGE];
        u32 seed[WARPS][STAGE];
        u8 prev[WARPS][STAGE];
    };

    // Writes the warp's `staged` children to the next level (every lane of the warp calls it).
    __device__ __forceinline__ void flushStage(Stage& st, const u32 w, const u32 staged, const Step& s,
                                               Counters* ctr) {
        unsigned long long base = 0;
        if ((threadIdx.x & 31) == 0) {
            base = atomicAdd(&ctr->next, static_cast<unsigned long long>(staged));
        }
        base = __shfl_sync(0xFFFFFFFFu, base, 0);
        __syncwarp();
        for (u32 j = threadIdx.x & 31; j < staged; j += 32) {
            const u64 at = base + j;
            if (at < s.outCap) {
                s.out.b1[at] = st.b1[w][j];
                s.out.b2[at] = st.b2[w][j];
                s.out.prev[at] = st.prev[w][j];
                s.out.seed[at] = st.seed[w][j];
            }
        }
        __syncwarp();
    }

    // Children beyond s.outCap (non-leaf) or leaves beyond lo.oCap (LEAF_FILTER) are counted
    // but not written; the host sees the count and runs the chunk again smaller.
    // LEAF: the children are leaves (s.leaf); a separate build of each keeps the inner levels' kernel small.
    template<int LEAF>
    __global__ void __launch_bounds__(BLOCK, 3) expandNodes(const Step s, const LeafOut lo, Counters* ctr) {
        __shared__ u32 sMc[288], sMr[288];
        __shared__ u32 hist[256];
        __shared__ Stage stage;
        const u32 warp = threadIdx.x >> 5;
        u32 staged = 0;   // the same in every lane of the warp
        for (u32 j = threadIdx.x; j < 288; j += blockDim.x) {
            sMc[j] = c_mc[j];
            sMr[j] = c_mr[j];
        }
        const bool counts = LEAF && lo.mode == LEAF_COUNTS;
        if (counts) {
            hist[threadIdx.x] = 0;
        }
        __syncthreads();

        const u64 i = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
        const bool active = i < s.n;
        u64 b1 = 0, b2 = 0;
        u32 prev = NO_PREV, seed = 0;
        if (active) {
            b1 = s.in.b1[i];
            b2 = s.in.b2[i];
            prev = s.in.prev[i];
            seed = s.in.seed[i];
        }
        const int prevDir = prev == NO_PREV ? -1 : static_cast<int>(prev / 30);
        const u32 prevLane = (prev % 30) / 5, prevAmount = prev % 5 + 1;

        // the parent's count words
        u32 fullRow[6], fullCol[6];
#pragma unroll
        for (u32 y = 0; y < 6; ++y) {
            const u32 L = static_cast<u32>(rowLane(b1, b2, y));
            u32 w = 0;
#pragma unroll
            for (u32 x = 0; x < 6; ++x) {
                w += sMr[(y * 6 + x) * 8 + cellOf(L, x)];
            }
            fullRow[y] = w;
        }
#pragma unroll
        for (u32 x = 0; x < 6; ++x) {
            const u32 L = colLane32(b1, b2, x);
            u32 w = 0;
#pragma unroll
            for (u32 y = 0; y < 6; ++y) {
                w += sMc[(x * 6 + y) * 8 + cellOf(L, y)];
            }
            fullCol[x] = w;
        }
        const int limit = c_limit;
        const bool pruneOn = s.prune && limit >= 0;
        int rowH = 0, colH = 0;
        if (pruneOn) {
#pragma unroll
            for (u32 j = 0; j < 6; ++j) {
                rowH = max(rowH, minNibble(fullRow[j]));
                colH = max(colH, minNibble(fullCol[j]));
            }
        }

        u32 leaves = 0, inRange = 0;
#pragma unroll 1
        for (u32 dl = 0; dl < THREADS_PER_NODE; ++dl) {
            const u32 dir = dl < 6 ? 0u : 1u;
            const u32 lane = dl - 6 * dir;
            u64 cb1[5], cb2[5];
            u32 mask = 0;
            if (active && !(static_cast<int>(dir) == prevDir && lane <= prevLane)) {
                int room = 99;
                if (pruneOn) {
                    room = limit + s.rem - (dir == 0 ? rowH : colH);
                }
                if (room >= 0) {
                    if (dir == 0) {
                        // after a column move, the row moves that commute with it
                        u32 crossSkip = 0;
                        if (prevDir == 1) {
                            const u32 x1 = prevLane, y1 = lane;
                            const u32 pivot = colorAt(b1, b2, x1, y1);
                            if (colorAt(b1, b2, x1, (y1 + prevAmount) % 6) == pivot) {
                                for (u32 k = 0; k < 5; ++k) {
                                    if (colorAt(b1, b2, (x1 + 5 - k) % 6, y1) == pivot) {
                                        crossSkip |= 1U << k;
                                    }
                                }
                            }
                        }
                        mask = laneChildren<0>(b1, b2, lane, static_cast<u32>(rowLane(b1, b2, lane)), fullCol, sMc,
                                               room, crossSkip, cb1, cb2);
                    } else {
                        mask = laneChildren<1>(b1, b2, lane, colLane32(b1, b2, lane), fullRow, sMr, room, 0, cb1,
                                               cb2);
                    }
                }
            }
            const u32 code = dir * 30 + lane * 5;

            if (!LEAF) {
                u32 total = 0;
                const u32 offset = warpPrefix(__popc(mask), total);
                if (staged + total > STAGE) {
                    flushStage(stage, warp, staged, s, ctr);
                    staged = 0;
                }
                u32 at = staged + offset;
#pragma unroll
                for (u32 k = 0; k < 5; ++k) {
                    if (mask & (1u << k)) {
                        stage.b1[warp][at] = cb1[k];
                        stage.b2[warp][at] = cb2[k];
                        stage.prev[warp][at] = static_cast<u8>(code + k);
                        stage.seed[warp][at] = seed;
                        ++at;
                    }
                }
                staged += total;
                continue;
            }

            // leaves
            u64 key[5];
            u32 inMask = 0;
#pragma unroll
            for (u32 k = 0; k < 5; ++k) {
                key[k] = 0;
                if (mask & (1u << k)) {
                    key[k] = mixKey(stateHash(cb1[k], cb2[k]));
                    const u32 p = static_cast<u32>(key[k] >> 56);
                    if (p >= lo.pBegin && p < lo.pEnd) {
                        inMask |= 1u << k;
                    }
                }
            }
            leaves += __popc(mask);
            if (lo.mode == LEAF_COUNTS) {
#pragma unroll
                for (u32 k = 0; k < 5; ++k) {
                    if (mask & (1u << k)) {
                        atomicAdd(&hist[key[k] >> 56], 1u);
                    }
                }
            } else if (lo.mode == LEAF_CHECKSUM) {
#pragma unroll
                for (u32 k = 0; k < 5; ++k) {
                    if (mask & (1u << k)) {
                        atomicAdd(&lo.cnt[seed], 1ULL);
                        atomicAdd(&lo.sum[seed], static_cast<unsigned long long>(key[k]));
                        atomicXor(&lo.xr[seed], static_cast<unsigned long long>(key[k]));
                    }
                }
            } else if (lo.mode == LEAF_KEYS) {
                u32 total = 0;
                const u32 offset = warpPrefix(__popc(inMask), total);
                unsigned long long base = 0;
                if ((threadIdx.x & 31) == 0 && total != 0) {
                    base = atomicAdd(&ctr->passed, static_cast<unsigned long long>(total));
                }
                base = __shfl_sync(0xFFFFFFFFu, base, 0);
                u64 at = base + offset;
#pragma unroll
                for (u32 k = 0; k < 5; ++k) {
                    if (inMask & (1u << k)) {
                        if (at < lo.oCap) {
                            lo.okey[at] = key[k];
                        } else {
                            ctr->overflow = 1;
                        }
                        ++at;
                    }
                }
            } else {   // LEAF_FILTER
                inRange += __popc(inMask);
#pragma unroll
                for (u32 k = 0; k < 5; ++k) {
                    if ((inMask & (1u << k)) && leafPasses(lo, key[k])) {
                        const unsigned long long at = atomicAdd(&ctr->passed, 1ULL);
                        if (at < lo.oCap) {
                            lo.ob1[at] = cb1[k];
                            lo.ob2[at] = cb2[k];
                            lo.okey[at] = key[k];
                            lo.oseed[at] = seed;
                        }
                    }
                }
            }
        }
        if (LEAF) {
            warpAdd(&ctr->leaves, leaves);
            warpAdd(&ctr->inRange, inRange);
        } else if (staged != 0) {
            flushStage(stage, warp, staged, s, ctr);
        }
        if (counts) {
            __syncthreads();
            if (hist[threadIdx.x] != 0) {
                atomicAdd(&lo.partCounts[threadIdx.x], static_cast<unsigned long long>(hist[threadIdx.x]));
            }
        }
    }

    // ---------------- index build ----------------

    // first position in sorted keys[0, n) whose key is >= value
    __device__ __forceinline__ u64 lowerBound(const u64* keys, u64 lo, u64 hi, const u64 value) {
        while (lo < hi) {
            const u64 mid = lo + (hi - lo) / 2;
            if (keys[mid] < value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    // bounds[i] = first key of partition p0 + i (i = 0..parts); bounds[parts] = n
    __global__ void partitionBoundsKernel(const u64* keys, const u64 n, const u32 p0, const u32 parts, u64* bounds) {
        const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i <= parts) {
            const u32 p = p0 + i;
            bounds[i] = p >= 256 ? n : lowerBound(keys, 0, n, static_cast<u64>(p) << 56);
        }
    }

    __global__ void fingerprintKernel(const u64* keys, const u64 n, unsigned short* fps) {
        const u64 j = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (j < n) {
            fps[j] = static_cast<unsigned short>(keys[j]);
        }
    }

    // starts[b] = (first key in [lo, hi) whose bucket is >= b) - lo, b = 0..buckets
    __global__ void bucketStartsKernel(const u64* keys, const u64 lo, const u64 hi, const u32 bucketBits, u32* starts) {
        const u64 b = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
        const u64 buckets = 1ULL << bucketBits;
        if (b > buckets) {
            return;
        }
        if (b == buckets) {
            starts[b] = static_cast<u32>(hi - lo);
            return;
        }
        // keys of one partition share the top 8 bits; the bucket is the next bucketBits bits
        const u64 top = keys[lo] & (0xFFULL << 56);
        const u64 value = top | (b << (56 - bucketBits));
        starts[b] = static_cast<u32>(lowerBound(keys, lo, hi, value) - lo);
    }

    __global__ void bloomKernel(const u64* keys, const u64 lo, const u64 hi, const u64 blocksPerPartition, u32* bloom) {
        const u64 j = lo + static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (j >= hi) {
            return;
        }
        const u64 h = mixKey(keys[j] ^ 0x5851f42d4c957f2dULL);
        u32* w = bloom + (((h >> 32) * blocksPerPartition) >> 32) * 8;
        const u32 low = static_cast<u32>(h);
#pragma unroll
        for (u32 i = 0; i < 8; ++i) {
            atomicOr(&w[i], 1U << ((low * c_salt[i]) >> 27));
        }
    }

    __global__ void boundKernel(const u64* states, const u64 n, int* h) {
        const u64 i = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) {
            const u64 b1 = states[2 * i], b2 = states[2 * i + 1];
            h[i] = rowHalf(b1, b2) + colHalf(b1, b2);
        }
    }

    // ---------------- host side ----------------

    std::string g_error;

    bool fail(const std::string& what) {
        g_error = what;
        return false;
    }

#define MB_CUDA(call)                                                                          \
    do {                                                                                       \
        const cudaError_t e_ = (call);                                                         \
        if (e_ != cudaSuccess) {                                                               \
            return fail(std::string(#call) + ": " + cudaGetErrorString(e_));                   \
        }                                                                                      \
    } while (0)

    struct Device {
        bool open = false;
        int index = -1;
        u64 cap = 0;          // VRAM this module may use
        u64 used = 0;         // VRAM it holds
        u64 levelCap = 0;     // nodes per level buffer
        Nodes level[MBGPU_MAX_DEPTH] = {};
        Counters* ctr = nullptr;
        // output of leaves that passed
        u64 outCap = 0;
        u64 *ob1 = nullptr, *ob2 = nullptr, *okey = nullptr;
        u32* oseed = nullptr;
        // filter
        u32* bloom = nullptr;
        u64 bloomWords = 0;
        u64 blocksPerPartition = 0;
        u32 pBegin = 0, pEnd = 256;
        int limit = -1;
        bool targetSet = false;
    };

    Device g_dev;

    bool allocate(void** p, const u64 bytes) {
        if (g_dev.used + bytes > g_dev.cap) {
            return fail("VRAM cap reached: " + std::to_string((g_dev.used + bytes) >> 20) + " MB wanted, cap " +
                        std::to_string(g_dev.cap >> 20) + " MB");
        }
        MB_CUDA(cudaMalloc(p, bytes));
        g_dev.used += bytes;
        return true;
    }

    void release(void* p, const u64 bytes) {
        if (p != nullptr) {
            cudaFree(p);
            g_dev.used -= bytes;
        }
    }

    // Device memory freed when the scope ends.
    struct Scratch {
        void* p = nullptr;
        u64 bytes = 0;
        bool get(const u64 b) {
            bytes = b;
            return allocate(&p, b);
        }
        ~Scratch() { release(p, p != nullptr ? bytes : 0); }
    };

    struct Run {
        int depth = 0;
        int prune = 1;
        int audit = 0;
        LeafOut lo{};
        mbgpu_sink_fn sink = nullptr;
        void* ctx = nullptr;
        mbgpu_stats* stats = nullptr;
        std::vector<u64> hb1, hb2, hkey;
        std::vector<u32> hseed;
        double ratio[MBGPU_MAX_DEPTH] = {60, 60, 60, 60, 60, 60};   // children per node, as measured
        Counters snap{};   // the device counters after the last launch
    };

    bool readCounters(Counters& c) {
        MB_CUDA(cudaMemcpy(&c, g_dev.ctr, sizeof(Counters), cudaMemcpyDeviceToHost));
        if (c.overflow) {
            return fail("buffer overflow on the device");
        }
        return true;
    }

    bool flushOutput(Run& run) {
        Counters c{};
        if (!readCounters(c)) {
            return false;
        }
        const u64 n = c.passed;
        if (n == 0) {
            return true;
        }
        run.hb1.resize(n);
        run.hb2.resize(n);
        run.hkey.resize(n);
        run.hseed.resize(n);
        MB_CUDA(cudaMemcpy(run.hb1.data(), g_dev.ob1, n * 8, cudaMemcpyDeviceToHost));
        MB_CUDA(cudaMemcpy(run.hb2.data(), g_dev.ob2, n * 8, cudaMemcpyDeviceToHost));
        MB_CUDA(cudaMemcpy(run.hkey.data(), g_dev.okey, n * 8, cudaMemcpyDeviceToHost));
        MB_CUDA(cudaMemcpy(run.hseed.data(), g_dev.oseed, n * 4, cudaMemcpyDeviceToHost));
        if (run.sink != nullptr) {
            run.sink(run.ctx, run.hb1.data(), run.hb2.data(), run.hkey.data(), run.hseed.data(), n);
        }
        run.stats->passed += n;
        const unsigned long long zero = 0;
        MB_CUDA(cudaMemcpy(&g_dev.ctr->passed, &zero, sizeof(zero), cudaMemcpyHostToDevice));
        run.snap.passed = 0;
        return true;
    }

    Nodes offsetNodes(const Nodes& a, const u64 off) {
        return Nodes{a.b1 + off, a.b2 + off, a.prev + off, a.seed + off};
    }

    bool launch(Run& run, const int d, const u64 off, const u64 m, const bool leaf) {
        Step s{};
        s.in = offsetNodes(g_dev.level[d], off);
        s.n = m;
        s.out = leaf ? Nodes{} : g_dev.level[d + 1];
        s.outCap = g_dev.levelCap;
        s.rem = run.depth - (d + 1);
        s.leaf = leaf ? 1 : 0;
        s.prune = run.prune;
        s.audit = run.audit;
        if (run.audit) {
            const u64 threads = m * THREADS_PER_NODE;
            expandKernel<<<static_cast<u32>((threads + BLOCK - 1) / BLOCK), BLOCK>>>(s, run.lo, g_dev.ctr);
        } else {
            if (leaf) {
                expandNodes<1><<<static_cast<u32>((m + BLOCK - 1) / BLOCK), BLOCK>>>(s, run.lo, g_dev.ctr);
            } else {
                expandNodes<0><<<static_cast<u32>((m + BLOCK - 1) / BLOCK), BLOCK>>>(s, run.lo, g_dev.ctr);
            }
        }
        MB_CUDA(cudaGetLastError());
        run.stats->launches++;
        return true;
    }

    // Expands level `d` (n nodes in g_dev.level[d]) down to the leaves. Chunks are sized from
    // the children per node measured so far (80% of the next buffer); a chunk whose children
    // overflow it is run again smaller (its partial output is never used). Leaf chunks take
    // the whole level; with LEAF_FILTER, a chunk whose passing leaves overflow the output
    // buffer is undone (counters restored, the leaves before it handed over) and run in
    // halves. The audit's kernel keeps worst-case chunks (its edge counters cannot be undone).
    bool processLevel(Run& run, const int d, const u64 n) {
        const int childDepth = d + 1;
        const bool leaf = childDepth == run.depth;
        u64 mMax = n;
        for (u64 off = 0; off < n;) {
            u64 m = std::min(n - off, mMax);
            if (run.audit) {
                m = std::min(m, std::max<u64>(1, g_dev.levelCap / MAX_CHILDREN));
            } else if (!leaf) {
                const double r = std::max(run.ratio[d], 1e-3);
                m = std::min(m, std::max<u64>(1, static_cast<u64>(0.8 * static_cast<double>(g_dev.levelCap) / r)));
            }
            if (!leaf) {
                const unsigned long long zero = 0;
                MB_CUDA(cudaMemcpy(&g_dev.ctr->next, &zero, sizeof(zero), cudaMemcpyHostToDevice));
            }
            if (!launch(run, d, off, m, leaf)) {
                return false;
            }
            Counters c{};
            if (!readCounters(c)) {
                return false;
            }
            if (!leaf) {
                const double r = static_cast<double>(c.next) / static_cast<double>(m);
                if (c.next > g_dev.levelCap) {
                    if (run.audit || m == 1) {
                        return fail("level buffer overflow");
                    }
                    run.ratio[d] = std::max(run.ratio[d], r);
                    run.stats->retries++;
                    continue;
                }
                run.ratio[d] = r;
                run.snap = c;
                run.stats->nodes[childDepth] += c.next;
                off += m;
                if (c.next != 0 && !processLevel(run, childDepth, c.next)) {
                    return false;
                }
                continue;
            }
            if (run.lo.mode == LEAF_FILTER && c.passed > g_dev.outCap) {
                if (m == 1) {
                    return fail("output buffer overflow");
                }
                MB_CUDA(cudaMemcpy(g_dev.ctr, &run.snap, sizeof(Counters), cudaMemcpyHostToDevice));
                if (!flushOutput(run)) {
                    return false;
                }
                mMax = m / 2;
                run.stats->retries++;
                continue;
            }
            run.snap = c;
            off += m;
            if (run.lo.mode == LEAF_FILTER && c.passed > g_dev.outCap / 2 && !flushOutput(run)) {
                return false;
            }
        }
        return true;
    }

    bool streamAll(Run& run, const u64* states, const u8* prev, const u64 count, Counters* totals = nullptr) {
        if (!g_dev.open) {
            return fail("mbgpu_open has not been called");
        }
        if ((run.prune || run.audit) && !g_dev.targetSet) {
            return fail("mbgpu_set_target has not been called");
        }
        if (run.depth < 1 || run.depth > MBGPU_MAX_DEPTH) {
            return fail("depth must be 1.." + std::to_string(MBGPU_MAX_DEPTH));
        }
        if (count > 0xFFFFFFFFULL) {
            return fail("too many seeds for 32-bit seed indices");
        }
        if (count != 0 && (states[0] & FAT_BOOL_BIT)) {
            return fail("2x2 boards are not supported yet");
        }
        const auto start = std::chrono::steady_clock::now();
        Counters zero{};
        MB_CUDA(cudaMemcpy(g_dev.ctr, &zero, sizeof(Counters), cudaMemcpyHostToDevice));
        run.snap = zero;
        std::vector<u32> seedIdx;
        std::vector<u64> hb1, hb2;
        std::vector<u8> hprev;
        for (u64 off = 0; off < count; off += g_dev.levelCap) {
            const u64 m = std::min(g_dev.levelCap, count - off);
            seedIdx.resize(m);
            hb1.resize(m);
            hb2.resize(m);
            hprev.resize(m);
            for (u64 j = 0; j < m; ++j) {
                seedIdx[j] = static_cast<u32>(off + j);
                hb1[j] = states[2 * (off + j)];
                hb2[j] = states[2 * (off + j) + 1];
                hprev[j] = prev != nullptr ? prev[off + j] : static_cast<u8>(NO_PREV);
            }
            const Nodes& l0 = g_dev.level[0];
            MB_CUDA(cudaMemcpy(l0.b1, hb1.data(), m * 8, cudaMemcpyHostToDevice));
            MB_CUDA(cudaMemcpy(l0.b2, hb2.data(), m * 8, cudaMemcpyHostToDevice));
            MB_CUDA(cudaMemcpy(l0.prev, hprev.data(), m, cudaMemcpyHostToDevice));
            MB_CUDA(cudaMemcpy(l0.seed, seedIdx.data(), m * 4, cudaMemcpyHostToDevice));
            run.stats->nodes[0] += m;
            if (!processLevel(run, 0, m)) {
                return false;
            }
        }
        if (run.lo.mode == LEAF_FILTER && !flushOutput(run)) {
            return false;
        }
        Counters c{};
        if (!readCounters(c)) {
            return false;
        }
        run.stats->leaves += c.leaves;
        run.stats->inRange += c.inRange;
        run.stats->seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (totals != nullptr) {
            *totals = c;
        }
        return true;
    }

    // Per-seed checksum buffers on the device, copied out and freed at the end.
    bool checksumRun(Run& run, const u64* states, const u8* prev, const u64 count, uint64_t* cntOut,
                     uint64_t* sumOut, uint64_t* xrOut, Counters* totals) {
        const u64 bytes = std::max<u64>(1, count) * 8;
        Scratch cnt, sum, xr;
        if (!cnt.get(bytes) || !sum.get(bytes) || !xr.get(bytes)) {
            return false;
        }
        MB_CUDA(cudaMemset(cnt.p, 0, bytes));
        MB_CUDA(cudaMemset(sum.p, 0, bytes));
        MB_CUDA(cudaMemset(xr.p, 0, bytes));
        run.lo.mode = LEAF_CHECKSUM;
        run.lo.cnt = static_cast<unsigned long long*>(cnt.p);
        run.lo.sum = static_cast<unsigned long long*>(sum.p);
        run.lo.xr = static_cast<unsigned long long*>(xr.p);
        if (!streamAll(run, states, prev, count, totals)) {
            return false;
        }
        MB_CUDA(cudaMemcpy(cntOut, cnt.p, count * 8, cudaMemcpyDeviceToHost));
        MB_CUDA(cudaMemcpy(sumOut, sum.p, count * 8, cudaMemcpyDeviceToHost));
        MB_CUDA(cudaMemcpy(xrOut, xr.p, count * 8, cudaMemcpyDeviceToHost));
        return true;
    }

    // Raw keys one sort can take: two key buffers + CUB's scratch (~ one more) in what is left.
    u64 keyCapacity() {
        const u64 left = g_dev.cap > g_dev.used ? g_dev.cap - g_dev.used : 0;
        const u64 margin = 64ULL << 20;
        return left > margin ? (left - margin) / 25 : 0;
    }

    // The children's keys in partitions [p0, p1), sorted and unique, left in a.p (count *unique);
    // b is scratch of the same size. `reserve` bytes are kept free for the caller.
    bool sortedChildKeys(const u64* states, const u8* prev, const u64 count, const u32 p0, const u32 p1,
                         const u64 reserve, Scratch& a, Scratch& b, u64* unique) {
        const u64 left = g_dev.cap > g_dev.used + reserve ? g_dev.cap - g_dev.used - reserve : 0;
        const u64 cap = left > (64ULL << 20) ? (left - (64ULL << 20)) / 25 : 0;
        if (cap < (1ULL << 20)) {
            return fail("too little VRAM left to sort keys");
        }
        if (!a.get(cap * 8) || !b.get(cap * 8)) {
            return false;
        }
        mbgpu_stats st{};
        Run run;
        run.depth = 1;
        run.prune = 0;
        run.stats = &st;
        run.lo.mode = LEAF_KEYS;
        run.lo.pBegin = p0;
        run.lo.pEnd = p1;
        run.lo.okey = static_cast<u64*>(a.p);
        run.lo.oCap = cap;
        Counters c{};
        if (!streamAll(run, states, prev, count, &c)) {
            return false;
        }
        const u64 n = c.passed;
        if (n > cap) {
            return fail("raw keys of partitions [" + std::to_string(p0) + "," + std::to_string(p1) +
                        ") exceed the device buffer");
        }
        u64* keysA = static_cast<u64*>(a.p);
        u64* keysB = static_cast<u64*>(b.p);
        Scratch numOut;
        if (!numOut.get(8)) {
            return false;
        }
        size_t sortBytes = 0, uniqueBytes = 0;
        MB_CUDA(cub::DeviceRadixSort::SortKeys(nullptr, sortBytes, keysA, keysB, static_cast<int64_t>(n)));
        MB_CUDA(cub::DeviceSelect::Unique(nullptr, uniqueBytes, keysB, keysA, static_cast<u64*>(numOut.p),
                                          static_cast<int64_t>(n)));
        Scratch temp;
        if (!temp.get(std::max<u64>(1, std::max<u64>(sortBytes, uniqueBytes)))) {
            return false;
        }
        MB_CUDA(cub::DeviceRadixSort::SortKeys(temp.p, sortBytes, keysA, keysB, static_cast<int64_t>(n)));
        MB_CUDA(cub::DeviceSelect::Unique(temp.p, uniqueBytes, keysB, keysA, static_cast<u64*>(numOut.p),
                                          static_cast<int64_t>(n)));
        MB_CUDA(cudaMemcpy(unique, numOut.p, 8, cudaMemcpyDeviceToHost));
        return true;
    }

    bool childKeys(const u64* states, const u8* prev, const u64 count, const u32 p0, const u32 p1, u64* out,
                   const u64 outCap, u64* outCount) {
        Scratch a, b;
        u64 unique = 0;
        if (!sortedChildKeys(states, prev, count, p0, p1, 0, a, b, &unique)) {
            return false;
        }
        if (unique > outCap) {
            return fail("output buffer too small for the unique keys");
        }
        MB_CUDA(cudaMemcpy(out, a.p, unique * 8, cudaMemcpyDeviceToHost));
        *outCount = unique;
        return true;
    }

    // The per-partition device buffers of buildIndex: starts and Bloom blocks of one partition.
    u64 indexPartitionBytes(const u32 bucketBits, const u64 blocksPerPartition) {
        return ((1ULL << bucketBits) + 1) * 4 + blocksPerPartition * 32 + 257 * 8;
    }

    bool buildIndex(const u64* states, const u8* prev, const u64 count, const u32 p0, const u32 p1,
                    const u32 bucketBits, const u64 bpp, unsigned short* fpsOut, const u64 fpsCap, u32* startsOut,
                    u32* bloomOut, uint64_t* uniqueOut) {
        const u64 partBytes = indexPartitionBytes(bucketBits, bpp);
        Scratch a, b;
        u64 unique = 0;
        if (!sortedChildKeys(states, prev, count, p0, p1, partBytes, a, b, &unique)) {
            return false;
        }
        if (unique > fpsCap) {
            return fail("fingerprint buffer too small");
        }
        const u64* keys = static_cast<const u64*>(a.p);
        const u32 parts = p1 - p0;
        // partition boundaries
        Scratch bnd, starts, bloom;
        if (!bnd.get(257 * 8) || !starts.get(((1ULL << bucketBits) + 1) * 4) || !bloom.get(bpp * 32)) {
            return false;
        }
        partitionBoundsKernel<<<2, 256>>>(keys, unique, p0, parts, static_cast<u64*>(bnd.p));
        MB_CUDA(cudaGetLastError());
        std::vector<u64> bounds(parts + 1);
        MB_CUDA(cudaMemcpy(bounds.data(), bnd.p, (parts + 1) * 8, cudaMemcpyDeviceToHost));
        // fingerprints of the whole group (in b, which is free now)
        auto* fps = static_cast<unsigned short*>(b.p);
        if (unique != 0) {
            fingerprintKernel<<<static_cast<u32>((unique + BLOCK - 1) / BLOCK), BLOCK>>>(keys, unique, fps);
            MB_CUDA(cudaGetLastError());
            MB_CUDA(cudaMemcpy(fpsOut, fps, unique * 2, cudaMemcpyDeviceToHost));
        }
        const u64 dirPer = (1ULL << bucketBits) + 1;
        for (u32 i = 0; i < parts; ++i) {
            const u64 lo = bounds[i], hi = bounds[i + 1];
            uniqueOut[i] = hi - lo;
            u32* st = static_cast<u32*>(starts.p);
            if (hi > lo) {
                bucketStartsKernel<<<static_cast<u32>((dirPer + BLOCK - 1) / BLOCK), BLOCK>>>(keys, lo, hi, bucketBits, st);
            } else {
                MB_CUDA(cudaMemset(st, 0, dirPer * 4));
            }
            MB_CUDA(cudaGetLastError());
            MB_CUDA(cudaMemset(bloom.p, 0, bpp * 32));
            if (hi > lo) {
                bloomKernel<<<static_cast<u32>((hi - lo + BLOCK - 1) / BLOCK), BLOCK>>>(keys, lo, hi, bpp,
                                                                                       static_cast<u32*>(bloom.p));
                MB_CUDA(cudaGetLastError());
            }
            MB_CUDA(cudaMemcpy(startsOut + i * dirPer, st, dirPer * 4, cudaMemcpyDeviceToHost));
            MB_CUDA(cudaMemcpy(bloomOut + i * bpp * 8, bloom.p, bpp * 32, cudaMemcpyDeviceToHost));
        }
        return true;
    }

}  // namespace

extern "C" {

MBGPU_API int mbgpu_open(const char* nameMatch, const uint64_t vramCapBytes, char* info, const int infoLen) {
    g_error.clear();
    if (g_dev.open) {
        mbgpu_close();
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        g_error = "no CUDA device";
        return 1;
    }
    int found = -1;
    cudaDeviceProp prop{};
    for (int d = 0; d < count; ++d) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, d) == cudaSuccess && nameMatch != nullptr && std::strstr(p.name, nameMatch)) {
            found = d;
            prop = p;
            break;
        }
    }
    if (found < 0) {
        g_error = std::string("no CUDA device named like \"") + (nameMatch ? nameMatch : "") + "\"";
        return 1;
    }
    if (cudaSetDevice(found) != cudaSuccess) {
        g_error = "cudaSetDevice failed";
        return 1;
    }
    size_t freeBytes = 0, totalBytes = 0;
    cudaMemGetInfo(&freeBytes, &totalBytes);
    g_dev = Device{};
    g_dev.index = found;
    g_dev.cap = std::min<u64>(vramCapBytes, freeBytes > (256ULL << 20) ? freeBytes - (256ULL << 20) : 0);
    g_dev.open = true;

    // Level buffers: 21 B per node; up to 1/8 of the cap for all of them, at most 8 M nodes per level.
    const u64 perNode = 8 + 8 + 1 + 4;
    g_dev.levelCap = std::min<u64>(8ULL << 20, (g_dev.cap / 8) / (perNode * MBGPU_MAX_DEPTH));
    g_dev.outCap = std::max<u64>(g_dev.levelCap, 1ULL << 20);
    bool ok = g_dev.levelCap >= 1024;
    for (int d = 0; ok && d < MBGPU_MAX_DEPTH; ++d) {
        Nodes& l = g_dev.level[d];
        ok = allocate(reinterpret_cast<void**>(&l.b1), g_dev.levelCap * 8) &&
             allocate(reinterpret_cast<void**>(&l.b2), g_dev.levelCap * 8) &&
             allocate(reinterpret_cast<void**>(&l.prev), g_dev.levelCap) &&
             allocate(reinterpret_cast<void**>(&l.seed), g_dev.levelCap * 4);
    }
    ok = ok && allocate(reinterpret_cast<void**>(&g_dev.ctr), sizeof(Counters)) &&
         allocate(reinterpret_cast<void**>(&g_dev.ob1), g_dev.outCap * 8) &&
         allocate(reinterpret_cast<void**>(&g_dev.ob2), g_dev.outCap * 8) &&
         allocate(reinterpret_cast<void**>(&g_dev.okey), g_dev.outCap * 8) &&
         allocate(reinterpret_cast<void**>(&g_dev.oseed), g_dev.outCap * 4);
    if (!ok) {
        const std::string why = g_error.empty() ? "VRAM cap too small" : g_error;
        mbgpu_close();
        g_error = why;
        return 1;
    }
    const u32 salt[8] = {0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
                         0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U};
    cudaMemcpyToSymbol(c_salt, salt, sizeof(salt));
    if (info != nullptr && infoLen > 0) {
        std::snprintf(info, static_cast<size_t>(infoLen),
                      "%s (CUDA device %d, sm_%d%d, %d SMs), %.1f GB total, %.1f GB free, cap %.1f GB; "
                      "level buffers %llu nodes, %.2f GB in use",
                      prop.name, found, prop.major, prop.minor, prop.multiProcessorCount, totalBytes / 1e9,
                      freeBytes / 1e9, g_dev.cap / 1e9, g_dev.levelCap, g_dev.used / 1e9);
    }
    return 0;
}

MBGPU_API int mbgpu_set_target(const uint64_t goalB1, const uint64_t goalB2, const int limit) {
    g_error.clear();
    if (!g_dev.open) {
        g_error = "mbgpu_open has not been called";
        return 1;
    }
    if (goalB1 & FAT_BOOL_BIT) {
        g_error = "2x2 boards are not supported yet";
        return 1;
    }
    u64 rows[6], cols[6];
    for (u32 i = 0; i < 6; ++i) {
        rows[i] = rowLane(goalB1, goalB2, i);
        cols[i] = colLane(goalB1, goalB2, i);
    }
    // the node kernel's mismatch tables (c_mc, c_mr)
    u32 mc[288], mr[288];
    for (u32 a = 0; a < 6; ++a) {
        for (u32 p = 0; p < 6; ++p) {
            for (u32 v = 0; v < 8; ++v) {
                u32 ec = 0, er = 0;
                for (u32 k = 0; k < 6; ++k) {
                    const u32 q = (p + k) % 6;
                    ec |= (v != ((cols[a] >> (15 - 3 * q)) & 7u) ? 1u : 0u) << (4 * k);
                    er |= (v != ((rows[a] >> (15 - 3 * q)) & 7u) ? 1u : 0u) << (4 * k);
                }
                mc[(a * 6 + p) * 8 + v] = ec;
                mr[(a * 6 + p) * 8 + v] = er;
            }
        }
    }
    if (cudaMemcpyToSymbol(c_mc, mc, sizeof(mc)) != cudaSuccess ||
        cudaMemcpyToSymbol(c_mr, mr, sizeof(mr)) != cudaSuccess) {
        g_error = "copying the mismatch tables to the device failed";
        return 1;
    }
    if (cudaMemcpyToSymbol(c_rowTarget, rows, sizeof(rows)) != cudaSuccess ||
        cudaMemcpyToSymbol(c_colTarget, cols, sizeof(cols)) != cudaSuccess ||
        cudaMemcpyToSymbol(c_limit, &limit, sizeof(limit)) != cudaSuccess) {
        g_error = "copying the target to the device failed";
        return 1;
    }
    g_dev.limit = limit;
    g_dev.targetSet = true;
    return 0;
}

MBGPU_API int mbgpu_set_filter(const uint32_t* bloomWords, const uint64_t blocksPerPartition, const uint32_t pBegin,
                               const uint32_t pEnd) {
    g_error.clear();
    if (!g_dev.open) {
        g_error = "mbgpu_open has not been called";
        return 1;
    }
    if (pBegin >= pEnd || pEnd > 256) {
        g_error = "bad partition range";
        return 1;
    }
    release(g_dev.bloom, g_dev.bloomWords * 4);
    g_dev.bloom = nullptr;
    g_dev.bloomWords = 0;
    g_dev.pBegin = pBegin;
    g_dev.pEnd = pEnd;
    g_dev.blocksPerPartition = blocksPerPartition;
    if (bloomWords == nullptr) {
        return 0;
    }
    const u64 words = static_cast<u64>(pEnd - pBegin) * blocksPerPartition * 8;
    if (!allocate(reinterpret_cast<void**>(&g_dev.bloom), words * 4)) {
        return 1;
    }
    g_dev.bloomWords = words;
    if (cudaMemcpy(g_dev.bloom, bloomWords, words * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        g_error = "copying the Bloom blocks failed";
        return 1;
    }
    return 0;
}

MBGPU_API int mbgpu_stream(const uint64_t* states, const uint8_t* prevMove, const uint64_t count, const int depth,
                           const mbgpu_sink_fn sink, void* ctx, mbgpu_stats* stats) {
    g_error.clear();
    mbgpu_stats local{};
    Run run;
    run.depth = depth;
    run.sink = sink;
    run.ctx = ctx;
    run.stats = stats != nullptr ? stats : &local;
    run.lo.mode = LEAF_FILTER;
    run.lo.bloom = g_dev.bloom;
    run.lo.blocksPerPartition = g_dev.blocksPerPartition;
    run.lo.pBegin = g_dev.pBegin;
    run.lo.pEnd = g_dev.pEnd;
    run.lo.ob1 = g_dev.ob1;
    run.lo.ob2 = g_dev.ob2;
    run.lo.okey = g_dev.okey;
    run.lo.oseed = g_dev.oseed;
    run.lo.oCap = g_dev.outCap;
    return streamAll(run, states, prevMove, count) ? 0 : 1;
}

MBGPU_API int mbgpu_stream_checksums(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                                     const int depth, uint64_t* leafCount, uint64_t* keySum, uint64_t* keyXor,
                                     mbgpu_stats* stats) {
    g_error.clear();
    mbgpu_stats local{};
    Run run;
    run.depth = depth;
    run.stats = stats != nullptr ? stats : &local;
    return checksumRun(run, states, prevMove, count, leafCount, keySum, keyXor, nullptr) ? 0 : 1;
}

MBGPU_API int mbgpu_audit_stream(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                                 const int depth, uint64_t* wantedCount, uint64_t* wantedSum, uint64_t* wantedXor,
                                 mbgpu_audit_stats* stats) {
    g_error.clear();
    if (g_dev.limit < 0) {
        g_error = "the audit needs a limit (mbgpu_set_target)";
        return 1;
    }
    mbgpu_stats st{};
    Run run;
    run.depth = depth;
    run.prune = 0;
    run.audit = 1;
    run.stats = &st;
    Counters c{};
    if (!checksumRun(run, states, prevMove, count, wantedCount, wantedSum, wantedXor, &c)) {
        return 1;
    }
    if (stats != nullptr) {
        stats->nodes = 0;
        for (int d = 1; d < depth; ++d) {
            stats->nodes += st.nodes[d];
        }
        stats->nodes += c.leaves;
        stats->edges = c.edges;
        stats->violations = c.violations;
        stats->leaves = c.leaves;
        stats->overLimit = c.overLimit;
        stats->seconds = st.seconds;
    }
    return 0;
}

MBGPU_API int mbgpu_bounds(const uint64_t* states, const uint64_t count, int32_t* h) {
    g_error.clear();
    if (!g_dev.targetSet) {
        g_error = "mbgpu_set_target has not been called";
        return 1;
    }
    if (count == 0) {
        return 0;
    }
    Scratch s, out;
    if (!s.get(count * 16) || !out.get(count * 4)) {
        return 1;
    }
    if (cudaMemcpy(s.p, states, count * 16, cudaMemcpyHostToDevice) != cudaSuccess) {
        g_error = "copying states failed";
        return 1;
    }
    boundKernel<<<static_cast<u32>((count + BLOCK - 1) / BLOCK), BLOCK>>>(static_cast<const u64*>(s.p), count,
                                                                         static_cast<int*>(out.p));
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(h, out.p, count * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        g_error = "bound kernel failed";
        return 1;
    }
    return 0;
}

MBGPU_API int mbgpu_child_counts(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                                 uint64_t partitionCounts[256]) {
    g_error.clear();
    Scratch counts;
    if (!counts.get(256 * 8) || cudaMemset(counts.p, 0, 256 * 8) != cudaSuccess) {
        return 1;
    }
    mbgpu_stats st{};
    Run run;
    run.depth = 1;
    run.prune = 0;
    run.stats = &st;
    run.lo.mode = LEAF_COUNTS;
    run.lo.partCounts = static_cast<unsigned long long*>(counts.p);
    if (!streamAll(run, states, prevMove, count)) {
        return 1;
    }
    if (cudaMemcpy(partitionCounts, counts.p, 256 * 8, cudaMemcpyDeviceToHost) != cudaSuccess) {
        g_error = "copying partition counts failed";
        return 1;
    }
    return 0;
}

MBGPU_API int mbgpu_child_keys(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                               const uint32_t p0, const uint32_t p1, uint64_t* out, const uint64_t outCap,
                               uint64_t* outCount) {
    g_error.clear();
    if (!g_dev.open) {
        g_error = "mbgpu_open has not been called";
        return 1;
    }
    if (p0 >= p1 || p1 > 256) {
        g_error = "bad partition range";
        return 1;
    }
    return childKeys(states, prevMove, count, p0, p1, out, outCap, outCount) ? 0 : 1;
}

MBGPU_API int mbgpu_match_children(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                                   const uint64_t* sortedKeys, const uint64_t keyCount, const mbgpu_sink_fn sink,
                                   void* ctx, mbgpu_stats* stats) {
    g_error.clear();
    if (!g_dev.open) {
        g_error = "mbgpu_open has not been called";
        return 1;
    }
    if (keyCount == 0 || count == 0) {
        return 0;
    }
    Scratch keys;
    if (!keys.get(keyCount * 8) ||
        cudaMemcpy(keys.p, sortedKeys, keyCount * 8, cudaMemcpyHostToDevice) != cudaSuccess) {
        if (g_error.empty()) {
            g_error = "copying the keys to match failed";
        }
        return 1;
    }
    mbgpu_stats local{};
    Run run;
    run.depth = 1;
    run.prune = 0;
    run.sink = sink;
    run.ctx = ctx;
    run.stats = stats != nullptr ? stats : &local;
    run.lo.mode = LEAF_FILTER;
    run.lo.match = static_cast<const u64*>(keys.p);
    run.lo.matchCount = keyCount;
    run.lo.pBegin = 0;
    run.lo.pEnd = 256;
    run.lo.ob1 = g_dev.ob1;
    run.lo.ob2 = g_dev.ob2;
    run.lo.okey = g_dev.okey;
    run.lo.oseed = g_dev.oseed;
    run.lo.oCap = g_dev.outCap;
    return streamAll(run, states, prevMove, count) ? 0 : 1;
}

MBGPU_API uint64_t mbgpu_child_key_capacity(void) {
    return g_dev.open ? keyCapacity() : 0;
}

MBGPU_API int mbgpu_build_index(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                                const uint32_t p0, const uint32_t p1, const uint32_t bucketBits,
                                const uint64_t bloomBlocksPerPartition, uint16_t* fpsOut, const uint64_t fpsCap,
                                uint32_t* startsOut, uint32_t* bloomOut, uint64_t* uniqueOut) {
    g_error.clear();
    if (!g_dev.open) {
        g_error = "mbgpu_open has not been called";
        return 1;
    }
    if (p0 >= p1 || p1 > 256 || bucketBits > 32 || bloomBlocksPerPartition == 0) {
        g_error = "bad index layout";
        return 1;
    }
    return buildIndex(states, prevMove, count, p0, p1, bucketBits, bloomBlocksPerPartition,
                      reinterpret_cast<unsigned short*>(fpsOut), fpsCap, startsOut, bloomOut, uniqueOut)
               ? 0
               : 1;
}

MBGPU_API const char* mbgpu_last_error(void) {
    return g_error.c_str();
}

MBGPU_API void mbgpu_close(void) {
    if (!g_dev.open) {
        return;
    }
    for (Nodes& l : g_dev.level) {
        cudaFree(l.b1);
        cudaFree(l.b2);
        cudaFree(l.prev);
        cudaFree(l.seed);
    }
    cudaFree(g_dev.ctr);
    cudaFree(g_dev.ob1);
    cudaFree(g_dev.ob2);
    cudaFree(g_dev.okey);
    cudaFree(g_dev.oseed);
    cudaFree(g_dev.bloom);
    g_dev = Device{};
}

}  // extern "C"
