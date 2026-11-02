// src/gpu/mbgpu.cu
//
// The GPU left stream (interface and contract: src/gpu/mbgpu.h). Normal boards
// only so far.
//
// Layout and rules are the CPU's, re-implemented without its headers (the
// solver is built with MinGW, this file with nvcc + MSVC):
// - B1B2: rows 0-2 in b1, rows 3-5 in b2; cell (row r, column c) of a half at
//   bit 51 - 18 * (r % 3) - 3 * c (lane_rotations.hpp). A lane read as 18 bits
//   has its first cell in the top 3 bits; shifting it forward by n cells is a
//   right rotation by 3n.
// - Moves are coded dir * 30 + lane * 5 + (amount - 1), dir 0 = rows, 1 = columns
//   (perm_stream_detail::decodePrevMove).
// - stream_lanes_dfs (ASCENDING, CHECK_CROSS, CHECK_SIM): after a move on
//   (dir, lane), same-direction moves only on later lanes; a lane stops at the
//   first amount that gives the board back (its period); after a column move,
//   a row move that commutes with it (Board::doActISColMatchBatched) is skipped.
// - Rotation lane bound (lane_bound.hpp, Kind::Rotation): h = max over rows of
//   d(row) + max over columns of d(column), d = fewest cells that differ from the
//   target lane over its 6 rotations; a child with r moves still below it is
//   dropped when h > limit + r, the leaves included (bounds::LanePruneLeaves).
//
// Work is level by level: one thread per (node, lane) computes the lane's five
// shifts and appends the children it keeps to the next level's buffer (one
// atomic per warp). A level is processed in chunks small enough that their
// children always fit the next buffer, and every chunk's subtree is finished
// before the next chunk starts, so each level needs one fixed buffer and every
// kernel launch is short (WDDM resets a GPU whose kernel runs for ~2 s).

#define MBGPU_BUILD_DLL
#include "src/gpu/mbgpu.h"

#include <cuda_runtime.h>

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
    constexpr u32 BLOCK = 256;

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
        const u32 hi = static_cast<u32>(b1 >> (15 - 3 * x)), lo = static_cast<u32>(b2 >> (15 - 3 * x));
        // rows 0-2 sit 18 bits apart in each half: bits 36, 18 and 0 above the column's base
        return ((static_cast<u32>(b1 >> (51 - 3 * x)) & 7u) << 15) | ((static_cast<u32>(b1 >> (33 - 3 * x)) & 7u) << 12) |
               ((hi & 7u) << 9) | ((static_cast<u32>(b2 >> (51 - 3 * x)) & 7u) << 6) |
               ((static_cast<u32>(b2 >> (33 - 3 * x)) & 7u) << 3) | (lo & 7u);
    }

    // The row (or column) half of h, stopping as soon as it exceeds `stopAbove`
    // (the caller only needs to know that it does).
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

    struct LeafOut {
        int checksums;                 // 1: per-seed checksums, 0: filter + output
        unsigned long long* cnt;       // per seed (checksums)
        unsigned long long* sum;
        unsigned long long* xr;
        const u32* bloom;              // nullptr: every leaf in range passes
        u64 blocksPerPartition;
        u32 pBegin, pEnd;
        u64* ob1;                      // leaves that passed
        u64* ob2;
        u64* okey;
        u32* oseed;
        u64 oCap;
    };

    struct Counters {
        unsigned long long next;       // nodes appended to the next level
        unsigned long long leaves;
        unsigned long long inRange;
        unsigned long long passed;     // leaves written to the output buffer
        int overflow;
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

    // One thread per (node, lane). rem: moves still below a child. leaf != 0: the
    // children are leaves (handled by `lo`), else they are appended to `out`.
    __global__ void expandKernel(const Nodes in, const u64 n, const Nodes out, const u64 outCap, const int rem,
                                 const int leaf, const LeafOut lo, Counters* ctr) {
        const u64 t = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
        const u64 i = t / THREADS_PER_NODE;

        u64 cb1[5], cb2[5];
        u32 amount[5];
        u32 keep = 0;
        u32 codeBase = 0;
        u32 seed = 0;

        if (i < n) {
            const u32 dl = static_cast<u32>(t % THREADS_PER_NODE);
            const u32 dir = dl / 6, lane = dl % 6;
            const u64 b1 = in.b1[i], b2 = in.b2[i];
            const u32 prev = in.prev[i];
            seed = in.seed[i];
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
                int fixedHalf = 0;
                bool allFar = false;
                if (limit >= 0) {
                    fixedHalf = dir == 0 ? rowHalf(b1, b2) : colHalf(b1, b2);
                    allFar = fixedHalf > limit + rem;
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
                        if (limit >= 0) {
                            const int room = limit + rem - fixedHalf;   // the other half may be at most this
                            if ((dir == 0 ? colHalf(o1, o2, room) : rowHalf(o1, o2, room)) > room) {
                                continue;
                            }
                        }
                        cb1[keep] = o1;
                        cb2[keep] = o2;
                        amount[keep] = k;
                        ++keep;
                    }
                }
            }
        }

        if (!leaf) {
            u32 total = 0;
            const u32 offset = warpPrefix(keep, total);
            unsigned long long base = 0;
            if ((threadIdx.x & 31) == 0 && total != 0) {
                base = atomicAdd(&ctr->next, static_cast<unsigned long long>(total));
            }
            base = __shfl_sync(0xFFFFFFFFu, base, 0);
            for (u32 j = 0; j < keep; ++j) {
                const u64 at = base + offset + j;
                if (at >= outCap) {
                    ctr->overflow = 1;
                    break;
                }
                out.b1[at] = cb1[j];
                out.b2[at] = cb2[j];
                out.prev[at] = static_cast<u8>(codeBase + amount[j]);
                out.seed[at] = seed;
            }
            return;
        }

        u32 inRange = 0;
        for (u32 j = 0; j < keep; ++j) {
            const u64 key = mixKey(stateHash(cb1[j], cb2[j]));
            if (lo.checksums) {
                atomicAdd(&lo.cnt[seed], 1ULL);
                atomicAdd(&lo.sum[seed], static_cast<unsigned long long>(key));
                atomicXor(&lo.xr[seed], static_cast<unsigned long long>(key));
                continue;
            }
            const u32 p = static_cast<u32>(key >> 56);
            if (p < lo.pBegin || p >= lo.pEnd) {
                continue;
            }
            ++inRange;
            if (lo.bloom != nullptr && !bloomPasses(lo, key)) {
                continue;
            }
            const unsigned long long at = atomicAdd(&ctr->passed, 1ULL);
            if (at >= lo.oCap) {
                ctr->overflow = 1;
                continue;
            }
            lo.ob1[at] = cb1[j];
            lo.ob2[at] = cb2[j];
            lo.okey[at] = key;
            lo.oseed[at] = seed;
        }
        u32 leavesTotal = 0, rangeTotal = 0;
        warpPrefix(keep, leavesTotal);
        warpPrefix(inRange, rangeTotal);
        if ((threadIdx.x & 31) == 0) {
            if (leavesTotal != 0) {
                atomicAdd(&ctr->leaves, static_cast<unsigned long long>(leavesTotal));
            }
            if (rangeTotal != 0) {
                atomicAdd(&ctr->inRange, static_cast<unsigned long long>(rangeTotal));
            }
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

    struct Run {
        int depth = 0;
        LeafOut lo{};
        mbgpu_sink_fn sink = nullptr;
        void* ctx = nullptr;
        mbgpu_stats* stats = nullptr;
        std::vector<u64> hb1, hb2, hkey;
        std::vector<u32> hseed;
    };

    bool flushOutput(Run& run) {
        Counters c{};
        MB_CUDA(cudaMemcpy(&c, g_dev.ctr, sizeof(Counters), cudaMemcpyDeviceToHost));
        if (c.overflow) {
            return fail("output buffer overflow");
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
        return true;
    }

    Nodes offsetNodes(const Nodes& a, const u64 off) {
        return Nodes{a.b1 + off, a.b2 + off, a.prev + off, a.seed + off};
    }

    // Expands level `d` (n nodes in g_dev.level[d]) down to the leaves.
    bool processLevel(Run& run, const int d, const u64 n) {
        const u64 chunk = std::max<u64>(1, g_dev.levelCap / MAX_CHILDREN);
        const int childDepth = d + 1;
        const int rem = run.depth - childDepth;
        const bool leaf = childDepth == run.depth;
        for (u64 off = 0; off < n; off += chunk) {
            const u64 m = std::min(chunk, n - off);
            if (leaf && !run.lo.checksums) {
                // room for every child of this chunk, else hand the buffer over first
                Counters c{};
                MB_CUDA(cudaMemcpy(&c, g_dev.ctr, sizeof(Counters), cudaMemcpyDeviceToHost));
                if (c.passed + m * MAX_CHILDREN > g_dev.outCap && !flushOutput(run)) {
                    return false;
                }
            }
            if (!leaf) {
                const unsigned long long zero = 0;
                MB_CUDA(cudaMemcpy(&g_dev.ctr->next, &zero, sizeof(zero), cudaMemcpyHostToDevice));
            }
            const u64 threads = m * THREADS_PER_NODE;
            const u32 blocks = static_cast<u32>((threads + BLOCK - 1) / BLOCK);
            expandKernel<<<blocks, BLOCK>>>(offsetNodes(g_dev.level[d], off), m,
                                            leaf ? Nodes{} : g_dev.level[childDepth], g_dev.levelCap, rem,
                                            leaf ? 1 : 0, run.lo, g_dev.ctr);
            MB_CUDA(cudaGetLastError());
            run.stats->launches++;
            if (!leaf) {
                Counters c{};
                MB_CUDA(cudaMemcpy(&c, g_dev.ctr, sizeof(Counters), cudaMemcpyDeviceToHost));
                if (c.overflow) {
                    return fail("level buffer overflow");
                }
                run.stats->nodes[childDepth] += c.next;
                if (c.next != 0 && !processLevel(run, childDepth, c.next)) {
                    return false;
                }
            }
        }
        return true;
    }

    bool streamAll(Run& run, const u64* states, const u8* prev, const u64 count) {
        if (!g_dev.open) {
            return fail("mbgpu_open has not been called");
        }
        if (!g_dev.targetSet) {
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
        if (!run.lo.checksums && !flushOutput(run)) {
            return false;
        }
        Counters c{};
        MB_CUDA(cudaMemcpy(&c, g_dev.ctr, sizeof(Counters), cudaMemcpyDeviceToHost));
        if (c.overflow) {
            return fail("buffer overflow");
        }
        run.stats->leaves += c.leaves;
        run.stats->inRange += c.inRange;
        run.stats->seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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
    run.lo.checksums = 0;
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
    run.lo.checksums = 1;
    run.lo.pBegin = 0;
    run.lo.pEnd = 256;
    const u64 bytes = count * 8;
    unsigned long long *cnt = nullptr, *sum = nullptr, *xr = nullptr;
    bool ok = allocate(reinterpret_cast<void**>(&cnt), bytes) && allocate(reinterpret_cast<void**>(&sum), bytes) &&
              allocate(reinterpret_cast<void**>(&xr), bytes);
    ok = ok && cudaMemset(cnt, 0, bytes) == cudaSuccess && cudaMemset(sum, 0, bytes) == cudaSuccess &&
         cudaMemset(xr, 0, bytes) == cudaSuccess;
    if (ok) {
        run.lo.cnt = cnt;
        run.lo.sum = sum;
        run.lo.xr = xr;
        ok = streamAll(run, states, prevMove, count);
    }
    ok = ok && cudaMemcpy(leafCount, cnt, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(keySum, sum, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(keyXor, xr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    release(cnt, cnt ? bytes : 0);
    release(sum, sum ? bytes : 0);
    release(xr, xr ? bytes : 0);
    if (!ok && g_error.empty()) {
        g_error = "checksum buffers failed";
    }
    return ok ? 0 : 1;
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
