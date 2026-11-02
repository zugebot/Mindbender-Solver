#pragma once
// src/gpu/mbgpu.h
//
// C interface of the GPU left stream (mbgpu.dll, built with nvcc + MSVC by
// src/gpu/build.cmd; the solver itself is built with MinGW and loads it at run
// time, so only plain C types cross the boundary). See src/gpu/README.md.
//
// What it does: from a list of left seeds (states + each seed's last move, as
// ssd::buildLastMoves gives them), it streams every seed `depth` more moves with
// the same canonical rules as perm_stream_detail::stream_lanes_dfs (normal
// boards) and the rotation lane bound at every level (the "both" mode of
// bounds::LanePruneLeaves), and for every leaf either
//   - accumulates per-seed checksums (count, sum and xor of the leaf keys), to
//     compare with the CPU stream (src/gpu/test_gpu_stream.cu), or
//   - tests the key against the goal side's Bloom blocks (the same split-block
//     filter as ssd::FingerprintRange) and hands the leaves that pass to a
//     callback; the caller finishes them on the CPU (buckets + fingerprints).
// A leaf's key is mixKey(StateHash(state)), as in the solver.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MBGPU_BUILD_DLL)
#define MBGPU_API __declspec(dllexport)
#else
#define MBGPU_API
#endif

#define MBGPU_NO_PREV_MOVE 255u   // a seed with no previous move: every first move allowed
#define MBGPU_MAX_DEPTH 6

typedef struct mbgpu_stats {
    uint64_t nodes[MBGPU_MAX_DEPTH + 1];   // nodes kept at each level (level 0 = seeds)
    uint64_t leaves;                        // leaves kept (after the bound)
    uint64_t inRange;                       // leaves whose key is in the filter's partition range
    uint64_t passed;                        // leaves that passed the Bloom filter (handed to the sink)
    uint64_t launches;                      // kernel launches
    double seconds;                         // wall time of the call
} mbgpu_stats;

// Called with batches of leaves that passed the filter (any order).
typedef void (*mbgpu_sink_fn)(void* ctx, const uint64_t* b1, const uint64_t* b2, const uint64_t* keys,
                              const uint32_t* seedIndex, uint64_t count);

// Opens the first CUDA device whose name contains `nameMatch` (e.g. "5060"); refuses
// to run on any other device. At most `vramCapBytes` of its memory is used. `info`
// receives a one-line description (device, memory, buffers). Returns 0 on success.
MBGPU_API int mbgpu_open(const char* nameMatch, uint64_t vramCapBytes, char* info, int infoLen);

// The bound's target (the goal-side root) and limit (the goal side's depth): a
// node with r moves still below it is dropped when h > limit + r. limit < 0: no
// pruning (every canonical sequence is streamed).
MBGPU_API int mbgpu_set_target(uint64_t goalB1, uint64_t goalB2, int limit);

// The Bloom blocks of partitions [pBegin, pEnd), laid out as FingerprintRange::bloom
// ((pEnd - pBegin) * blocksPerPartition blocks of 8 u32), copied to the device.
// bloomWords == NULL: no filter, every leaf in range passes.
MBGPU_API int mbgpu_set_filter(const uint32_t* bloomWords, uint64_t blocksPerPartition, uint32_t pBegin,
                               uint32_t pEnd);

// Seeds: `states` holds 2 x u64 per seed (b1, b2: the solver's B1B2 array as is);
// `prevMove` one move code per seed, or NULL for none (every first move allowed).
// Streams seeds [0, count) `depth` (1..MBGPU_MAX_DEPTH) moves; leaves that pass go to `sink`.
MBGPU_API int mbgpu_stream(const uint64_t* states, const uint8_t* prevMove, uint64_t count, int depth,
                           mbgpu_sink_fn sink, void* ctx, mbgpu_stats* stats);

// The same stream, but per seed: leaf count, wrapping sum and xor of the leaf keys
// (every leaf, no filter). For checking against the CPU stream.
MBGPU_API int mbgpu_stream_checksums(const uint64_t* states, const uint8_t* prevMove, uint64_t count, int depth,
                                     uint64_t* leafCount, uint64_t* keySum, uint64_t* keyXor, mbgpu_stats* stats);

// ---- bound audit (the GPU's own bound, before a run trusts it) ----

typedef struct mbgpu_audit_stats {
    uint64_t nodes;          // every node of the full (unpruned) streams, leaves included
    uint64_t edges;          // node -> child edges checked
    uint64_t violations;     // edges where h changed by more than 1
    uint64_t leaves;         // leaves of the full streams
    uint64_t overLimit;      // of them, leaves with h > limit
    double seconds;
} mbgpu_audit_stats;

// The full stream of each seed (no pruning) with h checked on every edge (|dh| <= 1), and
// per seed the leaf count and key sum / xor of the "wanted" leaves (h <= limit), to compare
// with mbgpu_stream_checksums of the same seeds (which prunes): they must be identical.
MBGPU_API int mbgpu_audit_stream(const uint64_t* states, const uint8_t* prevMove, uint64_t count, int depth,
                                 uint64_t* wantedCount, uint64_t* wantedSum, uint64_t* wantedXor,
                                 mbgpu_audit_stats* stats);

// h of each state as the GPU computes it (to compare with the CPU's reference definition).
MBGPU_API int mbgpu_bounds(const uint64_t* states, uint64_t count, int32_t* h);

// ---- goal-side index build ----

// Number of one-move children (the canonical first moves after each state's last move, no
// pruning) whose key falls in each of the 256 partitions (key >> 56).
MBGPU_API int mbgpu_child_counts(const uint64_t* states, const uint8_t* prevMove, uint64_t count,
                                 uint64_t partitionCounts[256]);

// The children's keys in partitions [p0, p1), sorted and without duplicates, into `out`
// (capacity `outCap`); `outCount` receives how many. Fails if the raw keys of the range do
// not fit the device (mbgpu_child_key_capacity tells how many do).
MBGPU_API int mbgpu_child_keys(const uint64_t* states, const uint8_t* prevMove, uint64_t count, uint32_t p0,
                               uint32_t p1, uint64_t* out, uint64_t outCap, uint64_t* outCount);

// How many raw keys one mbgpu_child_keys / mbgpu_build_index call can sort with the VRAM left now.
MBGPU_API uint64_t mbgpu_child_key_capacity(void);

// The fingerprint index of partitions [p0, p1) built from the children's sorted unique keys,
// in ssd::FingerprintRange's layout (no_table_d6.hpp builds the same on the CPU):
//   fpsOut:    u16 low key bits of every key, partition after partition (capacity fpsCap)
//   startsOut: per partition 2^bucketBits + 1 u32 offsets (first key of each bucket, relative to
//              the partition's first key; the last one = its key count); bucket = key bits below
//              the top 8, `bucketBits` of them
//   bloomOut:  per partition bloomBlocksPerPartition blocks of 8 u32 (split-block Bloom filter,
//              10 bits per state), overwritten
//   uniqueOut: per partition the number of unique keys
MBGPU_API int mbgpu_build_index(const uint64_t* states, const uint8_t* prevMove, uint64_t count, uint32_t p0,
                                uint32_t p1, uint32_t bucketBits, uint64_t bloomBlocksPerPartition,
                                uint16_t* fpsOut, uint64_t fpsCap, uint32_t* startsOut, uint32_t* bloomOut,
                                uint64_t* uniqueOut);

// The last error as text ("" if none).
MBGPU_API const char* mbgpu_last_error(void);

MBGPU_API void mbgpu_close(void);

#ifdef __cplusplus
}
#endif
