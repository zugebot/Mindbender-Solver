// src/gpu/vulkan/shaders/common.glsl
//
// Shared by the Vulkan kernels: the board layout, hashes, the bound and the leaf tests,
// line for line the CUDA module's (src/gpu/mbgpu.cu, which documents the rules). Buffers
// are reached by device address (GL_EXT_buffer_reference), so there are no descriptor sets:
// every kernel gets its addresses in push constants or in the Globals block.
//
// Differences from the CUDA kernels, none of them in what is computed:
// - a node's previous move and its seed index share one u32 (prev << 24 | seed within the
//   launch's chunk of seeds, < 2^23), so no 8-bit storage is needed;
// - device counters are 32-bit and set per launch (the host adds them up), and the per-seed
//   checksums and partition counts are 64-bit sums made of two 32-bit atomics with a carry,
//   so no 64-bit atomics are needed;
// - subgroup operations stand in for warp shuffles, for any subgroup size that divides 256.

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require

#define BLOCK 256u
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 8) buffer U64Ref { uint64_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32Ref { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer U128Ref { uvec4 v[]; };

// Per-launch counters (mbvk.cpp: struct Counters).
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Counters {
    uint next;         // nodes appended to the next level
    uint leaves;
    uint inRange;
    uint passed;       // leaves (or keys) in the output buffer (kept across launches)
    uint edges;        // audit
    uint violations;
    uint overLimit;
    uint overflow;
};

#define LEAF_FILTER 0
#define LEAF_CHECKSUM 1
#define LEAF_KEYS 2
#define LEAF_COUNTS 3

// Set per call by the host (mbvk.cpp: struct Globals, the same std430 layout).
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer Globals {
    uint64_t rowTarget[6];
    uint64_t colTarget[6];
    uint64_t cnt;          // LEAF_CHECKSUM: per seed, 2 x u32 each (low, high)
    uint64_t sum;
    uint64_t xr;
    uint64_t bloom;        // LEAF_FILTER: the first Bloom buffer; 0 = every leaf in range passes
    uint64_t match;        // LEAF_FILTER: if set, only leaves whose key is in this sorted list
    uint64_t matchCount;
    uint64_t blocksPerPartition;
    uint64_t ob1;          // LEAF_FILTER output
    uint64_t ob2;
    uint64_t okey;         // LEAF_FILTER and LEAF_KEYS output
    uint64_t oseed;
    uint64_t partCounts;   // LEAF_COUNTS: 256 x (2 x u32)
    uint64_t bloomBuf[8];  // the Bloom blocks, `bloomPartsPerBuf` partitions per buffer (a driver
                           // may cap one buffer at 4 GB, Mesa RADV does)
    int limit;             // < 0: no pruning
    int mode;
    uint pBegin;
    uint pEnd;
    uint oCap;
    uint bloomPartsPerBuf;
    uint keyMode;          // 0: state keys; 1, 2: class keys (mbgpu.h: mbgpu_set_key_mode)
    uint pad2;
    uint mc[288];          // packed mismatch counts (mbgpu.cu: c_mc, c_mr)
    uint mr[288];
    uint64_t minRot;       // key modes 1-2: class_key::minRotation of every 18-bit lane code (u32 each)
};

const uint64_t LANE_BITS = 0x3FFFFul;
const uint NO_PREV = 255u;
const uint THREADS_PER_NODE = 12u;
const uint64_t FAT_X_FIELD = 7ul << 61;
const uint64_t FAT_Y_FIELD = 7ul << 58;
const uint SALT[8] = uint[8](0x47b6137bu, 0x44974d91u, 0x8824ad5bu, 0xa2b7289du,
                             0x705495c7u, 0x2df1424bu, 0x9efc4947u, 0x5c6bfb31u);

// The workgroup's index in a 2D dispatch (the host splits large launches over y).
uint groupLinear() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

uint threadLinear() {
    return groupLinear() * BLOCK + gl_LocalInvocationID.x;
}

// ---------------- board helpers ----------------

uint64_t rotateLane(uint64_t lane, uint cells) {
    uint s = 3u * cells;
    return ((lane >> s) | (lane << (18u - s))) & LANE_BITS;
}

uint64_t rowLane(uint64_t b1, uint64_t b2, uint r) {
    return ((r < 3u ? b1 : b2) >> (36u - 18u * (r % 3u))) & LANE_BITS;
}

uint64_t colLane(uint64_t b1, uint64_t b2, uint x) {
    uint64_t lane = 0ul;
    for (uint r = 0u; r < 6u; ++r) {
        uint64_t h = r < 3u ? b1 : b2;
        lane |= ((h >> (51u - 18u * (r % 3u) - 3u * x)) & 7ul) << (15u - 3u * r);
    }
    return lane;
}

uint colorAt(uint64_t b1, uint64_t b2, uint x, uint y) {
    return uint(((y < 3u ? b1 : b2) >> (51u - 18u * (y % 3u) - 3u * x)) & 7ul);
}

uint64_t columnMask(uint x) {
    return (7ul << (51u - 3u * x)) | (7ul << (33u - 3u * x)) | (7ul << (15u - 3u * x));
}

// the board with lane (dir, lane) shifted by n (1..5); L = that lane
void shiftLane(uint64_t b1, uint64_t b2, uint dir, uint lane, uint64_t L, uint n, out uint64_t o1, out uint64_t o2) {
    uint64_t moved = rotateLane(L, n);
    if (dir == 0u) {
        uint sh = 36u - 18u * (lane % 3u);
        if (lane < 3u) {
            o1 = (b1 & ~(LANE_BITS << sh)) | (moved << sh);
            o2 = b2;
        } else {
            o1 = b1;
            o2 = (b2 & ~(LANE_BITS << sh)) | (moved << sh);
        }
    } else {
        uint64_t m = columnMask(lane);
        uint64_t n1 = b1 & ~m, n2 = b2 & ~m;
        for (uint r = 0u; r < 3u; ++r) {
            n1 |= ((moved >> (15u - 3u * r)) & 7ul) << (51u - 18u * r - 3u * lane);
            n2 |= ((moved >> (6u - 3u * r)) & 7ul) << (51u - 18u * r - 3u * lane);
        }
        o1 = n1;
        o2 = n2;
    }
}

uint64_t stateHash(uint64_t x, uint64_t y) {
    return x ^ ((y << 1) | (y >> 63)) ^ (x >> 17) ^ (y << 13);
}

uint64_t mixKey(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ul;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebul;
    x ^= x >> 31;
    return x;
}

// ---- class keys (src/solver/ssd/class_key.hpp, line for line; the --block join) ----

const uint64_t CK_COLUMN_SALT = 0x9E3779B97F4A7C15ul;
const uint64_t CK_PARENT_FLAG = 1ul << 63;   // b2 bit 63: a survivor that stands for its leaves

uint64_t ckRotate(uint64_t v, uint cells) {
    uint s = 3u * (cells % 6u);
    return s == 0u ? v : (((v << s) | (v >> (18u - s))) & 0x3FFFFul);
}

uint64_t ckRowCode(uint64_t b1, uint64_t b2, uint y) {
    return ((y < 3u ? b1 : b2) >> (36u - 18u * (y % 3u))) & 0x3FFFFul;
}

uint64_t ckColCode(uint64_t b1, uint64_t b2, uint x) {
    uint64_t v = 0ul;
    for (uint y = 0u; y < 6u; ++y) {
        v = (v << 3) | (((y < 3u ? b1 : b2) >> (51u - 18u * (y % 3u) - 3u * x)) & 7ul);
    }
    return v;
}

uint64_t classKey(Globals g, uint64_t b1, uint64_t b2, bool cols) {
    bool f = ((b1 >> 57) & 1ul) != 0ul;
    uint px = f ? uint(b1 >> 61) & 7u : 0u;
    uint py = f ? uint(b1 >> 58) & 7u : 0u;
    U32Ref table = U32Ref(g.minRot);
    uint64_t code[6];
    for (uint i = 0u; i < 6u; ++i) {
        uint64_t raw = cols ? ckColCode(b1, b2, i) : ckRowCode(b1, b2, i);
        bool pair = f && (cols ? (i == px || i == px + 1u) : (i == py || i == py + 1u));
        code[i] = pair ? ckRotate(raw, cols ? py : px) : uint64_t(table.v[uint(raw)]);
    }
    uint64_t lo = code[0] | (code[1] << 18) | (code[2] << 36);
    uint64_t hi = code[3] | (code[4] << 18) | (code[5] << 36);
    hi |= uint64_t(f ? 1u + (cols ? px : py) : 0u) << 56;
    uint64_t h = lo ^ ((hi << 1) | (hi >> 63)) ^ (lo >> 17) ^ (hi << 13);
    return mixKey(h ^ (cols ? CK_COLUMN_SALT : 0ul));
}

// ---------------- bound from its definition (the audit kernel and mbgpu_bounds) ----------------

int laneDistance(uint lane, uint target) {
    int best = 6;
    for (uint k = 0u; k < 6u; ++k) {
        uint s = 3u * k;
        uint x = (((lane >> s) | (lane << (18u - s))) & 0x3FFFFu) ^ target;
        x = (x | (x >> 1) | (x >> 2)) & 0x9249u;
        best = min(best, bitCount(x));
    }
    return best;
}

uint colLane32(uint64_t b1, uint64_t b2, uint x) {
    return ((uint(b1 >> (51u - 3u * x)) & 7u) << 15) | ((uint(b1 >> (33u - 3u * x)) & 7u) << 12) |
           ((uint(b1 >> (15u - 3u * x)) & 7u) << 9) | ((uint(b2 >> (51u - 3u * x)) & 7u) << 6) |
           ((uint(b2 >> (33u - 3u * x)) & 7u) << 3) | (uint(b2 >> (15u - 3u * x)) & 7u);
}

int rowHalf(Globals g, uint64_t b1, uint64_t b2) {
    int r = 0;
    for (uint i = 0u; i < 6u; ++i) {
        r = max(r, laneDistance(uint(rowLane(b1, b2, i)), uint(g.rowTarget[i])));
    }
    return r;
}

int colHalf(Globals g, uint64_t b1, uint64_t b2) {
    int c = 0;
    for (uint i = 0u; i < 6u; ++i) {
        c = max(c, laneDistance(colLane32(b1, b2, i), uint(g.colTarget[i])));
    }
    return c;
}

// ---------------- 64-bit sums from 32-bit atomics ----------------

// words[2i] (low), words[2i + 1] (high) += v, mod 2^64: each add's carry out of the low word
// goes to the high word, so the pair ends as the exact wrapping sum
void atomicAdd64(uint64_t words, uint i, uint64_t v) {
    U32Ref r = U32Ref(words);
    uint lo = uint(v), hi = uint(v >> 32);
    uint old = atomicAdd(r.v[2u * i], lo);
    uint carry = old + lo < old ? 1u : 0u;
    if (hi + carry != 0u) {
        atomicAdd(r.v[2u * i + 1u], hi + carry);
    }
}

void atomicXor64(uint64_t words, uint i, uint64_t v) {
    U32Ref r = U32Ref(words);
    if (uint(v) != 0u) {
        atomicXor(r.v[2u * i], uint(v));
    }
    if (uint(v >> 32) != 0u) {
        atomicXor(r.v[2u * i + 1u], uint(v >> 32));
    }
}

// ---------------- leaf tests ----------------

// The key's Bloom block address; `low` gets the bits that pick the 8 tested bits.
uint64_t bloomBlock(Globals g, uint64_t key, out uint low) {
    uint64_t h = mixKey(key ^ 0x5851f42d4c957f2dul);
    uint part = uint(key >> 56) - g.pBegin;
    uint buf = part / g.bloomPartsPerBuf;
    uint64_t block = uint64_t(part - buf * g.bloomPartsPerBuf) * g.blocksPerPartition +
                     (((h >> 32) * g.blocksPerPartition) >> 32);
    low = uint(h);
    return g.bloomBuf[buf] + block * 32ul;
}

// The block's 8 words (a, b) have the key's 8 bits set.
bool bloomHit(uvec4 a, uvec4 b, uint low) {
    uint missing = (~a.x & (1u << ((low * SALT[0]) >> 27))) | (~a.y & (1u << ((low * SALT[1]) >> 27))) |
                   (~a.z & (1u << ((low * SALT[2]) >> 27))) | (~a.w & (1u << ((low * SALT[3]) >> 27))) |
                   (~b.x & (1u << ((low * SALT[4]) >> 27))) | (~b.y & (1u << ((low * SALT[5]) >> 27))) |
                   (~b.z & (1u << ((low * SALT[6]) >> 27))) | (~b.w & (1u << ((low * SALT[7]) >> 27)));
    return missing == 0u;
}

bool bloomPasses(Globals g, uint64_t key) {
    uint low;
    // the 32-byte block as two 16-byte loads
    U128Ref w = U128Ref(bloomBlock(g, key, low));
    uvec4 a = w.v[0], b = w.v[1];
    return bloomHit(a, b, low);
}

bool leafPasses(Globals g, uint64_t key) {
    if (g.match != 0ul) {
        U64Ref m = U64Ref(g.match);
        uint64_t a = 0ul, b = g.matchCount;
        while (a < b) {
            uint64_t mid = (a + b) / 2ul;
            if (m.v[uint(mid)] < key) {
                a = mid + 1ul;
            } else {
                b = mid;
            }
        }
        return a < g.matchCount && m.v[uint(a)] == key;
    }
    return g.bloom == 0ul || bloomPasses(g, key);
}

// Adds v of every invocation of the subgroup to a counter with one atomic (all must call it).
#define SUBGROUP_ADD_TO(field, v) { uint t_ = subgroupAdd(v); if (subgroupElect() && t_ != 0u) { atomicAdd(field, t_); } }
