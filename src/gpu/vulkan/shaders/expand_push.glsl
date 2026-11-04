// src/gpu/vulkan/shaders/expand_push.glsl: the push constants of the expand kernels
// (mbvk.cpp: struct ExpandPush). One level step: nodes [0, n) of `in` expanded, children
// appended to `out` (inner levels) or handed to the leaf mode.

layout(push_constant) uniform ExpandPush {
    uint64_t globals;   // Globals
    uint64_t counters;  // Counters
    uint64_t inB1;      // this launch's first node (the host adds the chunk offset)
    uint64_t inB2;
    uint64_t inPk;      // prev << 24 | seed within the chunk
    uint64_t outB1;     // the next level's buffer (inner levels)
    uint64_t outB2;
    uint64_t outPk;
    uint n;             // nodes in this launch
    uint outCap;        // the next level's capacity
    int rem;            // moves still below a child
    uint prune;         // apply the bound (when limit >= 0)
    uint seedBase;      // index of the chunk's first seed in the call (outputs and checksums)
} pc;
