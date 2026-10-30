#pragma once
// src/perm_stream.hpp

#include "src/chuzzles/perms.hpp"
#include "src/chuzzles/lane_rotations.hpp"
#include "src/utils/jvec.hpp"
#include "src/utils/processor.hpp"

#include <array>
#include <type_traits>

namespace perm_stream_detail {

    template<typename T, i32 MAX_DEPTH>
    struct StreamBuildState {
        static_assert(AllowedPermsType<T>, "T must be Board or B1B2");

        std::array<i32, MAX_DEPTH> dirSeq{};
        std::array<i32, MAX_DEPTH> sectSeq{};
        std::array<i32, MAX_DEPTH> baseSeq{};
        std::array<u64, MAX_DEPTH> curSeq{};
        std::array<bool, MAX_DEPTH> checkRCSeq{};
        std::array<u8, MAX_DEPTH> intersectSeq{};
    };

    template<typename T>
    struct StreamChunk {
        JVec<T> data{};
        JVec<u64> hashes{};
        u32 count = 0;
    };

    template<typename T, typename Sink>
    MU static void flushChunk(StreamChunk<T>& chunk, Sink& sink) {
        if (chunk.count == 0) {
            return;
        }

        sink(chunk.data, chunk.hashes, chunk.count);

        chunk.data.resize(chunk.data.capacity());
        chunk.hashes.resize(chunk.hashes.capacity());
        chunk.count = 0;
    }

    template<typename T, i32 FINAL_DEPTH, typename Sink>
    MU static void emitState(const Board& board_in,
                             const u64 move_prev,
                             StreamChunk<T>& chunk,
                             Sink& sink) {
        static_assert(AllowedPermsType<T>, "T must be Board or B1B2");

        T& out = chunk.data[chunk.count];

        if constexpr (std::is_same_v<T, Board>) {
            out = board_in;
            if constexpr (FINAL_DEPTH > 0) {
                out.memory.template setNextNMove<FINAL_DEPTH>(move_prev);
            }
        } else if constexpr (std::is_same_v<T, B1B2>) {
            out = board_in.asB1B2();
        }

        chunk.hashes[chunk.count] = StateHash::computeHash(out);

        ++chunk.count;
        if EXPECT_FALSE(chunk.count == chunk.data.capacity()) {
            flushChunk(chunk, sink);
        }
    }

    template<typename Sink>
    FORCEINLINE static void emitStateB1B2(const B1B2& state,
                                          StreamChunk<B1B2>& chunk,
                                          Sink& sink) {
        chunk.data[chunk.count] = state;
        chunk.hashes[chunk.count] = StateHash::computeHash(state);

        ++chunk.count;
        if EXPECT_FALSE(chunk.count == chunk.data.capacity()) {
            flushChunk(chunk, sink);
        }
    }

    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM,
             typename Sink>
    static void stream_perm_list_inner(
            const Board& board_in,
            StreamChunk<T>& chunk,
            StreamBuildState<T, MAX_DEPTH>& state,
            u64 move_prev,
            Sink& sink);

    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM,
             bool CHANGE_SECT_START, eSequenceDir SECT_DIR,
             typename Sink>
    static void stream_perm_list_outer(
            const Board& board_in,
            StreamChunk<T>& chunk,
            StreamBuildState<T, MAX_DEPTH>& state,
            Sink& sink);

    template<typename T,
             i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM,
             bool CHANGE_SECT_START, eSequenceDir SECT_DIR,
             typename Sink>
    static void stream_perm_list(
            const Board& board_in,
            StreamChunk<T>& chunk,
            Sink& sink);

    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             eSequenceDir SECT_DIR, bool DIRECTION,
             typename Sink>
    static void stream_fat_perm_list_helper(
            const Board& board,
            StreamChunk<T>& chunk,
            u64 move,
            u8 previousGlobalAction,
            u8 startIndex,
            u8 endIndex,
            Sink& sink);

    template<typename T,
             i32 DEPTH,
             eSequenceDir SECT_DIR,
             typename Sink>
    static void stream_fat_perm_list(
            const Board& board_in,
            StreamChunk<T>& chunk,
            Sink& sink);

    // Board view of a packed state, for Board-only helpers such as the
    // column->row commuting check.
    FORCEINLINE Board makeBoardStateView(const B1B2& state) {
        Board board;
        board.b1 = state.b1;
        board.b2 = state.b2;
        return board;
    }

    template<i32 CUR_DEPTH, i32 MAX_DEPTH,
             eSequenceDir SECT_DIR,
             typename Sink>
    static void stream_fat_lanes_dfs(
            const B1B2& board,
            StreamChunk<B1B2>& chunk,
            Sink& sink,
            i32 prevDir,
            i32 prevLane,
            i32 prevNormalCol,
            u32 prevAmount);

    // Optional prune for stream_lanes_dfs: skipMask<BELOW>(nodes, colLane) is
    // asked once per lane for its 5 shifted nodes at every level above the
    // leaves, BELOW = moves still to come below those nodes, colLane = the
    // shifted lane is a column (bit k set: drop node k's whole subtree). Five
    // at a time so a pruner can overlap its memory reads and share the
    // unmoved half. LEAVES: also asked (BELOW = 0) for the leaves, which are
    // then dropped before they are hashed and emitted. The default compiles to
    // nothing (if constexpr on ENABLED).
    struct NoParentPrune {
        static constexpr bool ENABLED = false;
        static constexpr bool LEAVES = false;
        template<i32 BELOW>
        static FORCEINLINE u32 skipMask(const B1B2 (&)[5], bool) { return 0; }
    };

    template<i32 CUR_DEPTH, i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM,
             eSequenceDir SECT_DIR,
             typename PARENT_PRUNE = NoParentPrune,
             typename Sink>
    static void stream_lanes_dfs(
            const B1B2& board,
            StreamChunk<B1B2>& chunk,
            Sink& sink,
            i32 prevDir,
            i32 prevSect,
            u32 prevAmount);

    // A normal-board move as one byte, in stream_lanes_dfs's own order:
    // dir * 30 + lane * 5 + (amount - 1). NO_PREV_MOVE: the stream starts
    // with no previous move (every first move allowed).
    inline constexpr u8 NO_PREV_MOVE = 255;
    struct PrevMove {
        i32 dir;
        i32 sect;
        u32 amount;
    };
    FORCEINLINE PrevMove decodePrevMove(const u8 m) {
        if (m == NO_PREV_MOVE) {
            return {-1, -1, 0};
        }
        return {m / 30, (m % 30) / 5, static_cast<u32>(m % 5 + 1)};
    }

} // namespace perm_stream_detail


template<typename T>
class PermStream {
    static_assert(AllowedPermsType<T>, "T must be Board or B1B2");

    template<eSequenceDir SECT_DIR, i32 DEPTH, typename Sink>
    MU static void streamDepthImpl(const Board& board_in,
                                   Sink& sink,
                                   const u32 chunkCapacity,
                                   const u8 prevMove = perm_stream_detail::NO_PREV_MOVE) {
        if EXPECT_FALSE(chunkCapacity == 0) {
            return;
        }

        perm_stream_detail::StreamChunk<T> chunk;
        chunk.data.reserve(chunkCapacity);
        chunk.data.resize(chunkCapacity);

        chunk.hashes.reserve(chunkCapacity);
        chunk.hashes.resize(chunkCapacity);

        chunk.count = 0;

        if (board_in.getFatBool()) {
            if constexpr (std::is_same_v<T, B1B2>) {
                if constexpr (DEPTH == 0) {
                    perm_stream_detail::emitStateB1B2(board_in.asB1B2(), chunk, sink);
                } else {
                    perm_stream_detail::stream_fat_lanes_dfs<0, DEPTH, SECT_DIR>(
                            board_in.asB1B2(), chunk, sink, -1, -1, -1, 0);
                }
                perm_stream_detail::flushChunk(chunk, sink);
                return;
            }

            perm_stream_detail::stream_fat_perm_list<
                    T,
                    DEPTH,
                    SECT_DIR>(
                    board_in,
                    chunk,
                    sink
            );

            perm_stream_detail::flushChunk(chunk, sink);
            return;
        }

        // CHECK_CROSS: skip a C -> R pair that commutes on the current board;
        // the R -> C order keeps the same state. Exact-state completeness is
        // covered by testExactSolverRegression.
        if constexpr (std::is_same_v<T, B1B2>) {
            if constexpr (DEPTH == 0) {
                perm_stream_detail::emitStateB1B2(board_in.asB1B2(), chunk, sink);
            } else {
                const perm_stream_detail::PrevMove pm = perm_stream_detail::decodePrevMove(prevMove);
                perm_stream_detail::stream_lanes_dfs<0, DEPTH, true, true, SECT_DIR>(
                        board_in.asB1B2(), chunk, sink, pm.dir, pm.sect, pm.amount);
            }
            perm_stream_detail::flushChunk(chunk, sink);
            return;
        }

        perm_stream_detail::stream_perm_list<
                T,
                DEPTH,
                true,
                true,
                (SECT_DIR != eSequenceDir::NONE),
                SECT_DIR>(
                board_in,
                chunk,
                sink
        );

        perm_stream_detail::flushChunk(chunk, sink);
    }

public:
    // prevMove (normal boards, B1B2 output): the move that reached board_in,
    // encoded as in perm_stream_detail::decodePrevMove, so the first streamed
    // move follows the canonical-order rules too. Only valid with ASCENDING
    // and the last move of board_in's lexicographically smallest path.
    template<eSequenceDir SECT_DIR, i32 DEPTH, typename Sink>
    MU static void streamDepth(const Board& board_in,
                               Sink& sink,
                               const u32 chunkCapacity = 1u << 20,
                               const u8 prevMove = perm_stream_detail::NO_PREV_MOVE) {
        // this assert is arbitrary only for template instantiation
        // and what is a reasonable amount of RAM a PC may have
        static_assert(DEPTH >= 0 && DEPTH <= 5, "DEPTH must be in range 0..5");
        streamDepthImpl<SECT_DIR, DEPTH>(board_in, sink, chunkCapacity, prevMove);
    }

    // streamDepth with PARENT_PRUNE (see perm_stream_detail::NoParentPrune)
    // asked about the nodes at every level above the leaves. Normal boards
    // with B1B2 output only; anything else streams unpruned.
    template<eSequenceDir SECT_DIR, i32 DEPTH, typename PARENT_PRUNE, typename Sink>
    MU static void streamDepthPruned(const Board& board_in,
                                     Sink& sink,
                                     const u32 chunkCapacity = 1u << 20,
                                     const u8 prevMove = perm_stream_detail::NO_PREV_MOVE) {
        static_assert(DEPTH >= 0 && DEPTH <= 5, "DEPTH must be in range 0..5");
        if constexpr (std::is_same_v<T, B1B2> && DEPTH >= 2) {
            if (!board_in.getFatBool() && chunkCapacity != 0) {
                perm_stream_detail::StreamChunk<T> chunk;
                chunk.data.reserve(chunkCapacity);
                chunk.data.resize(chunkCapacity);
                chunk.hashes.reserve(chunkCapacity);
                chunk.hashes.resize(chunkCapacity);
                chunk.count = 0;
                const perm_stream_detail::PrevMove pm = perm_stream_detail::decodePrevMove(prevMove);
                perm_stream_detail::stream_lanes_dfs<0, DEPTH, true, true, SECT_DIR, PARENT_PRUNE>(
                        board_in.asB1B2(), chunk, sink, pm.dir, pm.sect, pm.amount);
                perm_stream_detail::flushChunk(chunk, sink);
                return;
            }
        }
        streamDepthImpl<SECT_DIR, DEPTH>(board_in, sink, chunkCapacity, prevMove);
    }

    template<eSequenceDir SECT_DIR, typename Sink>
    MU static void streamDepthRuntime(const Board& board_in,
                                      Sink& sink,
                                      const u32 depth,
                                      const u32 chunkCapacity = 1u << 20) {
        switch (depth) {
            case 0: streamDepthImpl<SECT_DIR, 0>(board_in, sink, chunkCapacity); break;
            case 1: streamDepthImpl<SECT_DIR, 1>(board_in, sink, chunkCapacity); break;
            case 2: streamDepthImpl<SECT_DIR, 2>(board_in, sink, chunkCapacity); break;
            case 3: streamDepthImpl<SECT_DIR, 3>(board_in, sink, chunkCapacity); break;
            case 4: streamDepthImpl<SECT_DIR, 4>(board_in, sink, chunkCapacity); break;
            case 5: streamDepthImpl<SECT_DIR, 5>(board_in, sink, chunkCapacity); break;
            default: break;
        }
    }
};

#include "src/chuzzles/perm_stream_fat.tpp"
#include "src/chuzzles/perm_stream_nrm.tpp"