#pragma once
// src/chuzzles/first_moves.hpp
//
// The one-move children of a board that the canonical streams (stream_lanes_dfs,
// stream_fat_lanes_dfs) visit right after `prevMove`, in visiting order, each with
// its move code: dir * 30 + lane * 5 + (amount - 1), a fat lane coded at its first
// index (perm_stream_detail::decodePrevMove). NO_PREV_MOVE gives every first move.
// Same rules as the streams: same-direction lanes ascending, no move that gives the
// board back (and nothing past a lane's period on normal boards), a normal column
// followed by a normal row only when the pair does not commute, no fat amount that
// puts the 2x2 piece across the edge. Used to code each state's lexicographically
// smallest path (ssd::buildLastMoves) and to expand the goal side's frontiers.

#include "src/chuzzles/include.hpp"
#include "src/chuzzles/lane_rotations.hpp"
#include "src/chuzzles/perm_stream.hpp"

namespace perm_stream_detail {

    template<typename Fn>
    FORCEINLINE void forEachFirstMove(const B1B2& board, const u8 prevMove, Fn&& fn) {
        const bool fat = board.getFatBool();
        const PrevMove pm = decodePrevMove(prevMove);
        const u32 fatX = fat ? board.getFatX() : 7;
        const u32 fatY = fat ? board.getFatY() : 7;
        // the previous move was a normal column: the commuting rule applies to normal rows
        const i32 prevNormalCol = pm.dir == 1 && static_cast<u32>(pm.sect) != fatX ? pm.sect : -1;
        for (i32 dir = 0; dir < 2; ++dir) {
            const i32 fatLane = fat ? static_cast<i32>(dir == 0 ? fatY : fatX) : -2;
            for (i32 lane = dir == pm.dir ? pm.sect + 1 : 0; lane <= 5; ++lane) {
                if (lane == fatLane + 1) {
                    continue;   // second row / column of the fat pair
                }
                const bool isFatLane = lane == fatLane;
                B1B2 shifted[5];
                u32 skip = 0;
                if (dir == 0) {
                    if (isFatLane) {
                        lane_rotations::fatRowShifts(board, static_cast<u32>(lane), fatX, shifted);
                        skip = 1u << (4 - fatX);
                    } else {
                        lane_rotations::rowShifts(board, static_cast<u32>(lane), shifted);
                        if (prevNormalCol >= 0) {
                            skip = makeBoardStateView(board).doActISColMatchBatched(
                                    static_cast<u8>(prevNormalCol), static_cast<u8>(lane), static_cast<u8>(pm.amount));
                        }
                    }
                } else if (isFatLane) {
                    lane_rotations::fatColShifts(board, static_cast<u32>(lane), fatY, shifted);
                    skip = 1u << (4 - fatY);
                } else {
                    lane_rotations::colShifts(board, static_cast<u32>(lane), shifted);
                }
                for (u32 k = 0; k < 5; ++k) {
                    if (shifted[k] == board) {
                        if (fat) {
                            continue;   // a dropped fat amount's repeat must still run (stream_fat_lanes_dfs)
                        }
                        break;          // the lane's period: larger amounts repeat smaller ones
                    }
                    if (skip & (1u << k)) {
                        continue;
                    }
                    fn(static_cast<u8>(dir * 30 + lane * 5 + static_cast<i32>(k)), shifted[k]);
                }
            }
        }
    }

} // namespace perm_stream_detail
