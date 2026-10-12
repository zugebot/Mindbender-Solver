#pragma once
// src/solver/ssd/recovery.hpp
//
// From middle states to full solutions (exact recovery through every
// middle), and the known-solution checks used by --solutions.

#include "src/solver/ssd/common.hpp"

namespace ssd {

    inline std::vector<std::string> readSolutionLines(const std::string& file) {
        std::vector<std::string> lines;
        std::ifstream in(file);
        std::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
                line.pop_back();
            }
            if (!line.empty()) {
                lines.push_back(line);
            }
        }
        return lines;
    }

    // Middles implied by a complete optimal solution file: the board after the
    // first `prefixMoves` moves of every solution, applied to the true start.
    inline B1B2 middleOf(const BoardPair& pair, const std::string& line, const std::size_t prefixMoves) {
        const bool isFat = pair.getStartState().getFatBool();
        const std::vector<u8> moves = isFat ? Memory::parseFatMoveString(line)
                                            : Memory::parseNormMoveString(line);
        if (moves.size() < prefixMoves) {
            throw std::runtime_error("solution shorter than the middle depth: " + line);
        }
        Board board = pair.getStartState();
        for (std::size_t i = 0; i < prefixMoves; ++i) {
            allActStructList[moves[i]].action(board);
        }
        return board.asB1B2();
    }

    inline std::vector<B1B2> expectedMiddles(const BoardPair& pair,
                                      const std::vector<std::string>& solutions,
                                      const std::size_t prefixMoves) {
        std::vector<B1B2> out;
        for (const std::string& line : solutions) {
            out.push_back(middleOf(pair, line, prefixMoves));
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    template<int TOTAL_LEFT_DEPTH, int TABLE_DEPTH>
    int recoverWithLeftDepth(BoardSolverFrontier& solver, const bool reverse,
                             const std::vector<B1B2>& middles, const u32 threads) {
        if (reverse) {
            return solver.recoverSolutionsFromMiddles<TOTAL_LEFT_DEPTH, TABLE_DEPTH, true>(middles, static_cast<int>(threads));
        }
        return solver.recoverSolutionsFromMiddles<TOTAL_LEFT_DEPTH, TABLE_DEPTH, false>(middles, static_cast<int>(threads));
    }

    template<int TABLE_DEPTH>
    int recoverWithTableDepth(BoardSolverFrontier& solver, const Config& cfg, const std::vector<B1B2>& middles) {
        switch (cfg.seedDepth + cfg.streamDepth) {
            case 4: return recoverWithLeftDepth<4, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 5: return recoverWithLeftDepth<5, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 6: return recoverWithLeftDepth<6, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 7: return recoverWithLeftDepth<7, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 8: return recoverWithLeftDepth<8, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 9: return recoverWithLeftDepth<9, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            default: throw std::runtime_error("unsupported left depth for recovery");
        }
    }

    inline int recoverSolutions(BoardSolverFrontier& solver, const Config& cfg, const std::vector<B1B2>& middles) {
        return cfg.tableDepth == 7 ? recoverWithTableDepth<7>(solver, cfg, middles)
                                   : recoverWithTableDepth<6>(solver, cfg, middles);
    }

} // namespace ssd
