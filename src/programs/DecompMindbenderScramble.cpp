#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace mindbender_repro {

    struct Random55 {
        static constexpr std::uint32_t kMask = 0x3fffffffU;

        int idx0 = 0;
        int idx31 = 31;
        std::array<std::uint32_t, 55> state{};

        void seed(std::uint32_t seedValue) {
            state.fill(0);
            state[0] = seedValue & kMask;
            idx0 = 0;
            idx31 = 31;
            state[1] = 1;
            for (int i = 0; i < 53; ++i) {
                state[i + 2] = (state[i] + state[i + 1]) & kMask;
            }
        }

        int nextRaw() {
            const std::uint32_t value = (state[idx31] + state[idx0]) & kMask;
            state[idx0] = value;
            idx0 += 1;
            if (idx0 == 55) {
                idx0 = 0;
            }
            idx31 += 1;
            if (idx31 == 55) {
                idx31 = 0;
            }
            return static_cast<int>(value >> 6);
        }

        int next(int limit, bool signedResult = false) {
            if (limit == 0) {
                return 0;
            }
            if (limit < 0) {
                limit = -limit;
                signedResult = true;
            }
            int pow2 = 2;
            while (pow2 < limit) {
                pow2 *= 2;
            }
            const int value = (nextRaw() & (pow2 - 1)) % limit;
            if (signedResult && next(2, false) == 1) {
                return -value;
            }
            return value;
        }
    };

    struct Cell {
        int color = 0;
        bool fatTopLeft = false;
        bool fatCovered = false;
        int fatOffsetX = 0;
        int fatOffsetY = 0;
    };

    struct ParsedMindbenderLine {
        std::uint32_t seed = 0;
        int scrambleCount = 0;
        std::array<std::string, 6> rows{};
    };

    struct Board {
        std::array<std::array<Cell, 6>, 6> cells{};
        std::array<std::array<int, 6>, 6> originalColors{};

        static int parseColorChar(char ch, bool& fatTopLeft) {
            fatTopLeft = false;
            if (static_cast<unsigned char>(ch) < static_cast<unsigned char>('A')) {
                return static_cast<int>(ch) - static_cast<int>('0');
            }
            if (static_cast<unsigned char>(ch) < static_cast<unsigned char>('I')) {
                fatTopLeft = true;
                return static_cast<int>(ch) - static_cast<int>('A');
            }
            return static_cast<int>(ch) - static_cast<int>('I');
        }

        void loadFromRows(const std::array<std::string, 6>& rows) {
            for (int y = 0; y < 6; ++y) {
                if (rows[y].size() != 6) {
                    throw std::runtime_error("Each row must be exactly 6 characters.");
                }
            }

            for (auto& row : cells) {
                for (auto& cell : row) {
                    cell = Cell{};
                }
            }

            for (int y = 0; y < 6; ++y) {
                for (int x = 0; x < 6; ++x) {
                    bool fatTopLeft = false;
                    const int color = parseColorChar(rows[y][x], fatTopLeft);
                    cells[y][x].color = color;
                    originalColors[y][x] = color;

                    if (fatTopLeft) {
                        if (x + 1 >= 6 || y + 1 >= 6) {
                            throw std::runtime_error("Fat top-left marker is on the last row or column.");
                        }

                        cells[y][x].fatTopLeft = true;

                        cells[y][x + 1].fatCovered = true;
                        cells[y][x + 1].fatOffsetX = 1;
                        cells[y][x + 1].fatOffsetY = 0;

                        cells[y + 1][x].fatCovered = true;
                        cells[y + 1][x].fatOffsetX = 0;
                        cells[y + 1][x].fatOffsetY = 1;

                        cells[y + 1][x + 1].fatCovered = true;
                        cells[y + 1][x + 1].fatOffsetX = 1;
                        cells[y + 1][x + 1].fatOffsetY = 1;
                    }
                }
            }
        }

        static ParsedMindbenderLine parseLineExactlyLikeGame(const std::string& line) {
            if (line.size() < 48) {
                throw std::runtime_error("mindbenders.txt line is too short.");
            }

            ParsedMindbenderLine parsed;
            parsed.seed = static_cast<std::uint32_t>(std::atol(line.substr(0, 3).c_str()));
            parsed.scrambleCount = static_cast<int>(std::atol(line.substr(4, 2).c_str()));

            for (int row = 0; row < 6; ++row) {
                parsed.rows[row] = line.substr(7 + row * 7, 6);
            }
            return parsed;
        }

        static char serializeCell(const Cell& cell) {
            if (cell.fatTopLeft) {
                if (cell.color < 0 || cell.color > 7) {
                    return '?';
                }
                return static_cast<char>('A' + cell.color);
            }

            if (cell.color >= 0 && cell.color <= 9) {
                return static_cast<char>('0' + cell.color);
            }

            return static_cast<char>('I' + cell.color);
        }

        std::array<std::string, 6> toRows() const {
            std::array<std::string, 6> rows{};
            for (int y = 0; y < 6; ++y) {
                std::string row;
                row.reserve(6);
                for (int x = 0; x < 6; ++x) {
                    row.push_back(serializeCell(cells[y][x]));
                }
                rows[y] = row;
            }
            return rows;
        }

        int scanRowForPartnerDelta(int row) const {
            int delta = 0;
            for (int x = 0; x < 6; ++x) {
                const Cell& cell = cells[row][x];
                if (cell.fatTopLeft && delta == 0) {
                    delta = 1;
                }
                if (cell.fatCovered && delta == 0) {
                    delta = -1;
                }
            }
            return delta;
        }

        int scanColForPartnerDelta(int col) const {
            int delta = 0;
            for (int y = 0; y < 6; ++y) {
                const Cell& cell = cells[y][col];
                if (cell.fatTopLeft && delta == 0) {
                    delta = 1;
                }
                if (cell.fatCovered && delta == 0) {
                    delta = -1;
                }
            }
            return delta;
        }

        void rotateRowLeftOne(int row) {
            const Cell first = cells[row][0];
            for (int x = 0; x < 5; ++x) {
                cells[row][x] = cells[row][x + 1];
            }
            cells[row][5] = first;
        }

        void rotateColUpOne(int col) {
            const Cell first = cells[0][col];
            for (int y = 0; y < 5; ++y) {
                cells[y][col] = cells[y + 1][col];
            }
            cells[5][col] = first;
        }

        bool hasTopLeftOnRightmostColOrBottomRow() const {
            for (int y = 0; y < 6; ++y) {
                if (cells[y][5].fatTopLeft) {
                    return true;
                }
            }
            for (int x = 0; x < 6; ++x) {
                if (cells[5][x].fatTopLeft) {
                    return true;
                }
            }
            return false;
        }

        void scrambleExactlyLikeGame(std::uint32_t seed, int scrambleCount) {
            Random55 rng;
            rng.seed(seed);

            if (scrambleCount <= 0) {
                return;
            }

            for (int scrambleIndex = 0; scrambleIndex < scrambleCount; ++scrambleIndex) {
                const int selectedCol = rng.next(6, false);
                const int selectedRow = rng.next(6, false);
                const int axis = rng.next(2, false);

                const bool doRowMove = (axis == 0);
                const bool doColMove = (axis == 1);

                int repetitions = rng.next(12, false) + 1;
                int performed = 0;

                while (performed < repetitions) {
                    if (doRowMove) {
                        const int partnerDelta = scanRowForPartnerDelta(selectedRow);
                        rotateRowLeftOne(selectedRow);
                        if (partnerDelta != 0) {
                            const int partnerRow = selectedRow + partnerDelta;
                            if (partnerRow < 0 || partnerRow >= 6) {
                                throw std::runtime_error("Illegal fat partner row encountered.");
                            }
                            rotateRowLeftOne(partnerRow);
                        }
                    }

                    if (doColMove) {
                        const int partnerDelta = scanColForPartnerDelta(selectedCol);
                        rotateColUpOne(selectedCol);
                        if (partnerDelta != 0) {
                            const int partnerCol = selectedCol + partnerDelta;
                            if (partnerCol < 0 || partnerCol >= 6) {
                                throw std::runtime_error("Illegal fat partner column encountered.");
                            }
                            rotateColUpOne(partnerCol);
                        }
                    }

                    if (hasTopLeftOnRightmostColOrBottomRow()) {
                        --performed;
                    }
                    ++performed;
                }
            }
        }
    };

    std::string formatMindbenderLine(std::uint32_t seed, int scrambleCount,
                                     const std::array<std::string, 6>& rows) {
        std::ostringstream out;
        out << std::setw(3) << std::setfill('0') << seed << ' '
            << std::setw(2) << std::setfill('0') << scrambleCount << ' ';
        for (int i = 0; i < 6; ++i) {
            if (i != 0) {
                out << ' ';
            }
            out << rows[i];
        }
        return out.str();
    }

    void printRows(const std::array<std::string, 6>& rows, const std::string& label) {
        std::cout << label << '\n';
        for (const auto& row : rows) {
            std::cout << "  " << row << '\n';
        }
    }

} // namespace mindbender_repro

int main() {
    using namespace mindbender_repro;

    const std::array<std::string, 6> rows = {
            "110011",
            "110011",
            "001100",
            "001100",
            "110011",
            "110011"
    };

    const std::uint32_t seed = 1;
    const int scrambleCount = 2;

    Board board;
    board.loadFromRows(rows);

    printRows(rows, "Initial rows:");
    std::cout << "Initial line: " << formatMindbenderLine(seed, scrambleCount, rows) << "\n\n";

    board.scrambleExactlyLikeGame(seed, scrambleCount);

    const auto finalRows = board.toRows();
    printRows(finalRows, "Scrambled rows:");
    std::cout << "Scrambled line: " << formatMindbenderLine(seed, scrambleCount, finalRows) << "\n";

    return 0;
}
