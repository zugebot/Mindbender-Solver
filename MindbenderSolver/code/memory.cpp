#include "memory.hpp"

#include <vector>

#include "rotations.hpp"
#include "segments.hpp"

#include <sstream>



// ############################################################
// #                       u64 hash                           #
// ############################################################


void Memory::precomputeHash2(c_u64 b1, c_u64 b2) {
    c_u64 above = getSegment2bits(b1);
    c_u64 below = getSegment2bits(b2);
    setHash(above << 18 | below);
}


void Memory::precomputeHash3(c_u64 b1, c_u64 b2) {
    c_u64 above = getSegment3bits(b1);
    c_u64 below = getSegment3bits(b2);
    setHash(above << 30 | below);
}


void Memory::precomputeHash4(c_u64 b1, c_u64 b2) {
    setHash(prime_func1(b2, b1));
}


MU Memory::HasherPtr Memory::getHashFunc(const Board& board) {
    c_u64 colorCount = board.getColorCount();
    if (board.getFatBool() || colorCount > 3) {
        return &Memory::precomputeHash4;
    }
    if (colorCount == 1 || colorCount == 2) {
        return &Memory::precomputeHash2;
    }
    return &Memory::precomputeHash3;
}


// ############################################################
// #                       u64 moves                          #
// ############################################################


MUND u8 Memory::getMoveCount() const {
    return mem & MEMORY_MOVE_DATA_MASK;
}


u8 Memory::getMove(c_u8 index) const {
    return mem >> getShift(index) & MEMORY_MOVE_TYPE_MASK;
}


u8 Memory::getLastMove() const {
    return mem >> getShift(getMoveCount() - 1) & MEMORY_MOVE_TYPE_MASK;
}


// ############################################################
// #            To String -Similar- Functions                 #
// ############################################################


std::string removeTrailingSpace(std::string& str) {
    if (!str.empty() && str.back() == ' ') {
        str.pop_back();
    }
    return str;
}


std::string Memory::asmString(const Memory* other) const {
    std::string start = asmStringForwards();
    std::string end = other->asmStringBackwards();
    return start.empty() ? end : end.empty() ? start : start + " " + end;

}


std::string Memory::formatMoveString(c_u8 move, c_bool isForwards) {
    char temp[5] = {};
    if (isForwards) {
        memcpy(temp, allActStructList[move].name.data(), 4);
    } else {
        u32 index = move + allActStructList[move].tillNext - 1 - allActStructList[move].tillLast;
        memcpy(temp, allActStructList[index].name.data(), 4);
    }
    return temp;
}


std::string Memory::asmStringForwards() const {
    c_u32 count = getMoveCount();

    std::string moves_str;
    moves_str.reserve(3 * count);

    for (u32 i = 0; i < count; i++) {
        c_u8 move = getMove(i);
        moves_str += formatMoveString(move, true) + " ";
    }
    removeTrailingSpace(moves_str);

    return moves_str;
}


std::string Memory::asmStringBackwards() const {
    c_u32 count = getMoveCount();

    std::string moves_str;
    moves_str.reserve(3 * count);

    for (u32 i = count; i != 0; i--) {
        c_u8 move = getMove(i - 1);
        moves_str += formatMoveString(move, false) + " ";
    }

    removeTrailingSpace(moves_str);
    return moves_str;
}



std::string Memory::asmFatString(c_u8 fatPos, const Memory* other, c_u8 fatPosOther) const {
    std::string start = asmFatStringForwards(fatPos);
    if (other == nullptr) { return start; }
    std::string end = other->asmFatStringBackwards(fatPosOther);
    if (start.empty()) { return end; }
    if (end.empty()) { return start; }
    return start + " " + end;
}


std::string Memory::asmFatStringForwards(c_u8 fatPos) const {
    std::string moves_str;
    c_u32 count = getMoveCount();
    int x = fatPos / 5;
    int y = fatPos % 5;

    for (u32 i = 0; i < count; i++) {
        char temp[5] = {};
        memcpy(temp, allActStructList[
            fatActionsIndexes[x * 5 + y][getMove(i)]].name.data(), 4);
        u32 back = 2 + (temp[3] != '\0');
        moves_str += temp;

        if (back == 3) { // if it is a fat move
            if (temp[0] == 'R') {
                if (temp[1] - '0' == y) { // axisNum
                    x += temp[back] - '0'; // amount
                    x -= 6 * (x > 5);
                }
            } else if EXPECT_TRUE (temp[0] == 'C') {
                if (temp[1] - '0' == x) { // axisNum
                    y += temp[back] - '0'; // amount
                    y -= 6 * (y > 5);
                }
            }
        }
        if (i != count - 1) {
            moves_str += " ";
        }
    }
    return moves_str;
}


std::string Memory::asmFatStringBackwards(c_u8 fatPos) const {
    c_u32 count = getMoveCount();

    std::vector<std::string> moves_vec;
    moves_vec.resize(count);

    int x = fatPos / 5;
    int y = fatPos % 5;

    for (u32 i = 0; i < count; i++) {
        char temp[5] = {};
        memcpy(temp, allActStructList[fatActionsIndexes[x * 5 + y][getMove(i)]].name.data(), 4);
        u32 back = 2 + (temp[3] != '\0');

        if (back == 3) { // if it is a fat move
            if (temp[0] == 'R') {
                if (temp[1] - '0' == y) { // axisNum
                    x += temp[back] - '0'; // amount
                    x -= 6 * (x > 5);
                }
            } else if EXPECT_FALSE(temp[0] == 'C') {
                if (temp[1] - '0' == x) { // axisNum
                    y += temp[back] - '0'; // amount
                    y -= 6 * (y > 5);
                }
            }
        }

        temp[back] = static_cast<char>('f' - temp[back]);
        moves_vec[i] = temp;
    }

    std::string moves_str;

    for (int i = static_cast<int>(moves_vec.size()) - 1; i >= 0; i--) {
        moves_str.append(moves_vec[i]);
        if (i != 0) {
            moves_str += " ";
        }
    }

    return moves_str;
}

std::string Memory::toString() const {
    std::string str = "Move[";

    c_int moveCount = getMoveCount();
    for (int i = 0; i < moveCount; i++) {
        str.append(std::to_string(getMove(i)));
        if (i != moveCount - 1) {
            str.append(", ");
        }
    }
    str.append("]");
    return str;
}


template<bool HAS_FAT>
std::vector<u8> parseMoveStringTemplated(const std::string& input) {
    std::vector<u8> result;
    std::istringstream iss(input);
    std::string seg;

    // Iterate through the string, splitting by spaces
    while (iss >> seg) {
        if (seg.length() == 3) {
            // getActionFromName(seg);
            c_u8 baseValue = seg[0] == 'R' ? 0 : 30;  // R=0, C=30
            c_u32 value = baseValue + (seg[1] - '0') * 5 + (seg[2] - '0') - 1;
            result.push_back(value);

        } else if (seg.length() == 4) {
            // getActionFromName(seg);
            c_u8 baseValue = seg[0] == 'R' ? 60 : 85;  // R=60, C=85
            c_u32 value = baseValue + (seg[1] - '0') * 5 + (seg[3] - '0') - 1;
            result.push_back(value);
        }

    }

    return result;
}


std::vector<u8> Memory::parseNormMoveString(const std::string& input) {
    return parseMoveStringTemplated<false>(input);
}


std::vector<u8> Memory::parseFatMoveString(const std::string& input) {
    return parseMoveStringTemplated<true>(input);
}
