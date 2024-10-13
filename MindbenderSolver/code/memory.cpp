#include "memory.hpp"

#include <vector>

#include "rotations.hpp"


MUND u8 Memory::getMoveCount() const {
    return moves & MEMORY_MOVE_DATA_MASK;
}


u8 Memory::getMove(c_u8 index) const {
    return moves >> getShift(index) & MEMORY_MOVE_TYPE_MASK;
}


u8 Memory::getLastMove() const {
    return moves >> getShift(getMoveCount() - 1) & MEMORY_MOVE_TYPE_MASK;
}


// ############################################################
// #            To String -Similar- Functions                 #
// ############################################################


static constexpr u32 NORMAL_PERMUTATION_LENGTH = 3;
static constexpr u32 LONGER_PERMUTATION_LENGTH = 4;


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


std::string Memory::formatMoveString(c_u8 move, c_bool isBackwards) {
    c_u8 rowCol = move % 30 / 5;
    c_u8 amount = isBackwards ? 6 - (1 + move % 5) : 1 + move % 5;
    c_char letter = static_cast<char>('C' + 15 * (move < 30));
    return letter + std::to_string(rowCol) + std::to_string(amount);
}


std::string Memory::asmStringForwards() const {
    c_u32 count = getMoveCount();

    std::string moves_str;
    moves_str.reserve(NORMAL_PERMUTATION_LENGTH * count);

    for (u32 i = 0; i < count; i++) {
        c_u8 move = getMove(i);
        moves_str += formatMoveString(move, false) + " ";
    }
    removeTrailingSpace(moves_str);

    return moves_str;
}


std::string Memory::asmStringBackwards() const {
    c_u32 count = getMoveCount();

    std::string moves_str;
    moves_str.reserve(NORMAL_PERMUTATION_LENGTH * count);

    for (u32 i = count; i != 0; i--) {
        c_u8 move = getMove(i - 1);
        moves_str += formatMoveString(move, true) + " ";
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
        c_u8 move = getMove(i);

        c_auto func = allActionsList[fatActionsIndexes[x * 5 + y][move]];
        std::string segment = getNameFromAction(func);
        moves_str += segment;


        if (segment.size() == LONGER_PERMUTATION_LENGTH) {
            c_int axisNum = segment.at(2) - '0';
            c_int amount = segment.back() - '0';
            if (segment.at(0) == 'R') {
                if (axisNum == y) { x = (x + amount) % 6; }
            } else {
                if (axisNum == x) { y = (y + amount) % 6; }
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
        c_u8 move = getMove(i);
        c_auto func = allActionsList[fatActionsIndexes[x * 5 + y][move]];

        std::string segment = getNameFromAction(func);


        c_char new_amount = static_cast<char>('f' - segment.back());
        segment = segment.substr(0, segment.size() - 1) + new_amount;

        moves_vec[i] = segment;

        if (segment.size() == LONGER_PERMUTATION_LENGTH) {
            c_int axisNum = segment.at(2) - '0';
            c_int amount = segment.back() - '0';
            if (segment.at(0) == 'R') {
                if (axisNum == y) { x = (x + amount) % 6; }
            } else {
                if (axisNum == x) { y = (y + amount) % 6; }
            }
        }
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