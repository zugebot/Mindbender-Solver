#include "memory.hpp"
#include <vector>

#include "rotations.hpp"


/// uint64_t moveValue = a;
void Memory::setNext1Move(uint64_t moveValue) {
    c_u32 moveCount = moves & 0xF;
    c_u8 shiftAmount = 4 + moveCount * 6;
    moves = (moves & ~(0x3FLL << shiftAmount | 0xFULL))
            | (moveValue << shiftAmount)
            | ((moveCount + 1) & 0xF);
}


/// uint64_t moveValue = a | b << 6;
void Memory::setNext2Move(uint64_t moveValue) {
    c_u32 moveCount = moves & 0xF;
    c_u8 shiftAmount = 4 + moveCount * 6;
    moves = (moves & ~(0xFFFULL << shiftAmount | 0xFULL))
            | (moveValue << shiftAmount)
            | ((moveCount + 2) & 0xF);
}


/// uint64_t moveValue = a | b << 6 | c << 12;
void Memory::setNext3Move(uint64_t moveValue) {
    c_u32 moveCount = moves & 0xF;
    c_u8 shiftAmount = 4 + moveCount * 6;
    moves = (moves & ~(0x3FFFFULL << shiftAmount | 0xFULL))
            | (moveValue << shiftAmount)
            | ((moveCount + 3) & 0xF);
}


/// uint64_t moveValue = a | b << 6 | c << 12 | d << 18;
void Memory::setNext4Move(uint64_t moveValue) {
    c_u32 moveCount = moves & 0xF;
    c_u8 shiftAmount = 4 + moveCount * 6;
    moves = (moves & ~(0xFFFFFFULL << shiftAmount | 0xFULL))
            | (moveValue << shiftAmount)
            | ((moveCount + 4) & 0xF);
}


void Memory::setNext5Move(uint64_t moveValue) {
    c_u32 moveCount = moves & 0xF;
    c_u8 shiftAmount = 4 + moveCount * 6;
    moves = (moves & ~(0x3FFFFFFFULL << shiftAmount | 0xFULL))
            | (moveValue << shiftAmount)
            | ((moveCount + 5) & 0xF);
}


u8 Memory::getMove(u8 index) const {
    c_u8 shiftAmount = 4 + (index * 6);
    return (moves >> shiftAmount) & 0x3F;
}


std::string Memory::assembleMoveString(Memory* other) const {
    std::string start = assembleMoveStringForwards();
    std::string end = other->assembleMoveStringBackwards();
    if (start.empty()) {
        return end;
    } else if (end.empty()) {
        return start;
    } else {
        return start + " " + end;
    }
}


std::string Memory::assembleMoveStringForwards() const {
    std::string moves_str;
    const int count = getMoveCount();
    for (int i = 0; i < count; i++) {
        u8 move = getMove(i);
        u8 rowCol = (move % 30) / 5;
        u8 amount = 1 + move % 5;
        char letter = (char)('C' + (15 * (move < 30)));
        moves_str += letter + std::to_string(rowCol)
                 + std::to_string(amount);
        if (i != count - 1) {
            moves_str += " ";
        }
    }
    return moves_str;
}


std::string Memory::assembleMoveStringBackwards() const {
    std::string moves_str;
    const int count = getMoveCount();
    for (int i = count - 1; i >= 0; i--) {
        u8 move = getMove(i);
        u8 rowCol = (move % 30) / 5;
        u8 amount = 6 - (1 + move % 5);
        char letter = (char)('C' + (15 * (move < 30)));
        moves_str += letter + std::to_string(rowCol)
                 + std::to_string(amount);
        if (i != 0) {
            moves_str += " ";
        }
    }
    return moves_str;
}


std::string Memory::assembleFatMoveString(c_u8 fatPos, Memory* other, u8 fatPosOther) const {
    std::string start = assembleFatMoveStringForwards(fatPos);
    if (other == nullptr) {
        return start;
    }
    std::string end = other->assembleFatMoveStringBackwards(fatPosOther);
    if (start.empty()) {
        return end;
    } else if (end.empty()) {
        return start;
    } else {
        return start + " " + end;
    }
}


/*
 int x = board1.getFatX();
   int y = board1.getFatY();
   int move = 21;

fatActions[x * 5 + y][move](board1);
board1.mem.setNext1Move(move);

auto func = fatActions[x * 5 + y][move];
auto permStr1 = RCNameForwardLookup[func];

int amount = permStr1.at(2)  - '0';
x += amount;
x %= 6;
 */

std::string Memory::assembleFatMoveStringForwards(MU u8 fatPos) const {
    std::string moves_str;
    const int count = getMoveCount();
    int x = fatPos / 5;
    int y = fatPos % 5;

    for (int i = 0; i < count; i++) {
        MU u8 move = getMove(i);
        auto func = fatActions[x * 5 + y][move];
        auto segment = RCNameForwardLookup[func];
        moves_str += segment;

        char direction = segment.at(0);
        int axisNum = segment.at(2) - '0';
        int amount = segment.at(segment.size() - 1)  - '0';
        bool movesFat = segment.size() == 6;


        if (movesFat) {
            if (direction == 'R') {
                if (axisNum == y) {
                    x += amount;
                    x %= 6;
                }
            } else {
                if (axisNum == x) {
                    y += amount;
                    y %= 6;
                }
            }
        }



        if (i != count - 1) {
            moves_str += " ";
        }

    }
    return moves_str;
}


std::string Memory::assembleFatMoveStringBackwards(MU u8 fatPos) const {
    std::vector<std::string> moves_vec;
    const int count = getMoveCount();
    int x = fatPos / 5;
    int y = fatPos % 5;

    for (int i = 0; i < count; i++) {
        MU u8 move = getMove(i);
        auto func = fatActions[x * 5 + y][move];
        auto segment = RCNameForwardLookup[func];

        char direction = segment.at(0);
        int axisNum = segment.at(2) - '0';
        int amount = segment.at(segment.size() - 1) - '0';
        char new_amount = static_cast<char>(6 - amount + '0');
        segment = segment.substr(0, segment.size() - 1) + new_amount;
        bool movesFat = segment.size() == 6;

        moves_vec.push_back(segment);

        if (movesFat) {
            if (direction == 'R') {
                if (axisNum == y) {
                    x += amount;
                    x %= 6;
                }
            } else {
                if (axisNum == x) {
                    y += amount;
                    y %= 6;
                }
            }
        }
    }

    std::string moves_str;

    for (int i = (int)moves_vec.size() - 1; i >= 0; i--) {
        moves_str.append(moves_vec[i]);
        if (i != 0) {
            moves_str += " ";
        }
    }

    return moves_str;
}




std::string Memory::toString() const {
    std::string str = "Move[";
    int moveCount = getMoveCount();
    for (int i = 0; i < moveCount; i++) {
        str.append(std::to_string(getMove(i)));
        if (i != moveCount - 1) {
            str.append(", ");
        }
    }
    str.append("]");
    return str;
}


SetNextMoveFunc setNextMoveFuncs[] = {
        nullptr,
        &Memory::setNext1Move,
        &Memory::setNext2Move,
        &Memory::setNext3Move,
        &Memory::setNext4Move,
        &Memory::setNext5Move
};