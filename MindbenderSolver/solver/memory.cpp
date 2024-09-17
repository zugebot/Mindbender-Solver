#include "memory.hpp"


/// uint64_t moveValue = a;
void Memory::setNext1Move(uint64_t moveValue) {
    const uint32_t moveCount = moves & 0xF;
    const uint8_t shiftAmount = 4 + moveCount * 6;
    moves = (moves & ~(MOVE_MASK << shiftAmount | 0xFULL))
            | (moveValue << shiftAmount)
            | ((moveCount + 1) & 0xF);
}


/// uint64_t moveValue = a | b << 6;
void Memory::setNext2Move(uint64_t moveValue) {
    const uint32_t moveCount = moves & 0xF;
    const uint8_t shiftAmount = 4 + moveCount * 6;
    moves = (moves & ~(0xFFFULL << shiftAmount | 0xFULL))
            | (moveValue << shiftAmount)
            | ((moveCount + 2) & 0xF);
}


/// uint64_t moveValue = a | b << 6 | c << 12;
void Memory::setNext3Move(uint64_t moveValue) {
    const uint32_t moveCount = moves & 0xF;
    const uint8_t shiftAmount = 4 + moveCount * 6;
    moves = (moves & ~(0x3FFFFULL << shiftAmount | 0xFULL))
            | (moveValue << shiftAmount)
            | ((moveCount + 3) & 0xF);
}


/// uint64_t moveValue = a | b << 6 | c << 12 | d << 18;
void Memory::setNext4Move(uint64_t moveValue) {
    const uint32_t moveCount = moves & 0xF;
    const uint8_t shiftAmount = 4 + moveCount * 6;
    moves = (moves & ~(0xFFFFFFULL << shiftAmount | 0xFULL))
            | (moveValue << shiftAmount)
            | ((moveCount + 4) & 0xF);
}


void Memory::setNext5Move(uint64_t moveValue) {
    const uint32_t moveCount = moves & 0xF;
    const uint8_t shiftAmount = 4 + moveCount * 6;
    moves = (moves & ~(0x3FFFFFFFULL << shiftAmount | 0xFULL))
            | (moveValue << shiftAmount)
            | ((moveCount + 5) & 0xF);
}


MU ND uint8_t Memory::getMove(uint8_t index) const {
    const uint8_t shiftAmount = 4 + (index * 6);
    return (moves >> shiftAmount) & MOVE_MASK;
}


MU ND std::string Memory::toString() const {
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
        &Memory::setNext1Move,
        &Memory::setNext2Move,
        &Memory::setNext3Move,
        &Memory::setNext4Move,
        &Memory::setNext5Move
};