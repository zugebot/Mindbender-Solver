#pragma once

#include <cstdint>
#include <string>


template<int BASE=1024>
std::string bytesFormatted(uint64_t bytes) {
    int bases = 0;

    uint64_t bytesCopy = bytes;
    while (bytesCopy > BASE) {
        bytesCopy /= BASE;
        bases += 1;
    }

    std::string baseStrs[6] = {"B", "KB", "MB", "GB", "TB", "?B"};
    if (bases > 5) bases = 5;

    std::string ret;
    ret.append(std::to_string(bytesCopy));
    ret.append(baseStrs[bases]);
    return ret;
}