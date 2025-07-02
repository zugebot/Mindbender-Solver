#pragma once

#include <cstdint>
#include <string>


template<int BASE=1024>
std::string bytesFormatted(const uint64_t bytes) {
    int bases = 0;
    int leftover = 0;

    uint64_t bytesCopy = bytes;
    while (bytesCopy > BASE) {
        leftover = bytesCopy % BASE;
        bytesCopy /= BASE;
        bases += 1;
    }

    const std::string baseStrs[6] = {"B", "KB", "MB", "GB", "TB", "?B"};
    if (bases > 5) bases = 5;

    const std::string floatStr = std::to_string(static_cast<float>(leftover) / static_cast<float>(BASE));


    std::string ret;
    ret.append(std::to_string(bytesCopy));
    ret.append(floatStr.substr(1, std::min(static_cast<int>(floatStr.size() - 1), 4)));
    ret.append(baseStrs[bases]);
    return ret;
}