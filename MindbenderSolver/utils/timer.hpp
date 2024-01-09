#pragma once

#include <cstdint>


static uint64_t getNanoSeconds();
static uint64_t getMilliseconds();
static uint64_t getSeconds();


class Timer {
    uint64_t time = 0;
public:
    Timer();
    [[nodiscard]] float getSeconds() const;
};