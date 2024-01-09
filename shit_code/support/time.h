#pragma once

#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

class Time {
public:

    static double getTime() {
        auto now = std::chrono::system_clock::now();
        return (double)(std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count());
    }

    static double time(double start) {
        return (getTime() - start) / 1000.0;
    }
};