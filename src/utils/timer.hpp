#pragma once

#include <chrono>


class Timer {
public:
    Timer() {
        start_time = std::chrono::steady_clock::now();
    }
    [[nodiscard]] double getSeconds() const {
        const auto end_time = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end_time - start_time;
        return elapsed.count();
    }
    [[maybe_unused]] void reset() {
        start_time = std::chrono::steady_clock::now();
    }

private:
    using clock = std::chrono::steady_clock;
    using time_point = std::chrono::time_point<clock>;

    time_point start_time;
};