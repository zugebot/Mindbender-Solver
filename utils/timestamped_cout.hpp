#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>

class TimestampedStreamBuf final : public std::streambuf {
private:
    std::streambuf* dest_;
    std::chrono::steady_clock::time_point startTime_;
    bool atLineStart_ = true;

private:
    static std::string formatElapsed(std::chrono::steady_clock::duration elapsed) {
        using namespace std::chrono;

        const auto totalMs = duration_cast<milliseconds>(elapsed).count();

        const long long hours = totalMs / 3'600'000;
        const long long minutes = (totalMs / 60'000) % 60;
        const long long seconds = (totalMs / 1'000) % 60;
        const long long millis = totalMs % 1'000;

        std::ostringstream oss;
        oss << '['
            << hours << ':'
            << std::setw(2) << std::setfill('0') << minutes << ':'
            << std::setw(2) << std::setfill('0') << seconds << '.'
            << std::setw(3) << std::setfill('0') << millis
            << "] ";
        return oss.str();
    }

    void writeTimestampIfNeeded() {
        if (!atLineStart_) {
            return;
        }

        const std::string prefix = formatElapsed(std::chrono::steady_clock::now() - startTime_);
        dest_->sputn(prefix.data(), static_cast<std::streamsize>(prefix.size()));
        atLineStart_ = false;
    }

protected:
    int overflow(int ch) override {
        if (ch == traits_type::eof()) {
            return sync() == 0 ? traits_type::not_eof(ch) : traits_type::eof();
        }

        writeTimestampIfNeeded();

        if (dest_->sputc(static_cast<char>(ch)) == traits_type::eof()) {
            return traits_type::eof();
        }

        if (ch == '\n') {
            atLineStart_ = true;
        }

        return ch;
    }

    std::streamsize xsputn(const char* s, std::streamsize count) override {
        std::streamsize written = 0;

        for (std::streamsize i = 0; i < count; ++i) {
            if (overflow(static_cast<unsigned char>(s[i])) == traits_type::eof()) {
                break;
            }
            ++written;
        }

        return written;
    }

    int sync() override {
        return dest_->pubsync();
    }

public:
    explicit TimestampedStreamBuf(std::streambuf* dest)
        : dest_(dest), startTime_(std::chrono::steady_clock::now()) {}

    void resetTimer() {
        startTime_ = std::chrono::steady_clock::now();
    }
};

class TimestampedCout final : public std::ostream {
private:
    TimestampedStreamBuf buffer_;

public:
    TimestampedCout()
        : std::ostream(&buffer_), buffer_(std::cout.rdbuf()) {}

    void resetTimer() {
        buffer_.resetTimer();
    }
};

inline TimestampedCout tcout;