#pragma once

#include <string>
#include <vector>

class Colors {
public:
    static constexpr std::string Red = "\033[31m";
    static constexpr std::string Orange = "\033[38;5;208m";
    static constexpr std::string Yellow = "\033[93m";
    static constexpr std::string Green = "\033[32m";
    static constexpr std::string Cyan = "\033[96m";
    static constexpr std::string CyanBold = "\033[96;1m";
    static constexpr std::string Blue = "\033[34m";
    static constexpr std::string Magenta = "\033[35m";
    static constexpr std::string White = "\033[37m";
    static constexpr std::string Black = "\033[30m";
    static constexpr std::string Reset = "\033[0;0m";

    static constexpr std::vector<const std::string&> colors = {
        Red, Orange, Yellow, Green, Cyan, Blue, Magenta, White, Black, Reset
    };

    static std::string getColor(const unsigned char index) {
        if (index < 9) {
            return colors[index];
        }
        return colors[9];
    }
};


