#pragma once

#include <string>
#include <vector>

class Colors {
public:
    static const std::string Red = "\033[31m";
    static const std::string Orange = "\033[38;5;208m";
    static const std::string Yellow = "\033[93m";
    static const std::string Green = "\033[32m";
    static const std::string Cyan = "\033[96m";
    static const std::string CyanBold = "\033[96;1m";
    static const std::string Blue = "\033[34m";
    static const std::string Magenta = "\033[35m";
    static const std::string White = "\033[37m";
    static const std::string Black = "\033[30m";
    static const std::string Reset = "\033[0;0m";

    static const std::vector<const std::string&> colors = {
        Red, Orange, Yellow, Green, Cyan, Blue, Magenta, White, Black, Reset
    };

    static std::string getColor(const unsigned char index) {
        if (index < 9) {
            return colors[index];
        }
        return colors[9];
    }
};


