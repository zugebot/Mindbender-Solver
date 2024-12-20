#pragma once

#include <string>


/**
RED     0
GREEN   1
BLUE    2
ORANGE  3
YELLOW  4
PURPLE  5
WHITE   6
CYAN    7
BLACK   8
RESET   9
 */
class Colors {
public:
    static constexpr std::string Red = "\033[31m";
    static constexpr std::string Green = "\033[32m";
    static constexpr std::string Blue = "\033[34m";
    static constexpr std::string Orange = "\033[38;5;208m";
    // static constexpr std::string Yellow = "\033[93m";
    static constexpr std::string Yellow = "\033[33m";
    static constexpr std::string Magenta = "\033[35m";
    static constexpr std::string White = "\033[37m";
    static constexpr std::string Cyan = "\033[96m";
    static constexpr std::string Black = "\033[90m";
    static constexpr std::string Reset = "\033[0;0m";
    // static constexpr std::string CyanBold = "\033[96;1m";

    static constexpr std::string bgRed      = "\033[41m";
    static constexpr std::string bgGreen    = "\033[42m";
    static constexpr std::string bgBlue     = "\033[44m";
    static constexpr std::string bgOrange   = "\033[48;5;166m";
    static constexpr std::string bgYellow   = "\033[43m";
    static constexpr std::string bgMagenta  = "\033[45m";
    static constexpr std::string bgWhite    = "\033[47m";
    static constexpr std::string bgCyan     = "\033[106m";
    static constexpr std::string bgBlack    = "\033[40m";
    static constexpr std::string bgReset    = "\033[0m";

    static const std::string colors[10];
    static const std::string bgColors[10];

    static std::string getColor(const int index) {
        if (index < 0) { return colors[8]; }
        if (index < 8) { return colors[index]; }
        return colors[8];
    }

    static std::string getBgColor(const int index) {
        if (index < 0) { return bgColors[8]; }
        if (index < 8) { return bgColors[index]; }
        return bgColors[8];
    }
};

const std::string Colors::colors[10] = {
        Red, Green, Blue, Orange, Yellow,
        Magenta, White, Cyan, Black, Reset
};

const std::string Colors::bgColors[10] = {
        bgRed, bgGreen, bgBlue, bgOrange, bgYellow,
        bgMagenta, bgWhite, bgCyan, bgBlack, bgReset
};