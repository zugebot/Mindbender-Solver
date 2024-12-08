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
    static constexpr const char* Red = "\033[31m";
    static constexpr const char* Green = "\033[32m";
    static constexpr const char* Blue = "\033[34m";
    static constexpr const char* Orange = "\033[38;5;208m";
    // static constexpr std::string Yellow = "\033[93m";
    static constexpr const char* Yellow = "\033[33m";
    static constexpr const char* Magenta = "\033[35m";
    static constexpr const char* White = "\033[37m";
    static constexpr const char* Cyan = "\033[96m";
    static constexpr const char* Black = "\033[90m";
    static constexpr const char* Reset = "\033[0;0m";
    // static constexpr std::string CyanBold = "\033[96;1m";

    static constexpr const char* bgRed      = "\033[41m";
    static constexpr const char* bgGreen    = "\033[42m";
    static constexpr const char* bgBlue     = "\033[44m";
    static constexpr const char* bgOrange   = "\033[48;5;166m";
    static constexpr const char* bgYellow   = "\033[43m";
    static constexpr const char* bgMagenta  = "\033[45m";
    static constexpr const char* bgWhite    = "\033[47m";
    static constexpr const char* bgCyan     = "\033[106m";
    static constexpr const char* bgBlack    = "\033[40m";
    static constexpr const char* bgReset    = "\033[0m";

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