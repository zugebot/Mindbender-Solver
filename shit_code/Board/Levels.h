#pragma once

#include "Board.h"

class Levels {
public:
    Board b1_1 = Board(
        "G G R R G G"
        "G G R R G G"
        "R R R G G G"
        "R R G G G R"
        "G G R R R G"
        "G G R R R G");
    Board s1_1 = Board(
        "G G R R G G"
        "G G R R G G"
        "R R G G R R"
        "R R G G R R"
        "G G R R G G"
        "G G R R G G");
    Board b1_2 = Board(
        "B O B B B B"
        "O O O O B B"
        "B B B B O B"
        "B B B B O B"
        "B O O O O B"
        "B O B B B B");
    Board s1_2 = Board(
        "B B B B B B"
        "B O O O O B"
        "B O B B O B"
        "B O B B O B"
        "B O O O O B"
        "B B B B B B");
    Board b1_3 = Board(
        "Y Y P Y Y Y"
        "Y P P P P Y"
        "Y P P P P Y"
        "Y P Y P P Y"
        "Y P Y P P Y"
        "Y Y P Y Y Y");
    Board s1_3 = Board(
        "Y Y Y Y Y Y"
        "Y P P P P Y"
        "Y P P P P Y"
        "Y P P P P Y"
        "Y P P P P Y"
        "Y Y Y Y Y Y");
    Board b1_4 = Board(
        "R Y R R R R"
        "Y Y R R Y R"
        "Y R Y Y R R"
        "R R Y Y R R"
        "R Y R R Y R"
        "R R Y R R R");
    Board s1_4 = Board(
        "Y R R R R Y"
        "R Y R R Y R"
        "R R Y Y R R"
        "R R Y Y R R"
        "R Y R R Y R"
        "Y R R R R Y");
    Board b1_5 = Board(
        "C C C C C C"
        "C B B B C C"
        "C B C C B C"
        "C B C C B C"
        "B C B B B B"
        "C C C C C C");
    Board s1_5 = Board(
        "C C C C C C"
        "C B B B B C"
        "C B C C B C"
        "C B C C B C"
        "C B B B B C"
        "C C C C C C");
    Board b2_1 = Board(
        "W R R R R W"
        "R R W W R R"
        "W W W W W R"
        "W W W W R W"
        "R R W W W R"
        "R R W W W R");
    Board s2_1 = Board(
        "R R W W R R"
        "R R W W R R"
        "W W W W W W"
        "W W W W W W"
        "R R W W R R"
        "R R W W R R");
    Board b2_2 = Board(
        "R B B B B B"
        "B R R R B B"
        "R R O O R B"
        "B R O O B B"
        "B R R R B B"
        "R B B B B B");
    Board s2_2 = Board(
        "B B B B B B"
        "B R R R R B"
        "B R O O R B"
        "B R O O R B"
        "B R R R R B"
        "B B B B B B");
    Board b2_3 = Board(
        "B W W B W B"
        "W B B W B W"
        "W W W B W B"
        "W B B W B W"
        "B W B B W B"
        "B W B W W B");
    Board s2_3 = Board(
        "B B B B B B"
        "W W W W W W"
        "B B B B B B"
        "W W W W W W"
        "B B B B B B"
        "W W W W W W");
    Board b2_4 = Board(
        "G G P Y G G"
        "G Y Y Y G G"
        "G P Y Y Y Y"
        "Y G Y P Y Y"
        "G G Y Y G G"
        "Y G P G G Y");
    Board s2_4 = Board(
        "G G Y Y G G"
        "G G Y Y G G"
        "Y Y P P Y Y"
        "Y Y P P Y Y"
        "G G Y Y G G"
        "G G Y Y G G");
    Board b2_5 = Board(
        "W R R R R W"
        "W W W R R R"
        "W W W R R R"
        "W R W W R R"
        "W W W W W R"
        "W W R W W W");
    Board s2_5 = Board(
        "W R R R R R"
        "W W R R R R"
        "W W W R R R"
        "W W W W R R"
        "W W W W W R"
        "W W W W W W");
    Board b3_1 = Board(
        "W B B B B B"
        "B B B B C B"
        "W C W W W W"
        "W B W W W W"
        "C B C C C C"
        "C W C C C C");
    Board s3_1 = Board(
        "B B B B B B"
        "B B B B B B"
        "W W W W W W"
        "W W W W W W"
        "C C C C C C"
        "C C C C C C");
    Board b3_2 = Board(
        "O Y Y Y Y Y"
        "Y Y O Y Y Y"
        "O Y Y Y Y O"
        "Y Y Y O Y Y"
        "Y Y Y Y Y O"
        "O O Y Y Y Y");
    Board s3_2 = Board(
        "O Y Y Y Y O"
        "Y Y Y Y Y Y"
        "Y Y O O Y Y"
        "Y Y O O Y Y"
        "Y Y Y Y Y Y"
        "O Y Y Y Y O");
    Board b3_3 = Board(
        "R Y O R R R"
        "O Y R R O O"
        "Y O R O Y Y"
        "Y R O Y Y Y"
        "O R Y Y O O"
        "Y O R O R R");
    Board s3_3 = Board(
        "R R R R R R"
        "O O O O O O"
        "Y Y Y Y Y Y"
        "Y Y Y Y Y Y"
        "O O O O O O"
        "R R R R R R");
    Board b3_4 = Board(
        "B G G B P G"
        "P B G P B P"
        "B P P B G G"
        "G B G G B P"
        "P B P P B G"
        "G B P P B G");
    Board s3_4 = Board(
        "G B P P B G"
        "G B P P B G"
        "G B P P B G"
        "G B P P B G"
        "G B P P B G"
        "G B P P B G");
    Board b3_5 = Board(
        "W R O W W W"
        "W O O O O W"
        "R R O O O O"
        "O R R O O R"
        "W O O R O W"
        "W W R O W W");
    Board s3_5 = Board(
        "W W R R W W"
        "W O O O O W"
        "R O O O O R"
        "R O O O O R"
        "W O O O O W"
        "W W R R W W");
    Board b4_1 = Board(
        "W W R W R W"
        "W W R W R W"
        "W R W R R R"
        "W R W W R W"
        "W R W R W W"
        "W R R W R W");
    Board s4_1 = Board(
        "W W W W W W"
        "W R R R R W"
        "W R W W R W"
        "W R W R R W"
        "W R W W W W"
        "W R R R R R");
    Board b4_2 = Board(
        "B Y Y Y Y Y"
        "Y Y Y Y Y B"
        "B Y B Y B Y"
        "Y B B B Y B"
        "B Y B Y - -"
        "Y Y Y B - -");
    Board s4_2 = Board(
        "Y Y Y Y B Y"
        "Y B B B Y Y"
        "B Y Y B Y Y"
        "B Y Y B B Y"
        "Y B B B - -"
        "Y Y Y Y - -");
    Board b4_3 = Board(
        "B W B B B W"
        "B B B B B B"
        "W B B B W B"
        "B B B B B B"
        "B W B B B W"
        "B B W W B B");
    Board s4_3 = Board(
        "B B B B B B"
        "B W B W B B"
        "B B W B W B"
        "B W B W B B"
        "B B W B W B"
        "B B B B B B");
    // FAT
    Board b4_4 = Board(
    "W W W W W R"
    "R R W R W W"
    "W W - - R R"
    "R W - - R R"
    "W R W W W W"
    "R W W R W W");
    Board s4_4 = Board(
        "W W R R W W"
        "W R W W R W"
        "R W - - W R"
        "R W - - W R"
        "W R W W R W"
        "W W R R W W");
    Board b4_5 = Board(
        "P P P P G P"
        "P G G P P G"
        "G P P P G P"
        "P P G G P G"
        "G P P G G G"
        "P G G G G G");
    Board s4_5 = Board(
        "P G P G P G"
        "P G P G P G"
        "P G P G P G"
        "P G P G P G"
        "P G P G P G"
        "P G P G P G");
    // FAT
    Board b5_1 = Board(
        "B B B W W W"
        "W W W B B W"
        "W W B W W W"
        "B B W - - W"
        "W W B - - W"
        "B B W W W B");
    Board s5_1 = Board(
        "W W W W W W"
        "W B B B B W"
        "W B B B B W"
        "W B B - - W"
        "W B B - - W"
        "W W W W W W");
    Board b5_2 = Board(
        "R Y R R R R"
        "R O Y R O Y"
        "R O O Y O R"
        "Y Y Y Y R O"
        "R R O Y R Y"
        "O O Y R R R");
    Board s5_2 = Board(
        "R R O Y R R"
        "R R O Y R R"
        "O O O Y O O"
        "Y Y Y Y Y Y"
        "R R O Y R R"
        "R R O Y R R");
    Board b5_3 = Board(
        "R R G R G R"
        "G G R G Y G"
        "G R G Y R R"
        "Y R G G R Y"
        "R R R R R Y"
        "G R G G G Y");
    Board s5_3 = Board(
        "R R R R R R"
        "G G G G G G"
        "R R G G R R"
        "Y R G G R Y"
        "Y R G G R Y"
        "Y R G G R Y");
    Board b5_4 = Board(
        "Y Y Y G Y Y"
        "G G Y Y Y G"
        "G G G Y G Y"
        "Y Y G Y Y G"
        "Y Y Y Y G Y"
        "G Y G Y G G");
    Board s5_4 = Board(
        "Y Y Y Y Y Y"
        "G G G G G Y"
        "Y Y Y Y G Y"
        "G G G Y G Y"
        "Y Y G Y G Y"
        "G Y G Y G Y");
    Board b5_5 = Board(
        "O Y Y Y R R"
        "Y Y Y O R O"
        "R O O Y O R"
        "O O R R R Y"
        "R O R O Y Y"
        "R Y O O Y R");
    Board s5_5 = Board(
        "Y Y Y Y Y Y"
        "R Y Y Y Y R"
        "R R Y Y R R"
        "R R O O R R"
        "R O O O O R"
        "O O O O O O");
    // FAT
    Board b6_1 = Board(
        "O Y R Y R O"
        "O R R O O O"
        "Y R - - R Y"
        "R O - - Y O"
        "R R Y Y O R"
        "Y O R R O O");
    Board s6_1 = Board(
        "R R O O R R"
        "R O Y Y O R"
        "O Y - - Y O"
        "O Y - - Y O"
        "R O Y Y O R"
        "R R O O R R");
    // FAT
    Board b6_2 = Board(
        "B Y W O O O"
        "Y Y O Y W Y"
        "W O - - O W"
        "O O - - B O"
        "Y W Y B Y O"
        "B W W W O O");
    Board s6_2 = Board(
        "W B O O Y W"
        "Y W O Y W B"
        "O Y - - O O"
        "O O - - Y O"
        "B W Y O W Y"
        "W Y O O B W");
    // FAT
    Board s6_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    // FAT
    Board b6_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s6_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    // FAT
    Board b6_5 = Board(
        "B W C W W W"
        "B W W B W C"
        "C W - - W W"
        "W W - - W C"
        "W W C B C C"
        "B B W B C B");
    Board s6_5 = Board(
        "B W C C W B"
        "W B W W B W"
        "C W - - W C"
        "C W - - W C"
        "W B W W B W"
        "B W C C W B");

    Board b7_1 = Board(
        "B B B B W C"
        "B C W C B C"
        "W W C B W W"
        "B B B C B C"
        "W W C W B B"
        "C C B C W C");
    Board s7_1 = Board(
        "B W B C C C"
        "W B W B C C"
        "B W B W B C"
        "C B W B W B"
        "C C B W B W"
        "C C C B W B");
    Board b7_2 = Board(
        "Y Y R O Y Y"
        "Y Y R O O Y"
        "R Y R R Y Y"
        "Y Y Y O O Y"
        "Y R O R R O"
        "Y Y Y Y O Y");
    Board s7_2 = Board(
        "R R Y Y O O"
        "R R Y Y O O"
        "Y Y Y Y Y Y"
        "Y Y Y Y Y Y"
        "O O Y Y R R"
        "O O Y Y R R");
    Board b7_3 = Board(
        "G G W G G G"
        "W G W W W W"
        "W W G W W G"
        "G G W W W W"
        "G G G G W W"
        "W G G G W W");
    Board s7_3 = Board(
        "W W W W W G"
        "W G G G W G"
        "W G W G W G"
        "W W W G W G"
        "G G G G W G"
        "G W W W W G");
    Board b7_4 = Board(
        "B Y Y B G B"
        "Y Y B G Y Y"
        "Y Y G B B G"
        "B G Y G B G"
        "Y G G G Y B"
        "G B B Y B G");
    Board s7_4 = Board(
        "Y Y Y Y Y Y"
        "G G G G G G"
        "B B B B B B"
        "Y Y Y Y Y Y"
        "G G G G G G"
        "B B B B B B");
    Board b7_5 = Board(
        "R R R W W P"
        "R R P P R P"
        "R W R R R R"
        "W P R R P W"
        "R P R P W W"
        "P P R P P W");
    Board s7_5 = Board(
        "P P P W R R"
        "P P W R R R"
        "P W R R R W"
        "W R R R W P"
        "R R R W P P"
        "R R W P P P");

    Board b8_1 = Board(
        "Y G B B Y G"
        "Y B Y B Y Y"
        "Y Y Y G G B"
        "G B G Y B G"
        "B Y G G G B"
        "B G B G Y B");
    Board s8_1 = Board(
        "B B Y Y B B"
        "B Y G G Y B"
        "Y G G G G Y"
        "Y G G G G Y"
        "B Y G G Y B"
        "B B Y Y B B");
    // FAT
    Board b8_2 = Board(
        "W B W B B W"
        "W W B W W W"
        "W W - - B B"
        "B B - - W B"
        "W W W B W W"
        "W B W B W W");
    Board s8_2 = Board(
        "B W B B W B"
        "W W W W W W"
        "B W - - W B"
        "B W - - W B"
        "W W W W W W"
        "B W B B W B");
    Board b8_3 = Board(
        "B W W W W B"
        "G G G W G W"
        "W W B W B W"
        "W G W B G G"
        "B B B W G B"
        "W B G G G G");
    Board s8_3 = Board(
        "W W W G G G"
        "W B B B G G"
        "W B W W B G"
        "G B W W B W"
        "G G B B B W"
        "G G G W W W");
    // FAT
    Board b8_4 = Board(
        "O R R R R R"
        "O O O O O O"
        "R O - - O O"
        "R O - - R O"
        "O R R O R O"
        "O R R R R R");
    Board s8_4 = Board(
        "R R R R R O"
        "R O O O R O"
        "R O - - R O"
        "R O - - R O"
        "R O R R R O"
        "R O O O O O");
    Board b8_5 = Board(
        "G G W W G W"
        "W W W G W G"
        "G W G G G W"
        "W G G G G W"
        "W W W G G W"
        "W G G G W W");
    Board s8_5 = Board(
        "G W G W G W"
        "W G W G W G"
        "G W G W G W"
        "W G W G W G"
        "G W G W G W"
        "W G W G W G");
    // FAT
    Board b9_1 = Board(
        "B B B W B B"
        "B B W W B W"
        "W W - - B W"
        "B B - - B W"
        "B W B W B W"
        "B B W B B B");
    Board s9_1 = Board(
        "B B W W B B"
        "B W B B W B"
        "W B - - B W"
        "W B - - B W"
        "B W B B W B"
        "B B W W B B");
    Board b9_2 = Board(
        "W R W R W W"
        "R R R B R R"
        "R W B B R W"
        "W W B W B B"
        "W B R R B R"
        "B B W B W B");
    Board s9_2 = Board(
        "B B W W R R"
        "B B W W R R"
        "B B W W R R"
        "B B W W R R"
        "B B W W R R"
        "B B W W R R");
    Board b9_3 = Board(
        "R G G R G W"
        "G R O W Y G"
        "R R W W R R"
        "W W R G G G"
        "O Y G G R R"
        "G R G W W R");
    Board s9_3 = Board(
        "R R R R R R"
        "R R R R R R"
        "W W O Y W W"
        "W W Y O W W"
        "G G G G G G"
        "G G G G G G");
    Board b9_4 = Board(
        "W W W B W W"
        "B B W B B W"
        "W W W B W W"
        "W W W B W B"
        "W W W W B W"
        "B W B W B W");
    Board s9_4 = Board(
        "W W B B W W"
        "W B W W B W"
        "B W W W W B"
        "B W W W W B"
        "W B W W B W"
        "W W B B W W");
    Board b9_5 = Board(
        "W W B W R W"
        "W R W W W W"
        "R W W B B W"
        "W B W R B W"
        "W W W B W R"
        "W W R W W W");
    Board s9_5 = Board(
        "W W W W W W"
        "W W R R W W"
        "W R R B B W"
        "W R R B B W"
        "W W B B W W"
        "W W W W W W");

    Board b10_1 = Board(
        "G Y R R R G"
        "G G G Y Y R"
        "R Y Y R G R"
        "G G Y G R Y"
        "G Y G R G G"
        "R Y R Y G G");
    Board s10_1 = Board(
        "R R Y Y Y Y"
        "R R R Y Y Y"
        "R R R G Y Y"
        "R R G G G Y"
        "R G G G G G"
        "G G G G G G");
    Board b10_2 = Board(
        "O O O G Y Y"
        "Y Y O R R G"
        "O R O G R Y"
        "R Y G Y G G"
        "O Y R R G G"
        "O O G R R Y");
    Board s10_2 = Board(
        "R R R O O O"
        "R R R O O O"
        "R R R O O O"
        "G G G Y Y Y"
        "G G G Y Y Y"
        "G G G Y Y Y");
    Board b10_3 = Board(
        "P P Y R P P"
        "P P O O R R"
        "R O P Y P R"
        "Y P O P Y R"
        "P P P Y O P"
        "P O P P P Y");
    Board s10_3 = Board(
        "O O O O O O"
        "R R R R R R"
        "Y Y Y Y Y Y"
        "P P P P P P"
        "P P P P P P"
        "P P P P P P");
    Board b10_4 = Board(
        "C B W B B C"
        "B W C C B C"
        "W C C C C C"
        "B B C C C B"
        "C W B B C W"
        "C C C C B B");
    Board s10_4 = Board(
        "C C C C C C"
        "C B B B B C"
        "C B W W B C"
        "C B W W B C"
        "C B B W B C"
        "C C C C B C");
    Board b10_5 = Board(
        "R P R B B B"
        "P B R R P B"
        "P R R R P P "
        "B R B R R R"
        "P P R R B B"
        "R R R R R P");
    Board s10_5 = Board(
        "R R R B B B"
        "R R R B B B"
        "R R R B B B"
        "R R R P P P"
        "R R R P P P"
        "R R R P P P");

    Board b11_1 = Board(
        "W W W R W B"
        "W B R R R B"
        "W W W R B R"
        "B W R R B W"
        "W R W W R W"
        "B R W R B B");
    Board s11_1 = Board(
        "B B B R R R"
        "B B B W W W"
        "B B B R R R"
        "W W W W W W"
        "R R R R R R"
        "W W W W W W");
    Board b11_2 = Board(
        "O B O B O Y"
        "B Y B O O B"
        "Y O Y B Y Y"
        "B Y Y O O B"
        "B B B Y Y Y"
        "O O Y B O O");
    Board s11_2 = Board(
        "B B B B B B"
        "B B B B B B"
        "O O O O O O"
        "O O O O O O"
        "Y Y Y Y Y Y"
        "Y Y Y Y Y Y");
    Board b11_3 = Board(
        "W W R R W W"
        "R R R W W W"
        "R W R W R W"
        "W W W W W W"
        "W R R W R W"
        "W W W W W R");
    Board s11_3 = Board(
        "W W W W W W"
        "W W R R W W"
        "W R R R R W"
        "W R R R R W"
        "W W R R W W"
        "W W W W W W");
    Board b11_4 = Board(
        "W B B W W W"
        "W W B W W B"
        "W W B B W B"
        "W W B B W B"
        "B B W W B B"
        "W W B C C B");
    Board s11_4 = Board(
        "B W B B B B"
        "W W W C W W"
        "B W B B B B"
        "W C W W W W"
        "B B B B B B"
        "W W W W W W");
    Board b11_5 = Board(
        "W R R B R R"
        "R R R R W R"
        "R W B R B R"
        "R R W W R W"
        "W R R R R B"
        "W R B B B B");
    Board s11_5 = Board(
        "W B R R B W"
        "B W R R W B"
        "R R R R R R"
        "R R R R R R"
        "B W R R W B"
        "W B R R B W");

    Board b12_1 = Board(
        "R G Y O B R"
        "R P B Y O O"
        "Y Y Y B P O"
        "G G B P G R"
        "B G G O B P"
        "O Y R R P P");
    Board s12_1 = Board(
        "R R R R R R"
        "O O O O O O"
        "Y Y Y Y Y Y"
        "G G G G G G"
        "B B B B B B"
        "P P P P P P");
    Board b12_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s12_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b12_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s12_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b12_4 = Board(
        "B C W C B B"
        "W C W W W B"
        "B B W B B C"
        "B C W B B C"
        "W W W B B B"
        "W B B B W B");
    Board s12_4 = Board(
        "B B B W B B"
        "B B W C W B"
        "B W C W C W"
        "W C W B W C"
        "C W B B B W"
        "W B B B B B");
    Board b12_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s12_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");

    Board b13_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s13_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b13_2 = Board(
        "G W W W Y W"
        "W W W W Y Y"
        "Y W Y W W Y"
        "W W G Y Y G"
        "G W Y Y W W"
        "Y W W Y W W");
    Board s13_2 = Board(
        "W W W W W W"
        "W Y Y Y Y W"
        "W Y G G Y W"
        "W Y G G Y W"
        "W Y Y Y Y W"
        "W W W W W W");
    Board b13_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s13_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b13_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s13_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b13_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s13_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");

    Board b14_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s14_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b14_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s14_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b14_3 = Board(
        "R W B R W W"
        "W G O R G W"
        "B Y W W P P"
        "Y Y P R G P"
        "W B W G B O"
        "W O O W W Y");
    Board s14_3 = Board(
        "R O Y G B P"
        "R O Y G B P"
        "W W W W W W"
        "W W W W W W"
        "P B G Y O R"
        "P B G Y O R");
    Board b14_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s14_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b14_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s14_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");

    Board b15_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s15_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b15_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s15_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b15_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s15_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b15_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s15_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b15_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s15_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");

    Board b16_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s16_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b16_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s16_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b16_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s16_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b16_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s16_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b16_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s16_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");

    Board b17_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s17_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b17_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s17_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b17_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s17_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b17_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s17_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b17_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s17_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");

    Board b18_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s18_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b18_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s18_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b18_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s18_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b18_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s18_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b18_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s18_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");

    Board b19_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s19_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b19_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s19_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b19_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s19_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b19_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s19_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b19_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s19_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");

    Board b20_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s20_1 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b20_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s20_2 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b20_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s20_3 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b20_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s20_4 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board b20_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
    Board s20_5 = Board(
        ""
        ""
        ""
        ""
        ""
        "");
};



