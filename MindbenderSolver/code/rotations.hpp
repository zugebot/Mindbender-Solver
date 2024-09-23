#pragma once

#include "board.hpp"

#include <map>


void R_0_1(Board &board);
void R_0_2(Board &board);
void R_0_3(Board &board);
void R_0_4(Board &board);
void R_0_5(Board &board);

void R_1_1(Board &board);
void R_1_2(Board &board);
void R_1_3(Board &board);
void R_1_4(Board &board);
void R_1_5(Board &board);

void R_2_1(Board &board);
void R_2_2(Board &board);
void R_2_3(Board &board);
void R_2_4(Board &board);
void R_2_5(Board &board);

void R_3_1(Board &board);
void R_3_2(Board &board);
void R_3_3(Board &board);
void R_3_4(Board &board);
void R_3_5(Board &board);

void R_4_1(Board &board);
void R_4_2(Board &board);
void R_4_3(Board &board);
void R_4_4(Board &board);
void R_4_5(Board &board);

void R_5_1(Board &board);
void R_5_2(Board &board);
void R_5_3(Board &board);
void R_5_4(Board &board);
void R_5_5(Board &board);

void C_0_1(Board &board);
void C_0_2(Board &board);
void C_0_3(Board &board);
void C_0_4(Board &board);
void C_0_5(Board &board);

void C_1_1(Board &board);
void C_1_2(Board &board);
void C_1_3(Board &board);
void C_1_4(Board &board);
void C_1_5(Board &board);

void C_2_1(Board &board);
void C_2_2(Board &board);
void C_2_3(Board &board);
void C_2_4(Board &board);
void C_2_5(Board &board);

void C_3_1(Board &board);
void C_3_2(Board &board);
void C_3_3(Board &board);
void C_3_4(Board &board);
void C_3_5(Board &board);

void C_4_1(Board &board);
void C_4_2(Board &board);
void C_4_3(Board &board);
void C_4_4(Board &board);
void C_4_5(Board &board);

void C_5_1(Board &board);
void C_5_2(Board &board);
void C_5_3(Board &board);
void C_5_4(Board &board);
void C_5_5(Board &board);


// for the memory, only make it remember the first rotation
void R_01_1(Board& board);
void R_01_2(Board& board);
void R_01_3(Board& board);
void R_01_4(Board& board);
void R_01_5(Board& board);
void R_12_1(Board& board);
void R_12_2(Board& board);
void R_12_3(Board& board);
void R_12_4(Board& board);
void R_12_5(Board& board);
void R_23_1(Board& board);
void R_23_2(Board& board);
void R_23_3(Board& board);
void R_23_4(Board& board);
void R_23_5(Board& board);
void R_34_1(Board& board);
void R_34_2(Board& board);
void R_34_3(Board& board);
void R_34_4(Board& board);
void R_34_5(Board& board);
void R_45_1(Board& board);
void R_45_2(Board& board);
void R_45_3(Board& board);
void R_45_4(Board& board);
void R_45_5(Board& board);

void C_01_1(Board& board);
void C_01_2(Board& board);
void C_01_3(Board& board);
void C_01_4(Board& board);
void C_01_5(Board& board);
void C_12_1(Board& board);
void C_12_2(Board& board);
void C_12_3(Board& board);
void C_12_4(Board& board);
void C_12_5(Board& board);
void C_23_1(Board& board);
void C_23_2(Board& board);
void C_23_3(Board& board);
void C_23_4(Board& board);
void C_23_5(Board& board);
void C_34_1(Board& board);
void C_34_2(Board& board);
void C_34_3(Board& board);
void C_34_4(Board& board);
void C_34_5(Board& board);
void C_45_1(Board& board);
void C_45_2(Board& board);
void C_45_3(Board& board);
void C_45_4(Board& board);
void C_45_5(Board& board);


typedef void (*Action)(Board &);

MU extern std::map<void (*)(Board &), std::string> RCNameForwardLookup;
MU extern std::map<void (*)(Board &), std::string> RCNameBackwardLookup;

MU extern void (*actions[60])(Board &);
MU extern void (*fatActions[25][48])(Board &);







