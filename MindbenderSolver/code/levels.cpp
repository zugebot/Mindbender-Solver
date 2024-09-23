#include "levels.hpp"


class LevelCells {
public:
    static constexpr u8 b1_1[36] = {1,1,0,0,1,1, 1,1,0,0,1,1, 0,0,0,1,1,1, 0,0,1,1,1,0, 1,1,0,0,0,1, 1,1,0,0,0,1};
    static constexpr u8 s1_1[36] = {1,1,0,0,1,1, 1,1,0,0,1,1, 0,0,1,1,0,0, 0,0,1,1,0,0, 1,1,0,0,1,1, 1,1,0,0,1,1};
    static constexpr u8 b1_2[36] = {2,3,2,2,2,2, 3,3,3,3,2,2, 2,2,2,2,3,2, 2,2,2,2,3,2, 2,3,3,3,3,2, 2,3,2,2,2,2};
    static constexpr u8 s1_2[36] = {2,2,2,2,2,2, 2,3,3,3,3,2, 2,3,2,2,3,2, 2,3,2,2,3,2, 2,3,3,3,3,2, 2,2,2,2,2,2};
    static constexpr u8 b1_3[36] = {4,4,5,4,4,4, 4,5,5,5,5,4, 4,5,5,5,5,4, 4,5,4,5,5,4, 4,5,4,5,5,4, 4,4,5,4,4,4};
    static constexpr u8 s1_3[36] = {4,4,4,4,4,4, 4,5,5,5,5,4, 4,5,5,5,5,4, 4,5,5,5,5,4, 4,5,5,5,5,4, 4,4,4,4,4,4};
    static constexpr u8 b1_4[36] = {0,4,0,0,0,0, 4,4,0,0,4,0, 4,0,4,4,0,0, 0,0,4,4,0,0, 0,4,0,0,4,0, 0,0,4,0,0,0};
    static constexpr u8 s1_4[36] = {4,0,0,0,0,4, 0,4,0,0,4,0, 0,0,4,4,0,0, 0,0,4,4,0,0, 0,4,0,0,4,0, 4,0,0,0,0,4};
    static constexpr u8 b1_5[36] = {7,7,7,7,7,7, 7,2,2,2,7,7, 7,2,7,7,2,7, 7,2,7,7,2,7, 2,7,2,2,2,2, 7,7,7,7,7,7};
    static constexpr u8 s1_5[36] = {7,7,7,7,7,7, 7,2,2,2,2,7, 7,2,7,7,2,7, 7,2,7,7,2,7, 7,2,2,2,2,7, 7,7,7,7,7,7};
    static constexpr u8 b2_1[36] = {6,0,0,0,0,6, 0,0,6,6,0,0, 6,6,6,6,6,0, 6,6,6,6,0,6, 0,0,6,6,6,0, 0,0,6,6,6,0};
    static constexpr u8 s2_1[36] = {0,0,6,6,0,0, 0,0,6,6,0,0, 6,6,6,6,6,6, 6,6,6,6,6,6, 0,0,6,6,0,0, 0,0,6,6,0,0};
    static constexpr u8 b2_2[36] = {0,2,2,2,2,2, 2,0,0,0,2,2, 0,0,3,3,0,2, 2,0,3,3,2,2, 2,0,0,0,2,2, 0,2,2,2,2,2};
    static constexpr u8 s2_2[36] = {2,2,2,2,2,2, 2,0,0,0,0,2, 2,0,3,3,0,2, 2,0,3,3,0,2, 2,0,0,0,0,2, 2,2,2,2,2,2};
    static constexpr u8 b2_3[36] = {2,6,6,2,6,2, 6,2,2,6,2,6, 6,6,6,2,6,2, 6,2,2,6,2,6, 2,6,2,2,6,2, 2,6,2,6,6,2};
    static constexpr u8 s2_3[36] = {2,2,2,2,2,2, 6,6,6,6,6,6, 2,2,2,2,2,2, 6,6,6,6,6,6, 2,2,2,2,2,2, 6,6,6,6,6,6};
    static constexpr u8 b2_4[36] = {1,1,5,4,1,1, 1,4,4,4,1,1, 1,5,4,4,4,4, 4,1,4,5,4,4, 1,1,4,4,1,1, 4,1,5,1,1,4};
    static constexpr u8 s2_4[36] = {1,1,4,4,1,1, 1,1,4,4,1,1, 4,4,5,5,4,4, 4,4,5,5,4,4, 1,1,4,4,1,1, 1,1,4,4,1,1};
    static constexpr u8 b2_5[36] = {6,0,0,0,0,6, 6,6,6,0,0,0, 6,6,6,0,0,0, 6,0,6,6,0,0, 6,6,6,6,6,0, 6,6,0,6,6,6};
    static constexpr u8 s2_5[36] = {6,0,0,0,0,0, 6,6,0,0,0,0, 6,6,6,0,0,0, 6,6,6,6,0,0, 6,6,6,6,6,0, 6,6,6,6,6,6};
    static constexpr u8 b3_1[36] = {6,2,2,2,2,2, 2,2,2,2,7,2, 6,7,6,6,6,6, 6,2,6,6,6,6, 7,2,7,7,7,7, 7,6,7,7,7,7};
    static constexpr u8 s3_1[36] = {2,2,2,2,2,2, 2,2,2,2,2,2, 6,6,6,6,6,6, 6,6,6,6,6,6, 7,7,7,7,7,7, 7,7,7,7,7,7};
    static constexpr u8 b3_2[36] = {3,4,4,4,4,4, 4,4,3,4,4,4, 3,4,4,4,4,3, 4,4,4,3,4,4, 4,4,4,4,4,3, 3,3,4,4,4,4};
    static constexpr u8 s3_2[36] = {3,4,4,4,4,3, 4,4,4,4,4,4, 4,4,3,3,4,4, 4,4,3,3,4,4, 4,4,4,4,4,4, 3,4,4,4,4,3};
    static constexpr u8 b3_3[36] = {0,4,3,0,0,0, 3,4,0,0,3,3, 4,3,0,3,4,4, 4,0,3,4,4,4, 3,0,4,4,3,3, 4,3,0,3,0,0};
    static constexpr u8 s3_3[36] = {0,0,0,0,0,0, 3,3,3,3,3,3, 4,4,4,4,4,4, 4,4,4,4,4,4, 3,3,3,3,3,3, 0,0,0,0,0,0};
    static constexpr u8 b3_4[36] = {2,1,1,2,5,1, 5,2,1,5,2,5, 2,5,5,2,1,1, 1,2,1,1,2,5, 5,2,5,5,2,1, 1,2,5,5,2,1};
    static constexpr u8 s3_4[36] = {1,2,5,5,2,1, 1,2,5,5,2,1, 1,2,5,5,2,1, 1,2,5,5,2,1, 1,2,5,5,2,1, 1,2,5,5,2,1};
    static constexpr u8 b3_5[36] = {6,0,3,6,6,6, 6,3,3,3,3,6, 0,0,3,3,3,3, 3,0,0,3,3,0, 6,3,3,0,3,6, 6,6,0,3,6,6};
    static constexpr u8 s3_5[36] = {6,6,0,0,6,6, 6,3,3,3,3,6, 0,3,3,3,3,0, 0,3,3,3,3,0, 6,3,3,3,3,6, 6,6,0,0,6,6};
    static constexpr u8 b4_1[36] = {6,6,0,6,0,6, 6,6,0,6,0,6, 6,0,6,0,0,0, 6,0,6,6,0,6, 6,0,6,0,6,6, 6,0,0,6,0,6};
    static constexpr u8 s4_1[36] = {6,6,6,6,6,6, 6,0,0,0,0,6, 6,0,6,6,0,6, 6,0,6,0,0,6, 6,0,6,6,6,6, 6,0,0,0,0,0};
    static constexpr u8 b4_2[36] = {2,4,4,4,4,4, 4,4,4,4,4,2, 2,4,2,4,2,4, 4,2,2,2,4,2, 2,4,2,4,6,6, 4,4,4,2,6,6};
    static constexpr u8 s4_2[36] = {4,4,4,4,4,4, 4,2,2,2,2,4, 4,2,6,6,2,4, 4,2,6,6,2,4, 4,2,2,2,2,4, 4,4,4,4,4,4};
    static constexpr u8 b4_3[36] = {2,6,2,2,2,6, 2,2,2,2,2,2, 6,2,2,2,6,2, 2,2,2,2,2,2, 2,6,2,2,2,6, 2,2,6,6,2,2};
    static constexpr u8 s4_3[36] = {2,2,2,2,2,2, 2,6,2,6,2,2, 2,2,6,2,6,2, 2,6,2,6,2,2, 2,2,6,2,6,2, 2,2,2,2,2,2};
    static constexpr u8 b4_4[36] = {6,6,6,6,6,0, 0,0,6,0,6,6, 6,6,6,6,0,0, 0,6,8,8,0,0, 6,0,6,6,6,6, 0,6,6,0,6,6};
    static constexpr u8 s4_4[36] = {6,6,0,0,6,6, 6,0,6,6,0,6, 0,6,6,6,6,0, 0,6,8,8,6,0, 6,0,6,6,0,6, 6,6,0,0,6,6};
    static constexpr u8 b4_5[36] = {5,5,5,5,1,5, 5,1,1,5,5,1, 1,5,5,5,1,5, 5,5,1,1,5,1, 1,5,5,1,1,1, 5,1,1,1,1,1};
    static constexpr u8 s4_5[36] = {5,1,5,1,5,1, 5,1,5,1,5,1, 5,1,5,1,5,1, 5,1,5,1,5,1, 5,1,5,1,5,1, 5,1,5,1,5,1};
    static constexpr u8 b5_1[36] = {2,2,2,6,6,6, 6,6,6,2,2,6, 6,6,2,6,6,6, 2,2,6,8,8,6, 6,6,2,8,8,6, 2,2,6,6,6,2};
    static constexpr u8 s5_1[36] = {6,6,6,6,6,6, 6,2,2,2,2,6, 6,2,2,2,2,6, 6,2,2,7,7,6, 6,2,2,7,7,6, 6,6,6,6,6,6};
    static constexpr u8 b5_2[36] = {0,4,0,0,0,0, 0,3,4,0,3,4, 0,3,3,4,3,0, 4,4,4,4,0,3, 0,0,3,4,0,4, 3,3,4,0,0,0};
    static constexpr u8 s5_2[36] = {0,0,3,4,0,0, 0,0,3,4,0,0, 3,3,3,4,3,3, 4,4,4,4,4,4, 0,0,3,4,0,0, 0,0,3,4,0,0};
    static constexpr u8 b5_3[36] = {0,0,1,0,1,0, 1,1,0,1,4,1, 1,0,1,4,0,0, 4,0,1,1,0,4, 0,0,0,0,0,4, 1,0,1,1,1,4};
    static constexpr u8 s5_3[36] = {0,0,0,0,0,0, 1,1,1,1,1,1, 0,0,1,1,0,0, 4,0,1,1,0,4, 4,0,1,1,0,4, 4,0,1,1,0,4};
    static constexpr u8 b5_4[36] = {4,4,4,1,4,4, 1,1,4,4,4,1, 1,1,1,4,1,4, 4,4,1,4,4,1, 4,4,4,4,1,4, 1,4,1,4,1,1};
    static constexpr u8 s5_4[36] = {4,4,4,4,4,4, 1,1,1,1,1,4, 4,4,4,4,1,4, 1,1,1,4,1,4, 4,4,1,4,1,4, 1,4,1,4,1,4};
    static constexpr u8 b5_5[36] = {3,4,4,4,0,0, 4,4,4,3,0,3, 0,3,3,4,3,0, 3,3,0,0,0,4, 0,3,0,3,4,4, 0,4,3,3,4,0};
    static constexpr u8 s5_5[36] = {4,4,4,4,4,4, 0,4,4,4,4,0, 0,0,4,4,0,0, 0,0,3,3,0,0, 0,3,3,3,3,0, 3,3,3,3,3,3};
    static constexpr u8 b6_1[36] = {3,4,0,4,3,0, 3,0,0,3,3,3, 0,4,4,0,0,3, 4,3,0,3,3,3, 0,0,4,4,6,6, 4,3,0,0,6,6};
    static constexpr u8 s6_1[36] = {0,0,3,3,0,0, 0,3,4,4,3,0, 3,4,6,6,4,3, 3,4,6,6,4,3, 0,3,4,4,3,0, 0,0,3,3,0,0};
    static constexpr u8 b6_2[36] = {2,4,6,6,3,3, 4,4,6,3,6,4, 6,3,3,4,3,6, 3,3,3,2,3,3, 6,3,3,4,3,4, 2,6,4,2,3,3};
    static constexpr u8 s6_2[36] = {6,2,3,3,4,6, 4,6,3,4,6,2, 3,4,3,3,3,3, 3,3,3,3,4,3, 2,6,4,3,6,4, 6,4,3,3,2,6};
    static constexpr u8 b6_3[36] = {3,3,5,2,2,3, 6,6,5,2,3,4, 6,6,2,3,5,5, 4,3,4,5,3,5, 5,2,4,4,4,5, 3,4,3,3,3,5};
    static constexpr u8 s6_3[36] = {6,6,4,4,3,3, 6,6,4,3,3,5, 4,4,4,3,5,5, 4,3,3,3,5,2, 3,3,5,5,5,2, 3,5,5,2,2,2};
    static constexpr u8 b6_4[36] = {6,6,4,5,4,3, 6,5,3,0,6,3, 0,0,6,6,5,7, 5,4,4,7,7,6, 5,5,7,7,7,7, 5,6,3,7,0,5};
    static constexpr u8 s6_4[36] = {3,5,0,6,7,3, 7,4,5,6,4,5, 6,6,7,7,5,0, 0,5,7,7,6,6, 5,4,6,5,4,7, 3,7,6,0,5,3};
    static constexpr u8 b6_5[36] = {2,6,7,7,7,6, 2,6,6,7,7,7, 6,7,6,2,7,6, 7,6,6,2,7,6, 6,6,7,6,6,7, 2,2,6,2,6,2};
    static constexpr u8 s6_5[36] = {2,6,7,7,6,2, 6,2,6,6,2,6, 7,6,7,7,6,7, 7,6,7,7,6,7, 6,2,6,6,2,6, 2,6,7,7,6,2};
    static constexpr u8 b7_1[36] = {2,2,2,2,6,7, 2,7,6,7,2,7, 6,6,7,2,6,6, 2,2,2,7,2,7, 6,6,7,6,2,2, 7,7,2,7,6,7};
    static constexpr u8 s7_1[36] = {2,6,2,7,7,7, 6,2,6,2,7,7, 2,6,2,6,2,7, 7,2,6,2,6,2, 7,7,2,6,2,6, 7,7,7,2,6,2};
    static constexpr u8 b7_2[36] = {4,4,0,3,4,4, 4,4,0,3,3,4, 0,4,0,0,4,4, 4,4,4,3,3,4, 4,0,3,0,0,3, 4,4,4,4,3,4};
    static constexpr u8 s7_2[36] = {0,0,4,4,3,3, 0,0,4,4,3,3, 4,4,4,4,4,4, 4,4,4,4,4,4, 3,3,4,4,0,0, 3,3,4,4,0,0};
    static constexpr u8 b7_3[36] = {1,1,6,1,1,1, 6,1,6,6,6,6, 6,6,1,6,6,1, 1,1,6,6,6,6, 1,1,1,1,6,6, 6,1,1,1,6,6};
    static constexpr u8 s7_3[36] = {6,6,6,6,6,1, 6,1,1,1,6,1, 6,1,6,1,6,1, 6,6,6,1,6,1, 1,1,1,1,6,1, 1,6,6,6,6,1};
    static constexpr u8 b7_4[36] = {2,4,4,2,1,2, 4,4,2,1,4,4, 4,4,1,2,2,1, 2,1,4,1,2,1, 4,1,1,1,4,2, 1,2,2,4,2,1};
    static constexpr u8 s7_4[36] = {4,4,4,4,4,4, 1,1,1,1,1,1, 2,2,2,2,2,2, 4,4,4,4,4,4, 1,1,1,1,1,1, 2,2,2,2,2,2};
    static constexpr u8 b7_5[36] = {0,0,0,6,6,5, 0,0,5,5,0,5, 0,6,0,0,0,0, 6,5,0,0,5,6, 0,5,0,5,6,6, 5,5,0,5,5,6};
    static constexpr u8 s7_5[36] = {5,5,5,6,0,0, 5,5,6,0,0,0, 5,6,0,0,0,6, 6,0,0,0,6,5, 0,0,0,6,5,5, 0,0,6,5,5,5};
    static constexpr u8 b8_1[36] = {4,1,2,2,4,1, 4,2,4,2,4,4, 4,4,4,1,1,2, 1,2,1,4,2,1, 2,4,1,1,1,2, 2,1,2,1,4,2};
    static constexpr u8 s8_1[36] = {2,2,4,4,2,2, 2,4,1,1,4,2, 4,1,1,1,1,4, 4,1,1,1,1,4, 2,4,1,1,4,2, 2,2,4,4,2,2};
    static constexpr u8 b8_2[36] = {6,2,6,2,6,6, 6,6,2,6,6,6, 2,2,6,6,2,6, 6,2,2,2,6,6, 6,6,6,2,2,2, 6,2,6,2,2,2};
    static constexpr u8 s8_2[36] = {2,6,2,2,6,2, 6,6,6,6,6,6, 2,6,2,2,6,2, 2,6,2,2,6,2, 6,6,6,6,6,6, 2,6,2,2,6,2};
    static constexpr u8 b8_3[36] = {2,6,6,6,6,2, 1,1,1,6,1,6, 6,6,2,6,2,6, 6,1,6,2,1,1, 2,2,2,6,1,2, 6,2,1,1,1,1};
    static constexpr u8 s8_3[36] = {6,6,6,1,1,1, 6,2,2,2,1,1, 6,2,6,6,2,1, 1,2,6,6,2,6, 1,1,2,2,2,6, 1,1,1,6,6,6};
    static constexpr u8 b8_4[36] = {3,0,0,3,0,0, 3,3,3,0,0,3, 3,0,3,0,0,3, 3,0,3,3,3,0, 3,0,0,4,4,3, 3,0,0,4,4,0};
    static constexpr u8 s8_4[36] = {0,0,0,0,0,3, 0,3,3,3,0,3, 0,3,4,4,0,3, 0,3,4,4,0,3, 0,3,0,0,0,3, 0,3,3,3,3,3};
    static constexpr u8 b8_5[36] = {1,1,6,6,1,6, 6,6,6,1,6,1, 1,6,1,1,1,6, 6,1,1,1,1,6, 6,6,6,1,1,6, 6,1,1,1,6,6};
    static constexpr u8 s8_5[36] = {1,6,1,6,1,6, 6,1,6,1,6,1, 1,6,1,6,1,6, 6,1,6,1,6,1, 1,6,1,6,1,6, 6,1,6,1,6,1};
    static constexpr u8 b9_1[36] = {2,2,6,2,2,2, 2,2,2,6,2,6, 6,6,6,6,2,6, 2,2,2,2,6,2, 6,2,2,2,6,2, 2,2,2,6,2,2};
    static constexpr u8 s9_1[36] = {2,2,6,6,2,2, 2,6,2,2,6,2, 6,2,2,2,2,6, 6,2,2,2,2,6, 2,6,2,2,6,2, 2,2,6,6,2,2};
    static constexpr u8 b9_2[36] = {6,0,6,0,6,6, 0,0,0,2,0,0, 0,6,2,2,0,6, 6,6,2,6,2,2, 6,2,0,0,2,0, 2,2,6,2,6,2};
    static constexpr u8 s9_2[36] = {2,2,6,6,0,0, 2,2,6,6,0,0, 2,2,6,6,0,0, 2,2,6,6,0,0, 2,2,6,6,0,0, 2,2,6,6,0,0};
    static constexpr u8 b9_3[36] = {0,1,1,0,1,6, 1,0,3,6,4,1, 0,0,6,6,0,0, 6,6,0,1,1,1, 3,4,1,1,0,0, 1,0,1,6,6,0};
    static constexpr u8 s9_3[36] = {0,0,0,0,0,0, 0,0,0,0,0,0, 6,6,3,4,6,6, 6,6,4,3,6,6, 1,1,1,1,1,1, 1,1,1,1,1,1};
    static constexpr u8 b9_4[36] = {6,6,6,2,6,6, 2,2,6,2,2,6, 6,6,6,2,6,6, 6,6,6,2,6,2, 6,6,6,6,2,6, 2,6,2,6,2,6};
    static constexpr u8 s9_4[36] = {6,6,2,2,6,6, 6,2,6,6,2,6, 2,6,6,6,6,2, 2,6,6,6,6,2, 6,2,6,6,2,6, 6,6,2,2,6,6};
    static constexpr u8 b9_5[36] = {6,6,2,6,0,6, 6,0,6,6,6,6, 0,6,6,2,2,6, 6,2,6,0,2,6, 6,6,6,2,6,0, 6,6,0,6,6,6};
    static constexpr u8 s9_5[36] = {6,6,6,6,6,6, 6,6,0,0,6,6, 6,0,0,2,2,6, 6,0,0,2,2,6, 6,6,2,2,6,6, 6,6,6,6,6,6};

    static constexpr u8 b10_1[36] = {1,4,0,0,0,1, 1,1,1,4,4,0, 0,4,4,0,1,0, 1,1,4,1,0,4, 1,4,1,0,1,1, 0,4,0,4,1,1};
    static constexpr u8 s10_1[36] = {0,0,4,4,4,4, 0,0,0,4,4,4, 0,0,0,1,4,4, 0,0,1,1,1,4, 0,1,1,1,1,1, 1,1,1,1,1,1};
    static constexpr u8 b10_2[36] = {3,3,3,1,4,4, 4,4,3,0,0,1, 3,0,3,1,0,4, 0,4,1,4,1,1, 3,4,0,0,1,1, 3,3,1,0,0,4};
    static constexpr u8 s10_2[36] = {0,0,0,3,3,3, 0,0,0,3,3,3, 0,0,0,3,3,3, 1,1,1,4,4,4, 1,1,1,4,4,4, 1,1,1,4,4,4};
    static constexpr u8 b10_3[36] = {5,5,4,0,5,5, 5,5,3,3,0,0, 0,3,5,4,5,0, 4,5,3,5,4,0, 5,5,5,4,3,5, 5,3,5,5,5,4};
    static constexpr u8 s10_3[36] = {3,3,3,3,3,3, 0,0,0,0,0,0, 4,4,4,4,4,4, 5,5,5,5,5,5, 5,5,5,5,5,5, 5,5,5,5,5,5};
    static constexpr u8 b10_4[36] = {7,2,6,2,2,7, 2,6,7,7,2,7, 6,7,7,7,7,7, 2,2,7,7,7,2, 7,6,2,2,7,6, 7,7,7,7,2,2};
    static constexpr u8 s10_4[36] = {7,7,7,7,7,7, 7,2,2,2,2,7, 7,2,6,6,2,7, 7,2,6,6,2,7, 7,2,2,6,2,7, 7,7,7,7,2,7};
    static constexpr u8 b10_5[36] = {0,5,0,2,2,2, 5,2,0,0,5,2, 5,0,0,0,5,5, 2,0,2,0,0,0, 5,5,0,0,2,2, 0,0,0,0,0,5};
    static constexpr u8 s10_5[36] = {0,0,0,2,2,2, 0,0,0,2,2,2, 0,0,0,2,2,2, 0,0,0,5,5,5, 0,0,0,5,5,5, 0,0,0,5,5,5};
    static constexpr u8 b11_1[36] = {6,6,6,0,6,2, 6,2,0,0,0,2, 6,6,6,0,2,0, 2,6,0,0,2,6, 6,0,6,6,0,6, 2,0,6,0,2,2};
    static constexpr u8 s11_1[36] = {2,2,2,0,0,0, 2,2,2,6,6,6, 2,2,2,0,0,0, 6,6,6,6,6,6, 0,0,0,0,0,0, 6,6,6,6,6,6};
    static constexpr u8 b11_2[36] = {3,2,3,2,3,4, 2,4,2,3,3,2, 4,3,4,2,4,4, 2,4,4,3,3,2, 2,2,2,4,4,4, 3,3,4,2,3,3};
    static constexpr u8 s11_2[36] = {2,2,2,2,2,2, 2,2,2,2,2,2, 3,3,3,3,3,3, 3,3,3,3,3,3, 4,4,4,4,4,4, 4,4,4,4,4,4};
    static constexpr u8 b11_3[36] = {6,6,0,0,6,6, 0,0,0,6,6,6, 0,6,0,6,0,6, 6,6,6,6,6,6, 6,0,0,6,0,6, 6,6,6,6,6,0};
    static constexpr u8 s11_3[36] = {6,6,6,6,6,6, 6,6,0,0,6,6, 6,0,0,0,0,6, 6,0,0,0,0,6, 6,6,0,0,6,6, 6,6,6,6,6,6};
    static constexpr u8 b11_4[36] = {6,2,2,6,6,6, 6,6,2,6,6,2, 6,6,2,2,6,2, 6,6,2,2,6,2, 2,2,6,6,2,2, 6,6,2,7,7,2};
    static constexpr u8 s11_4[36] = {2,6,2,2,2,2, 6,6,6,7,6,6, 2,6,2,2,2,2, 6,7,6,6,6,6, 2,2,2,2,2,2, 6,6,6,6,6,6};
    static constexpr u8 b11_5[36] = {6,0,0,2,0,0, 0,0,0,0,6,0, 0,6,2,0,2,0, 0,0,6,6,0,6, 6,0,0,0,0,2, 6,0,2,2,2,2};
    static constexpr u8 s11_5[36] = {6,2,0,0,2,6, 2,6,0,0,6,2, 0,0,0,0,0,0, 0,0,0,0,0,0, 2,6,0,0,6,2, 6,2,0,0,2,6};
    static constexpr u8 b12_1[36] = {0,1,4,3,2,0, 0,5,2,4,3,3, 4,4,4,2,5,3, 1,1,2,5,1,0, 2,1,1,3,2,5, 3,4,0,0,5,5};
    static constexpr u8 s12_1[36] = {0,0,0,0,0,0, 3,3,3,3,3,3, 4,4,4,4,4,4, 1,1,1,1,1,1, 2,2,2,2,2,2, 5,5,5,5,5,5};
    static constexpr u8 b12_2[36] = {0}; // YGYYYY GGYYYG GYGGGY GRRGGG GRRGYY YYGYGY
    static constexpr u8 s12_2[36] = {0}; // GGGYYY GGGYYY GGRRYY YYRRGG YYYGGG YYYGGG
    static constexpr u8 b12_3[36] = {0}; // RRYYOO YWRORR OOYWOO ORYOYW WOYROO OYOORO
    static constexpr u8 s12_3[36] = {0}; // YYOORR YYOORR OOWWOO OOWWOO RROOYY RROOYY
    static constexpr u8 b12_4[36] = {2,7,6,7,2,2,6,7,6,6,6,2,2,2,6,2,2,7,2,7,6,2,2,7,6,6,6,2,2,2,6,2,2,2,6,2};
    static constexpr u8 s12_4[36] = {2,2,2,6,2,2,2,2,6,7,6,2,2,6,7,6,7,6,6,7,6,2,6,7,7,6,2,2,2,6,6,2,2,2,2,2};
    static constexpr u8 b12_5[36] = {0}; // BRYGGG YGBGYR ROOBOG YYBGOG GGRYRO YYORBB
    static constexpr u8 s12_5[36] = {0}; // ROYGBB ROYGGB ROYYGG ROYYGG ROYGGB ROYGBB
    static constexpr u8 b13_1[36] = {0}; // YORPOR POROOO OOYRYO OOORYR RPYOYY OOOYRY
    static constexpr u8 s13_1[36] = {0}; // ROOYYO OOYYOO OYYOOR YYOORR YOORRP OORRPP
    static constexpr u8 b13_2[36] = {1,6,6,6,4,6,6,6,6,6,4,4,4,6,4,6,6,4,6,6,1,4,4,1,1,6,4,4,6,6,4,6,6,4,6,6};
    static constexpr u8 s13_2[36] = {6,6,6,6,6,6,6,4,4,4,4,6,6,4,1,1,4,6,6,4,1,1,4,6,6,4,4,4,4,6,6,6,6,6,6,6};
    static constexpr u8 b13_3[36] = {0}; // CBWBWB BBBCBB WCWCBW BWCBBW BWCCBC CCBBBB
    static constexpr u8 s13_3[36] = {0}; // CBCBCB BWBWBC CBWBWB BWBWBC CBWBWB BCBCBC
    static constexpr u8 b13_4[36] = {0}; // WYYGYY GWYGYG WWWWWY WYGGWW WGGGGW WGGWWW
    static constexpr u8 s13_4[36] = {0}; // YWGGWY WYWWYW GWGGWG GWGGWG WYWWYW YWGGWY
    static constexpr u8 b13_5[36] = {0}; // WWRWRW WRRWRW WWWWWW WRRWWW WWWWWR RWWWWR
    static constexpr u8 s13_5[36] = {0}; // RWRWRW WWWWWR RWWWWW WWWWWR RWWWWW WRWRWR
    static constexpr u8 b14_1[36] = {0}; // YWYWBY BWBWYY BBWBYY BYYWBB BYBYBY WWBBBB
    static constexpr u8 s14_1[36] = {0}; // YBWWBY BBYYBB WYBBYW WYBBYW BBYYBB YBWWBY
    static constexpr u8 b14_2[36] = {0}; // BOBYBY YBRBOO BRBBYR BBOBYO OOBBBB YYROBY
    static constexpr u8 s14_2[36] = {0}; // BBYYBB BBOOBB YORROY YORROY BBOOBB BBYYBB
    static constexpr u8 b14_3[36] = {0,6,2,0,6,6,6,1,3,0,1,6,2,4,6,6,5,5,4,4,5,0,1,5,6,2,6,1,2,3,6,3,3,6,6,4};
    static constexpr u8 s14_3[36] = {0,3,4,1,2,5,0,3,4,1,2,5,6,6,6,6,6,6,6,6,6,6,6,6,5,2,1,4,3,0,5,2,1,4,3,0};
    static constexpr u8 b14_4[36] = {0}; // PBPYWW BBCBGW OWWWWB BBRBRW BOBWBC GYWWWB
    static constexpr u8 s14_4[36] = {0}; // RBWWBR OBWWBO YBWWBY GBWWBG CBWWBC PBWWBP
    static constexpr u8 b14_5[36] = {0}; // PBCBBB BGBROB WCOYRB BCBBBY GCRBBW BBBBOP
    static constexpr u8 s14_5[36] = {0}; // BOBPBG RBCBYB BWBOBC CBRBWB BYBCBO GBPBRB
    static constexpr u8 b15_1[36] = {0}; // YRRYYB GGGPBR GYPORP BYOOBR PGOBOO PPBRYG
    static constexpr u8 s15_1[36] = {0}; // PBBPPB RPPRRP ORROOR YOOYYO GYYGGY BGGBBG
    static constexpr u8 b15_2[36] = {0}; // ROOOOG YYGGRR
    static constexpr u8 s15_2[36] = {0}; //
    static constexpr u8 b15_3[36] = {0}; // BRRCYG BRRWPR RPYOGO YWPYRC RBOOWC WGGBPC
    static constexpr u8 s15_3[36] = {0}; // BBCCGG BBCCGG PPRRYY PPRRYY WWRROO WWRROO
    static constexpr u8 b15_4[36] = {0}; // RGRGGP OPRGOP BOPYRP OOGRYR PBBYYB YBBGOY
    static constexpr u8 s15_4[36] = {0}; // GGYYGG GYOOYG YORROY ORPPRO RPBBPR PBBBBP
    static constexpr u8 b15_5[36] = {0}; //
    static constexpr u8 s15_5[36] = {0}; //
    static constexpr u8 b16_1[36] = {0}; // FAT
    static constexpr u8 s16_1[36] = {0}; // FAT
    static constexpr u8 b16_2[36] = {0}; // WPBWWW BRGWRW OWWWWG PWPWWW YYWWYO GOWRBW
    static constexpr u8 s16_2[36] = {0}; // ROYGBP WWWWWW PBGYOR WWWWWW ROYGBP WWWWWW
    static constexpr u8 b16_3[36] = {0}; //
    static constexpr u8 s16_3[36] = {0}; //
    static constexpr u8 b16_4[36] = {0}; // RWRWWY BWWOYC WWOWWW RWWRWG GWCPBW RWWWPR
    static constexpr u8 s16_4[36] = {0}; // RWCWRW WYWOWB PWGWRW WRWGWP BWOWYW WRWCWR
    static constexpr u8 b16_5[36] = {0}; // FAT
    static constexpr u8 s16_5[36] = {0}; // FAT
    static constexpr u8 b17_1[36] = {0}; //
    static constexpr u8 s17_1[36] = {0}; //
    static constexpr u8 b17_2[36] = {0}; // FAT
    static constexpr u8 s17_2[36] = {0}; // FAT
    static constexpr u8 b17_3[36] = {0}; //
    static constexpr u8 s17_3[36] = {0}; //
    static constexpr u8 b17_4[36] = {0}; // YRYYRO ORRRRO PORROR OOOOOO YPPRYY YROYRP
    static constexpr u8 s17_4[36] = {0}; // YORROY OPYOPO RORRYR RYRROR OPOYPO YORROY
    static constexpr u8 b17_5[36] = {0}; //
    static constexpr u8 s17_5[36] = {0}; //
    static constexpr u8 b18_1[36] = {0}; // FAT
    static constexpr u8 s18_1[36] = {0}; // FAT
    static constexpr u8 b18_2[36] = {0}; // FAT
    static constexpr u8 s18_2[36] = {0}; // FAT
    static constexpr u8 b18_3[36] = {0}; //
    static constexpr u8 s18_3[36] = {0}; //
    static constexpr u8 b18_4[36] = {0}; // FAT
    static constexpr u8 s18_4[36] = {0}; // FAT
    static constexpr u8 b18_5[36] = {0}; // FAT
    static constexpr u8 s18_5[36] = {0}; // FAT
    static constexpr u8 b19_1[36] = {0}; // GCGGGG WCGWWW GGCGCG CGGWCG WWWGGG CGGCWW
    static constexpr u8 s19_1[36] = {0}; // WGWGWG GCGCGW WGCGCG GCGCGW WGCGCG GWGWGW
    static constexpr u8 b19_2[36] = {0}; // FAT
    static constexpr u8 s19_2[36] = {0}; // FAT
    static constexpr u8 b19_3[36] = {0}; //
    static constexpr u8 s19_3[36] = {0}; //
    static constexpr u8 b19_4[36] = {0}; // FAT
    static constexpr u8 s19_4[36] = {0}; // FAT
    static constexpr u8 b19_5[36] = {0}; //
    static constexpr u8 s19_5[36] = {0}; //
    static constexpr u8 b20_1[36] = {0}; // POWORW WPWWOO WWWPWW WPOWOO WRWWOW PRWWPP
    static constexpr u8 s20_1[36] = {0}; // WWWWOW WPPRPW WOWWWW OOOROO WWWPWR POWPWP
    static constexpr u8 b20_2[36] = {0}; //
    static constexpr u8 s20_2[36] = {0}; //
    static constexpr u8 b20_3[36] = {0}; // FAT
    static constexpr u8 s20_3[36] = {0}; // FAT
    static constexpr u8 b20_4[36] = {0}; // YBYCCY YWWYWC BBYBCC WYYGWW BCYBYB WBWGWW
    static constexpr u8 s20_4[36] = {0}; // YWYWYW WCBCBY YBGBCW WCBGBY YBCBCW WYWYWY
    static constexpr u8 b20_5[36] = {0}; //
    static constexpr u8 s20_5[36] = {0}; //
};


class LevelBoards {
public:
    MU static const Board b1_1;
    MU static const Board s1_1;
    MU static const Board b1_2;
    MU static const Board s1_2;
    MU static const Board b1_3;
    MU static const Board s1_3;
    MU static const Board b1_4;
    MU static const Board s1_4;
    MU static const Board b1_5;
    MU static const Board s1_5;
    MU static const Board b2_1;
    MU static const Board s2_1;
    MU static const Board b2_2;
    MU static const Board s2_2;
    MU static const Board b2_3;
    MU static const Board s2_3;
    MU static const Board b2_4;
    MU static const Board s2_4;
    MU static const Board b2_5;
    MU static const Board s2_5;
    MU static const Board b3_1;
    MU static const Board s3_1;
    MU static const Board b3_2;
    MU static const Board s3_2;
    MU static const Board b3_3;
    MU static const Board s3_3;
    MU static const Board b3_4;
    MU static const Board s3_4;
    MU static const Board b3_5;
    MU static const Board s3_5;
    MU static const Board b4_1;
    MU static const Board s4_1;
    MU static const Board b4_2;
    MU static const Board s4_2;
    MU static const Board b4_3;
    MU static const Board s4_3;
    MU static const Board b4_4;
    MU static const Board s4_4;
    MU static const Board b4_5;
    MU static const Board s4_5;
    MU static const Board b5_1;
    MU static const Board s5_1;
    MU static const Board b5_2;
    MU static const Board s5_2;
    MU static const Board b5_3;
    MU static const Board s5_3;
    MU static const Board b5_4;
    MU static const Board s5_4;
    MU static const Board b5_5;
    MU static const Board s5_5;
    MU static const Board b6_1;
    MU static const Board s6_1;
    MU static const Board b6_2;
    MU static const Board s6_2;
    MU static const Board b6_3;
    MU static const Board s6_3;
    MU static const Board b6_4;
    MU static const Board s6_4;
    MU static const Board b6_5;
    MU static const Board s6_5;
    MU static const Board b7_1;
    MU static const Board s7_1;
    MU static const Board b7_2;
    MU static const Board s7_2;
    MU static const Board b7_3;
    MU static const Board s7_3;
    MU static const Board b7_4;
    MU static const Board s7_4;
    MU static const Board b7_5;
    MU static const Board s7_5;
    MU static const Board b8_1;
    MU static const Board s8_1;
    MU static const Board b8_2;
    MU static const Board s8_2;
    MU static const Board b8_3;
    MU static const Board s8_3;
    MU static const Board b8_4;
    MU static const Board s8_4;
    MU static const Board b8_5;
    MU static const Board s8_5;
    MU static const Board b9_1;
    MU static const Board s9_1;
    MU static const Board b9_2;
    MU static const Board s9_2;
    MU static const Board b9_3;
    MU static const Board s9_3;
    MU static const Board b9_4;
    MU static const Board s9_4;
    MU static const Board b9_5;
    MU static const Board s9_5;
    MU static const Board b10_1;
    MU static const Board s10_1;
    MU static const Board b10_2;
    MU static const Board s10_2;
    MU static const Board b10_3;
    MU static const Board s10_3;
    MU static const Board b10_4;
    MU static const Board s10_4;
    MU static const Board b10_5;
    MU static const Board s10_5;
    MU static const Board b11_1;
    MU static const Board s11_1;
    MU static const Board b11_2;
    MU static const Board s11_2;
    MU static const Board b11_3;
    MU static const Board s11_3;
    MU static const Board b11_4;
    MU static const Board s11_4;
    MU static const Board b11_5;
    MU static const Board s11_5;
    MU static const Board b12_1;
    MU static const Board s12_1;
    MU static const Board b12_2;
    MU static const Board s12_2;
    MU static const Board b12_3;
    MU static const Board s12_3;
    MU static const Board b12_4;
    MU static const Board s12_4;
    MU static const Board b12_5;
    MU static const Board s12_5;
    MU static const Board b13_1;
    MU static const Board s13_1;
    MU static const Board b13_2;
    MU static const Board s13_2;
    MU static const Board b13_3;
    MU static const Board s13_3;
    MU static const Board b13_4;
    MU static const Board s13_4;
    MU static const Board b13_5;
    MU static const Board s13_5;
    MU static const Board b14_1;
    MU static const Board s14_1;
    MU static const Board b14_2;
    MU static const Board s14_2;
    MU static const Board b14_3;
    MU static const Board s14_3;
    MU static const Board b14_4;
    MU static const Board s14_4;
    MU static const Board b14_5;
    MU static const Board s14_5;
    MU static const Board b15_1;
    MU static const Board s15_1;
    MU static const Board b15_2;
    MU static const Board s15_2;
    MU static const Board b15_3;
    MU static const Board s15_3;
    MU static const Board b15_4;
    MU static const Board s15_4;
    MU static const Board b15_5;
    MU static const Board s15_5;
    MU static const Board b16_1;
    MU static const Board s16_1;
    MU static const Board b16_2;
    MU static const Board s16_2;
    MU static const Board b16_3;
    MU static const Board s16_3;
    MU static const Board b16_4;
    MU static const Board s16_4;
    MU static const Board b16_5;
    MU static const Board s16_5;
    MU static const Board b17_1;
    MU static const Board s17_1;
    MU static const Board b17_2;
    MU static const Board s17_2;
    MU static const Board b17_3;
    MU static const Board s17_3;
    MU static const Board b17_4;
    MU static const Board s17_4;
    MU static const Board b17_5;
    MU static const Board s17_5;
    MU static const Board b18_1;
    MU static const Board s18_1;
    MU static const Board b18_2;
    MU static const Board s18_2;
    MU static const Board b18_3;
    MU static const Board s18_3;
    MU static const Board b18_4;
    MU static const Board s18_4;
    MU static const Board b18_5;
    MU static const Board s18_5;
    MU static const Board b19_1;
    MU static const Board s19_1;
    MU static const Board b19_2;
    MU static const Board s19_2;
    MU static const Board b19_3;
    MU static const Board s19_3;
    MU static const Board b19_4;
    MU static const Board s19_4;
    MU static const Board b19_5;
    MU static const Board s19_5;
    MU static const Board b20_1;
    MU static const Board s20_1;
    MU static const Board b20_2;
    MU static const Board s20_2;
    MU static const Board b20_3;
    MU static const Board s20_3;
    MU static const Board b20_4;
    MU static const Board s20_4;
    MU static const Board b20_5;
    MU static const Board s20_5;
};




const Board LevelBoards::b1_1 = Board(LevelCells::b1_1);
const Board LevelBoards::s1_1 = Board(LevelCells::s1_1);
const Board LevelBoards::b1_2 = Board(LevelCells::b1_2);
const Board LevelBoards::s1_2 = Board(LevelCells::s1_2);
const Board LevelBoards::b1_3 = Board(LevelCells::b1_3);
const Board LevelBoards::s1_3 = Board(LevelCells::s1_3);
const Board LevelBoards::b1_4 = Board(LevelCells::b1_4);
const Board LevelBoards::s1_4 = Board(LevelCells::s1_4);
const Board LevelBoards::b1_5 = Board(LevelCells::b1_5);
const Board LevelBoards::s1_5 = Board(LevelCells::s1_5);
const Board LevelBoards::b2_1 = Board(LevelCells::b2_1);
const Board LevelBoards::s2_1 = Board(LevelCells::s2_1);
const Board LevelBoards::b2_2 = Board(LevelCells::b2_2);
const Board LevelBoards::s2_2 = Board(LevelCells::s2_2);
const Board LevelBoards::b2_3 = Board(LevelCells::b2_3);
const Board LevelBoards::s2_3 = Board(LevelCells::s2_3);
const Board LevelBoards::b2_4 = Board(LevelCells::b2_4);
const Board LevelBoards::s2_4 = Board(LevelCells::s2_4);
const Board LevelBoards::b2_5 = Board(LevelCells::b2_5);
const Board LevelBoards::s2_5 = Board(LevelCells::s2_5);
const Board LevelBoards::b3_1 = Board(LevelCells::b3_1);
const Board LevelBoards::s3_1 = Board(LevelCells::s3_1);
const Board LevelBoards::b3_2 = Board(LevelCells::b3_2);
const Board LevelBoards::s3_2 = Board(LevelCells::s3_2);
const Board LevelBoards::b3_3 = Board(LevelCells::b3_3);
const Board LevelBoards::s3_3 = Board(LevelCells::s3_3);
const Board LevelBoards::b3_4 = Board(LevelCells::b3_4);
const Board LevelBoards::s3_4 = Board(LevelCells::s3_4);
const Board LevelBoards::b3_5 = Board(LevelCells::b3_5);
const Board LevelBoards::s3_5 = Board(LevelCells::s3_5);
const Board LevelBoards::b4_1 = Board(LevelCells::b4_1);
const Board LevelBoards::s4_1 = Board(LevelCells::s4_1);
const Board LevelBoards::b4_2 = Board(LevelCells::b4_2, 4, 4);
const Board LevelBoards::s4_2 = Board(LevelCells::s4_2, 2, 2);
const Board LevelBoards::b4_3 = Board(LevelCells::b4_3);
const Board LevelBoards::s4_3 = Board(LevelCells::s4_3);
const Board LevelBoards::b4_4 = Board(LevelCells::b4_4, 2, 2);
const Board LevelBoards::s4_4 = Board(LevelCells::s4_4, 2, 2);
const Board LevelBoards::b4_5 = Board(LevelCells::b4_5);
const Board LevelBoards::s4_5 = Board(LevelCells::s4_5);
const Board LevelBoards::b5_1 = Board(LevelCells::b5_1, 0, 2);
const Board LevelBoards::s5_1 = Board(LevelCells::s5_1, 3, 3);
const Board LevelBoards::b5_2 = Board(LevelCells::b5_2);
const Board LevelBoards::s5_2 = Board(LevelCells::s5_2);
const Board LevelBoards::b5_3 = Board(LevelCells::b5_3);
const Board LevelBoards::s5_3 = Board(LevelCells::s5_3);
const Board LevelBoards::b5_4 = Board(LevelCells::b5_4);
const Board LevelBoards::s5_4 = Board(LevelCells::s5_4);
const Board LevelBoards::b5_5 = Board(LevelCells::b5_5);
const Board LevelBoards::s5_5 = Board(LevelCells::s5_5);
const Board LevelBoards::b6_1 = Board(LevelCells::b6_1, 4, 4);
const Board LevelBoards::s6_1 = Board(LevelCells::s6_1, 2, 2);
const Board LevelBoards::b6_2 = Board(LevelCells::b6_2, 1, 3);
const Board LevelBoards::s6_2 = Board(LevelCells::s6_2, 2, 2);
const Board LevelBoards::b6_3 = Board(LevelCells::b6_3, 0, 1);
const Board LevelBoards::s6_3 = Board(LevelCells::s6_3, 0, 0);
const Board LevelBoards::b6_4 = Board(LevelCells::b6_4, 3, 3);
const Board LevelBoards::s6_4 = Board(LevelCells::s6_4, 2, 2);
const Board LevelBoards::b6_5 = Board(LevelCells::b6_5, 3, 0);
const Board LevelBoards::s6_5 = Board(LevelCells::s6_5, 2, 2);
const Board LevelBoards::b7_1 = Board(LevelCells::b7_1);
const Board LevelBoards::s7_1 = Board(LevelCells::s7_1);
const Board LevelBoards::b7_2 = Board(LevelCells::b7_2);
const Board LevelBoards::s7_2 = Board(LevelCells::s7_2);
const Board LevelBoards::b7_3 = Board(LevelCells::b7_3);
const Board LevelBoards::s7_3 = Board(LevelCells::s7_3);
const Board LevelBoards::b7_4 = Board(LevelCells::b7_4);
const Board LevelBoards::s7_4 = Board(LevelCells::s7_4);
const Board LevelBoards::b7_5 = Board(LevelCells::b7_5);
const Board LevelBoards::s7_5 = Board(LevelCells::s7_5);
const Board LevelBoards::b8_1 = Board(LevelCells::b8_1);
const Board LevelBoards::s8_1 = Board(LevelCells::s8_1);
const Board LevelBoards::b8_2 = Board(LevelCells::b8_2, 4, 4);
const Board LevelBoards::s8_2 = Board(LevelCells::s8_2, 2, 2);
const Board LevelBoards::b8_3 = Board(LevelCells::b8_3);
const Board LevelBoards::s8_3 = Board(LevelCells::s8_3);
const Board LevelBoards::b8_4 = Board(LevelCells::b8_4, 3, 4);
const Board LevelBoards::s8_4 = Board(LevelCells::s8_4, 2, 2);
const Board LevelBoards::b8_5 = Board(LevelCells::b8_5);
const Board LevelBoards::s8_5 = Board(LevelCells::s8_5);
const Board LevelBoards::b9_1 = Board(LevelCells::b9_1, 1, 3);
const Board LevelBoards::s9_1 = Board(LevelCells::s9_1, 2, 2);
const Board LevelBoards::b9_2 = Board(LevelCells::b9_2);
const Board LevelBoards::s9_2 = Board(LevelCells::s9_2);
const Board LevelBoards::b9_3 = Board(LevelCells::b9_3);
const Board LevelBoards::s9_3 = Board(LevelCells::s9_3);
const Board LevelBoards::b9_4 = Board(LevelCells::b9_4);
const Board LevelBoards::s9_4 = Board(LevelCells::s9_4);
const Board LevelBoards::b9_5 = Board(LevelCells::b9_5);
const Board LevelBoards::s9_5 = Board(LevelCells::s9_5);
const Board LevelBoards::b10_1 = Board(LevelCells::b10_1);
const Board LevelBoards::s10_1 = Board(LevelCells::s10_1);
const Board LevelBoards::b10_2 = Board(LevelCells::b10_2);
const Board LevelBoards::s10_2 = Board(LevelCells::s10_2);
const Board LevelBoards::b10_3 = Board(LevelCells::b10_3);
const Board LevelBoards::s10_3 = Board(LevelCells::s10_3);
const Board LevelBoards::b10_4 = Board(LevelCells::b10_4);
const Board LevelBoards::s10_4 = Board(LevelCells::s10_4);
const Board LevelBoards::b10_5 = Board(LevelCells::b10_5);
const Board LevelBoards::s10_5 = Board(LevelCells::s10_5);
const Board LevelBoards::b11_1 = Board(LevelCells::b11_1);
const Board LevelBoards::s11_1 = Board(LevelCells::s11_1);
const Board LevelBoards::b11_2 = Board(LevelCells::b11_2);
const Board LevelBoards::s11_2 = Board(LevelCells::s11_2);
const Board LevelBoards::b11_3 = Board(LevelCells::b11_3);
const Board LevelBoards::s11_3 = Board(LevelCells::s11_3);
const Board LevelBoards::b11_4 = Board(LevelCells::b11_4);
const Board LevelBoards::s11_4 = Board(LevelCells::s11_4);
const Board LevelBoards::b11_5 = Board(LevelCells::b11_5);
const Board LevelBoards::s11_5 = Board(LevelCells::s11_5);
const Board LevelBoards::b12_1 = Board(LevelCells::b12_1);
const Board LevelBoards::s12_1 = Board(LevelCells::s12_1);
const Board LevelBoards::b12_2 = Board(LevelCells::b12_2, 1, 3);
const Board LevelBoards::s12_2 = Board(LevelCells::s12_2, 2, 2);
const Board LevelBoards::b12_3 = Board(LevelCells::b12_3);
const Board LevelBoards::s12_3 = Board(LevelCells::s12_3);
const Board LevelBoards::b12_4 = Board(LevelCells::b12_4);
const Board LevelBoards::s12_4 = Board(LevelCells::s12_4);
const Board LevelBoards::b12_5 = Board(LevelCells::b12_5);
const Board LevelBoards::s12_5 = Board(LevelCells::s12_5);
const Board LevelBoards::b13_1 = Board(LevelCells::b13_1);
const Board LevelBoards::s13_1 = Board(LevelCells::s13_1);
const Board LevelBoards::b13_2 = Board(LevelCells::b13_2);
const Board LevelBoards::s13_2 = Board(LevelCells::s13_2);
const Board LevelBoards::b13_3 = Board(LevelCells::b13_3);
const Board LevelBoards::s13_3 = Board(LevelCells::s13_3);
const Board LevelBoards::b13_4 = Board(LevelCells::b13_4, 1, 4);
const Board LevelBoards::s13_4 = Board(LevelCells::s13_4, 2, 2);
const Board LevelBoards::b13_5 = Board(LevelCells::b13_5, 2, 4);
const Board LevelBoards::s13_5 = Board(LevelCells::s13_5, 2, 2);
const Board LevelBoards::b14_1 = Board(LevelCells::b14_1);
const Board LevelBoards::s14_1 = Board(LevelCells::s14_1);
const Board LevelBoards::b14_2 = Board(LevelCells::b14_2);
const Board LevelBoards::s14_2 = Board(LevelCells::s14_2);
const Board LevelBoards::b14_3 = Board(LevelCells::b14_3);
const Board LevelBoards::s14_3 = Board(LevelCells::s14_3);
const Board LevelBoards::b14_4 = Board(LevelCells::b14_4);
const Board LevelBoards::s14_4 = Board(LevelCells::s14_4);
const Board LevelBoards::b14_5 = Board(LevelCells::b14_5);
const Board LevelBoards::s14_5 = Board(LevelCells::s14_5);
const Board LevelBoards::b15_1 = Board(LevelCells::b15_1);
const Board LevelBoards::s15_1 = Board(LevelCells::s15_1);
const Board LevelBoards::b15_2 = Board(LevelCells::b15_2, 3, 4);
const Board LevelBoards::s15_2 = Board(LevelCells::s15_2, 2, 2);
const Board LevelBoards::b15_3 = Board(LevelCells::b15_3, 1, 0);
const Board LevelBoards::s15_3 = Board(LevelCells::s15_3, 2, 2);
const Board LevelBoards::b15_4 = Board(LevelCells::b15_4, 1, 4);
const Board LevelBoards::s15_4 = Board(LevelCells::s15_4, 2, 4);
const Board LevelBoards::b15_5 = Board(LevelCells::b15_5);
const Board LevelBoards::s15_5 = Board(LevelCells::s15_5);
const Board LevelBoards::b16_1 = Board(LevelCells::b16_1, 3, 4);
const Board LevelBoards::s16_1 = Board(LevelCells::s16_1, 2, 2);
const Board LevelBoards::b16_2 = Board(LevelCells::b16_2);
const Board LevelBoards::s16_2 = Board(LevelCells::s16_2);
const Board LevelBoards::b16_3 = Board(LevelCells::b16_3);
const Board LevelBoards::s16_3 = Board(LevelCells::s16_3);
const Board LevelBoards::b16_4 = Board(LevelCells::b16_4);
const Board LevelBoards::s16_4 = Board(LevelCells::s16_4);
const Board LevelBoards::b16_5 = Board(LevelCells::b16_5, 2, 0);
const Board LevelBoards::s16_5 = Board(LevelCells::s16_5, 2, 2);
const Board LevelBoards::b17_1 = Board(LevelCells::b17_1);
const Board LevelBoards::s17_1 = Board(LevelCells::s17_1);
const Board LevelBoards::b17_2 = Board(LevelCells::b17_2, 4, 4);
const Board LevelBoards::s17_2 = Board(LevelCells::s17_2, 1, 3);
const Board LevelBoards::b17_3 = Board(LevelCells::b17_3);
const Board LevelBoards::s17_3 = Board(LevelCells::s17_3);
const Board LevelBoards::b17_4 = Board(LevelCells::b17_4, 2, 1);
const Board LevelBoards::s17_4 = Board(LevelCells::s17_4, 2, 2);
const Board LevelBoards::b17_5 = Board(LevelCells::b17_5);
const Board LevelBoards::s17_5 = Board(LevelCells::s17_5);
const Board LevelBoards::b18_1 = Board(LevelCells::b18_1, 1, 4);
const Board LevelBoards::s18_1 = Board(LevelCells::s18_1, 2, 2);
const Board LevelBoards::b18_2 = Board(LevelCells::b18_2, 0, 0);
const Board LevelBoards::s18_2 = Board(LevelCells::s18_2, 2, 2);
const Board LevelBoards::b18_3 = Board(LevelCells::b18_3);
const Board LevelBoards::s18_3 = Board(LevelCells::s18_3);
const Board LevelBoards::b18_4 = Board(LevelCells::b18_4, 1, 2);
const Board LevelBoards::s18_4 = Board(LevelCells::s18_4, 4, 0);
const Board LevelBoards::b18_5 = Board(LevelCells::b18_5, 4, 3);
const Board LevelBoards::s18_5 = Board(LevelCells::s18_5, 2, 2);
const Board LevelBoards::b19_1 = Board(LevelCells::b19_1);
const Board LevelBoards::s19_1 = Board(LevelCells::s19_1);
const Board LevelBoards::b19_2 = Board(LevelCells::b19_2, 0, 3);
const Board LevelBoards::s19_2 = Board(LevelCells::s19_2, 2, 1);
const Board LevelBoards::b19_3 = Board(LevelCells::b19_3);
const Board LevelBoards::s19_3 = Board(LevelCells::s19_3);
const Board LevelBoards::b19_4 = Board(LevelCells::b19_4, 1, 4);
const Board LevelBoards::s19_4 = Board(LevelCells::s19_4, 2, 2);
const Board LevelBoards::b19_5 = Board(LevelCells::b19_5);
const Board LevelBoards::s19_5 = Board(LevelCells::s19_5);
const Board LevelBoards::b20_1 = Board(LevelCells::b20_1);
const Board LevelBoards::s20_1 = Board(LevelCells::s20_1);
const Board LevelBoards::b20_2 = Board(LevelCells::b20_2);
const Board LevelBoards::s20_2 = Board(LevelCells::s20_2);
const Board LevelBoards::b20_3 = Board(LevelCells::b20_3, 2, 2);
const Board LevelBoards::s20_3 = Board(LevelCells::s20_3, 2, 2);
const Board LevelBoards::b20_4 = Board(LevelCells::b20_4);
const Board LevelBoards::s20_4 = Board(LevelCells::s20_4);
const Board LevelBoards::b20_5 = Board(LevelCells::b20_5);
const Board LevelBoards::s20_5 = Board(LevelCells::s20_5);


const BoardPair LevelBoardPair::p1_1 = BoardPair(&LevelBoards::b1_1, &LevelBoards::s1_1, "1-1");
const BoardPair LevelBoardPair::p1_2 = BoardPair(&LevelBoards::b1_2, &LevelBoards::s1_2, "1-2");
const BoardPair LevelBoardPair::p1_3 = BoardPair(&LevelBoards::b1_3, &LevelBoards::s1_3, "1-3");
const BoardPair LevelBoardPair::p1_4 = BoardPair(&LevelBoards::b1_4, &LevelBoards::s1_4, "1-4");
const BoardPair LevelBoardPair::p1_5 = BoardPair(&LevelBoards::b1_5, &LevelBoards::s1_5, "1-5");
const BoardPair LevelBoardPair::p2_1 = BoardPair(&LevelBoards::b2_1, &LevelBoards::s2_1, "2-1");
const BoardPair LevelBoardPair::p2_2 = BoardPair(&LevelBoards::b2_2, &LevelBoards::s2_2, "2-2");
const BoardPair LevelBoardPair::p2_3 = BoardPair(&LevelBoards::b2_3, &LevelBoards::s2_3, "2-3");
const BoardPair LevelBoardPair::p2_4 = BoardPair(&LevelBoards::b2_4, &LevelBoards::s2_4, "2-4");
const BoardPair LevelBoardPair::p2_5 = BoardPair(&LevelBoards::b2_5, &LevelBoards::s2_5, "2-5");
const BoardPair LevelBoardPair::p3_1 = BoardPair(&LevelBoards::b3_1, &LevelBoards::s3_1, "3-1");
const BoardPair LevelBoardPair::p3_2 = BoardPair(&LevelBoards::b3_2, &LevelBoards::s3_2, "3-2");
const BoardPair LevelBoardPair::p3_3 = BoardPair(&LevelBoards::b3_3, &LevelBoards::s3_3, "3-3");
const BoardPair LevelBoardPair::p3_4 = BoardPair(&LevelBoards::b3_4, &LevelBoards::s3_4, "3-4");
const BoardPair LevelBoardPair::p3_5 = BoardPair(&LevelBoards::b3_5, &LevelBoards::s3_5, "3-5");
const BoardPair LevelBoardPair::p4_1 = BoardPair(&LevelBoards::b4_1, &LevelBoards::s4_1, "4-1");
const BoardPair LevelBoardPair::p4_2 = BoardPair(&LevelBoards::b4_2, &LevelBoards::s4_2, "4-2");
const BoardPair LevelBoardPair::p4_3 = BoardPair(&LevelBoards::b4_3, &LevelBoards::s4_3, "4-3");
const BoardPair LevelBoardPair::p4_4 = BoardPair(&LevelBoards::b4_4, &LevelBoards::s4_4, "4-4");
const BoardPair LevelBoardPair::p4_5 = BoardPair(&LevelBoards::b4_5, &LevelBoards::s4_5, "4-5");
const BoardPair LevelBoardPair::p5_1 = BoardPair(&LevelBoards::b5_1, &LevelBoards::s5_1, "5-1");
const BoardPair LevelBoardPair::p5_2 = BoardPair(&LevelBoards::b5_2, &LevelBoards::s5_2, "5-2");
const BoardPair LevelBoardPair::p5_3 = BoardPair(&LevelBoards::b5_3, &LevelBoards::s5_3, "5-3");
const BoardPair LevelBoardPair::p5_4 = BoardPair(&LevelBoards::b5_4, &LevelBoards::s5_4, "5-4");
const BoardPair LevelBoardPair::p5_5 = BoardPair(&LevelBoards::b5_5, &LevelBoards::s5_5, "5-5");
const BoardPair LevelBoardPair::p6_1 = BoardPair(&LevelBoards::b6_1, &LevelBoards::s6_1, "6-1");
const BoardPair LevelBoardPair::p6_2 = BoardPair(&LevelBoards::b6_2, &LevelBoards::s6_2, "6-2");
const BoardPair LevelBoardPair::p6_3 = BoardPair(&LevelBoards::b6_3, &LevelBoards::s6_3, "6-3");
const BoardPair LevelBoardPair::p6_4 = BoardPair(&LevelBoards::b6_4, &LevelBoards::s6_4, "6-4");
const BoardPair LevelBoardPair::p6_5 = BoardPair(&LevelBoards::b6_5, &LevelBoards::s6_5, "6-5");
const BoardPair LevelBoardPair::p7_1 = BoardPair(&LevelBoards::b7_1, &LevelBoards::s7_1, "7-1");
const BoardPair LevelBoardPair::p7_2 = BoardPair(&LevelBoards::b7_2, &LevelBoards::s7_2, "7-2");
const BoardPair LevelBoardPair::p7_3 = BoardPair(&LevelBoards::b7_3, &LevelBoards::s7_3, "7-3");
const BoardPair LevelBoardPair::p7_4 = BoardPair(&LevelBoards::b7_4, &LevelBoards::s7_4, "7-4");
const BoardPair LevelBoardPair::p7_5 = BoardPair(&LevelBoards::b7_5, &LevelBoards::s7_5, "7-5");
const BoardPair LevelBoardPair::p8_1 = BoardPair(&LevelBoards::b8_1, &LevelBoards::s8_1, "8-1");
const BoardPair LevelBoardPair::p8_2 = BoardPair(&LevelBoards::b8_2, &LevelBoards::s8_2, "8-2");
const BoardPair LevelBoardPair::p8_3 = BoardPair(&LevelBoards::b8_3, &LevelBoards::s8_3, "8-3");
const BoardPair LevelBoardPair::p8_4 = BoardPair(&LevelBoards::b8_4, &LevelBoards::s8_4, "8-4");
const BoardPair LevelBoardPair::p8_5 = BoardPair(&LevelBoards::b8_5, &LevelBoards::s8_5, "8-5");
const BoardPair LevelBoardPair::p9_1 = BoardPair(&LevelBoards::b9_1, &LevelBoards::s9_1, "9-1");
const BoardPair LevelBoardPair::p9_2 = BoardPair(&LevelBoards::b9_2, &LevelBoards::s9_2, "9-2");
const BoardPair LevelBoardPair::p9_3 = BoardPair(&LevelBoards::b9_3, &LevelBoards::s9_3, "9-3");
const BoardPair LevelBoardPair::p9_4 = BoardPair(&LevelBoards::b9_4, &LevelBoards::s9_4, "9-4");
const BoardPair LevelBoardPair::p9_5 = BoardPair(&LevelBoards::b9_5, &LevelBoards::s9_5, "9-5");
const BoardPair LevelBoardPair::p10_1 = BoardPair(&LevelBoards::b10_1, &LevelBoards::s10_1, "10-1");
const BoardPair LevelBoardPair::p10_2 = BoardPair(&LevelBoards::b10_2, &LevelBoards::s10_2, "10-2");
const BoardPair LevelBoardPair::p10_3 = BoardPair(&LevelBoards::b10_3, &LevelBoards::s10_3, "10-3");
const BoardPair LevelBoardPair::p10_4 = BoardPair(&LevelBoards::b10_4, &LevelBoards::s10_4, "10-4");
const BoardPair LevelBoardPair::p10_5 = BoardPair(&LevelBoards::b10_5, &LevelBoards::s10_5, "10-5");
const BoardPair LevelBoardPair::p11_1 = BoardPair(&LevelBoards::b11_1, &LevelBoards::s11_1, "11-1");
const BoardPair LevelBoardPair::p11_2 = BoardPair(&LevelBoards::b11_2, &LevelBoards::s11_2, "11-2");
const BoardPair LevelBoardPair::p11_3 = BoardPair(&LevelBoards::b11_3, &LevelBoards::s11_3, "11-3");
const BoardPair LevelBoardPair::p11_4 = BoardPair(&LevelBoards::b11_4, &LevelBoards::s11_4, "11-4");
const BoardPair LevelBoardPair::p11_5 = BoardPair(&LevelBoards::b11_5, &LevelBoards::s11_5, "11-5");
const BoardPair LevelBoardPair::p12_1 = BoardPair(&LevelBoards::b12_1, &LevelBoards::s12_1, "12-1");
const BoardPair LevelBoardPair::p12_2 = BoardPair(&LevelBoards::b12_2, &LevelBoards::s12_2, "12-2");
const BoardPair LevelBoardPair::p12_3 = BoardPair(&LevelBoards::b12_3, &LevelBoards::s12_3, "12-3");
const BoardPair LevelBoardPair::p12_4 = BoardPair(&LevelBoards::b12_4, &LevelBoards::s12_4, "12-4");
const BoardPair LevelBoardPair::p12_5 = BoardPair(&LevelBoards::b12_5, &LevelBoards::s12_5, "12-5");
const BoardPair LevelBoardPair::p13_1 = BoardPair(&LevelBoards::b13_1, &LevelBoards::s13_1, "13-1");
const BoardPair LevelBoardPair::p13_2 = BoardPair(&LevelBoards::b13_2, &LevelBoards::s13_2, "13-2");
const BoardPair LevelBoardPair::p13_3 = BoardPair(&LevelBoards::b13_3, &LevelBoards::s13_3, "13-3");
const BoardPair LevelBoardPair::p13_4 = BoardPair(&LevelBoards::b13_4, &LevelBoards::s13_4, "13-4");
const BoardPair LevelBoardPair::p13_5 = BoardPair(&LevelBoards::b13_5, &LevelBoards::s13_5, "13-5");
const BoardPair LevelBoardPair::p14_1 = BoardPair(&LevelBoards::b14_1, &LevelBoards::s14_1, "14-1");
const BoardPair LevelBoardPair::p14_2 = BoardPair(&LevelBoards::b14_2, &LevelBoards::s14_2, "14-2");
const BoardPair LevelBoardPair::p14_3 = BoardPair(&LevelBoards::b14_3, &LevelBoards::s14_3, "14-3");
const BoardPair LevelBoardPair::p14_4 = BoardPair(&LevelBoards::b14_4, &LevelBoards::s14_4, "14-4");
const BoardPair LevelBoardPair::p14_5 = BoardPair(&LevelBoards::b14_5, &LevelBoards::s14_5, "14-5");
const BoardPair LevelBoardPair::p15_1 = BoardPair(&LevelBoards::b15_1, &LevelBoards::s15_1, "15-1");
const BoardPair LevelBoardPair::p15_2 = BoardPair(&LevelBoards::b15_2, &LevelBoards::s15_2, "15-2");
const BoardPair LevelBoardPair::p15_3 = BoardPair(&LevelBoards::b15_3, &LevelBoards::s15_3, "15-3");
const BoardPair LevelBoardPair::p15_4 = BoardPair(&LevelBoards::b15_4, &LevelBoards::s15_4, "15-4");
const BoardPair LevelBoardPair::p15_5 = BoardPair(&LevelBoards::b15_5, &LevelBoards::s15_5, "15-5");
const BoardPair LevelBoardPair::p16_1 = BoardPair(&LevelBoards::b16_1, &LevelBoards::s16_1, "16-1");
const BoardPair LevelBoardPair::p16_2 = BoardPair(&LevelBoards::b16_2, &LevelBoards::s16_2, "16-2");
const BoardPair LevelBoardPair::p16_3 = BoardPair(&LevelBoards::b16_3, &LevelBoards::s16_3, "16-3");
const BoardPair LevelBoardPair::p16_4 = BoardPair(&LevelBoards::b16_4, &LevelBoards::s16_4, "16-4");
const BoardPair LevelBoardPair::p16_5 = BoardPair(&LevelBoards::b16_5, &LevelBoards::s16_5, "16-5");
const BoardPair LevelBoardPair::p17_1 = BoardPair(&LevelBoards::b17_1, &LevelBoards::s17_1, "17-1");
const BoardPair LevelBoardPair::p17_2 = BoardPair(&LevelBoards::b17_2, &LevelBoards::s17_2, "17-2");
const BoardPair LevelBoardPair::p17_3 = BoardPair(&LevelBoards::b17_3, &LevelBoards::s17_3, "17-3");
const BoardPair LevelBoardPair::p17_4 = BoardPair(&LevelBoards::b17_4, &LevelBoards::s17_4, "17-4");
const BoardPair LevelBoardPair::p17_5 = BoardPair(&LevelBoards::b17_5, &LevelBoards::s17_5, "17-5");
const BoardPair LevelBoardPair::p18_1 = BoardPair(&LevelBoards::b18_1, &LevelBoards::s18_1, "18-1");
const BoardPair LevelBoardPair::p18_2 = BoardPair(&LevelBoards::b18_2, &LevelBoards::s18_2, "18-2");
const BoardPair LevelBoardPair::p18_3 = BoardPair(&LevelBoards::b18_3, &LevelBoards::s18_3, "18-3");
const BoardPair LevelBoardPair::p18_4 = BoardPair(&LevelBoards::b18_4, &LevelBoards::s18_4, "18-4");
const BoardPair LevelBoardPair::p18_5 = BoardPair(&LevelBoards::b18_5, &LevelBoards::s18_5, "18-5");
const BoardPair LevelBoardPair::p19_1 = BoardPair(&LevelBoards::b19_1, &LevelBoards::s19_1, "19-1");
const BoardPair LevelBoardPair::p19_2 = BoardPair(&LevelBoards::b19_2, &LevelBoards::s19_2, "19-2");
const BoardPair LevelBoardPair::p19_3 = BoardPair(&LevelBoards::b19_3, &LevelBoards::s19_3, "19-3");
const BoardPair LevelBoardPair::p19_4 = BoardPair(&LevelBoards::b19_4, &LevelBoards::s19_4, "19-4");
const BoardPair LevelBoardPair::p19_5 = BoardPair(&LevelBoards::b19_5, &LevelBoards::s19_5, "19-5");
const BoardPair LevelBoardPair::p20_1 = BoardPair(&LevelBoards::b20_1, &LevelBoards::s20_1, "20-1");
const BoardPair LevelBoardPair::p20_2 = BoardPair(&LevelBoards::b20_2, &LevelBoards::s20_2, "20-2");
const BoardPair LevelBoardPair::p20_3 = BoardPair(&LevelBoards::b20_3, &LevelBoards::s20_3, "20-3");
const BoardPair LevelBoardPair::p20_4 = BoardPair(&LevelBoards::b20_4, &LevelBoards::s20_4, "20-4");
const BoardPair LevelBoardPair::p20_5 = BoardPair(&LevelBoards::b20_5, &LevelBoards::s20_5, "20-5");



const std::unordered_map<std::string, const BoardPair*> BoardLookup::boardPairDict = {
        {LevelBoardPair::p1_1.getName(), &LevelBoardPair::p1_1},
        {LevelBoardPair::p1_2.getName(), &LevelBoardPair::p1_2},
        {LevelBoardPair::p1_3.getName(), &LevelBoardPair::p1_3},
        {LevelBoardPair::p1_4.getName(), &LevelBoardPair::p1_4},
        {LevelBoardPair::p1_5.getName(), &LevelBoardPair::p1_5},
        {LevelBoardPair::p2_1.getName(), &LevelBoardPair::p2_1},
        {LevelBoardPair::p2_2.getName(), &LevelBoardPair::p2_2},
        {LevelBoardPair::p2_3.getName(), &LevelBoardPair::p2_3},
        {LevelBoardPair::p2_4.getName(), &LevelBoardPair::p2_4},
        {LevelBoardPair::p2_5.getName(), &LevelBoardPair::p2_5},
        {LevelBoardPair::p3_1.getName(), &LevelBoardPair::p3_1},
        {LevelBoardPair::p3_2.getName(), &LevelBoardPair::p3_2},
        {LevelBoardPair::p3_3.getName(), &LevelBoardPair::p3_3},
        {LevelBoardPair::p3_4.getName(), &LevelBoardPair::p3_4},
        {LevelBoardPair::p3_5.getName(), &LevelBoardPair::p3_5},
        {LevelBoardPair::p4_1.getName(), &LevelBoardPair::p4_1},
        {LevelBoardPair::p4_2.getName(), &LevelBoardPair::p4_2},
        {LevelBoardPair::p4_3.getName(), &LevelBoardPair::p4_3},
        {LevelBoardPair::p4_4.getName(), &LevelBoardPair::p4_4},
        {LevelBoardPair::p4_5.getName(), &LevelBoardPair::p4_5},
        {LevelBoardPair::p5_1.getName(), &LevelBoardPair::p5_1},
        {LevelBoardPair::p5_2.getName(), &LevelBoardPair::p5_2},
        {LevelBoardPair::p5_3.getName(), &LevelBoardPair::p5_3},
        {LevelBoardPair::p5_4.getName(), &LevelBoardPair::p5_4},
        {LevelBoardPair::p5_5.getName(), &LevelBoardPair::p5_5},
        {LevelBoardPair::p6_1.getName(), &LevelBoardPair::p6_1},
        {LevelBoardPair::p6_2.getName(), &LevelBoardPair::p6_2},
        {LevelBoardPair::p6_3.getName(), &LevelBoardPair::p6_3},
        {LevelBoardPair::p6_4.getName(), &LevelBoardPair::p6_4},
        {LevelBoardPair::p6_5.getName(), &LevelBoardPair::p6_5},
        {LevelBoardPair::p7_1.getName(), &LevelBoardPair::p7_1},
        {LevelBoardPair::p7_2.getName(), &LevelBoardPair::p7_2},
        {LevelBoardPair::p7_3.getName(), &LevelBoardPair::p7_3},
        {LevelBoardPair::p7_4.getName(), &LevelBoardPair::p7_4},
        {LevelBoardPair::p7_5.getName(), &LevelBoardPair::p7_5},
        {LevelBoardPair::p8_1.getName(), &LevelBoardPair::p8_1},
        {LevelBoardPair::p8_2.getName(), &LevelBoardPair::p8_2},
        {LevelBoardPair::p8_3.getName(), &LevelBoardPair::p8_3},
        {LevelBoardPair::p8_4.getName(), &LevelBoardPair::p8_4},
        {LevelBoardPair::p8_5.getName(), &LevelBoardPair::p8_5},
        {LevelBoardPair::p9_1.getName(), &LevelBoardPair::p9_1},
        {LevelBoardPair::p9_2.getName(), &LevelBoardPair::p9_2},
        {LevelBoardPair::p9_3.getName(), &LevelBoardPair::p9_3},
        {LevelBoardPair::p9_4.getName(), &LevelBoardPair::p9_4},
        {LevelBoardPair::p9_5.getName(), &LevelBoardPair::p9_5},
        {LevelBoardPair::p10_1.getName(), &LevelBoardPair::p10_1},
        {LevelBoardPair::p10_2.getName(), &LevelBoardPair::p10_2},
        {LevelBoardPair::p10_3.getName(), &LevelBoardPair::p10_3},
        {LevelBoardPair::p10_4.getName(), &LevelBoardPair::p10_4},
        {LevelBoardPair::p10_5.getName(), &LevelBoardPair::p10_5},
        {LevelBoardPair::p11_1.getName(), &LevelBoardPair::p11_1},
        {LevelBoardPair::p11_2.getName(), &LevelBoardPair::p11_2},
        {LevelBoardPair::p11_3.getName(), &LevelBoardPair::p11_3},
        {LevelBoardPair::p11_4.getName(), &LevelBoardPair::p11_4},
        {LevelBoardPair::p11_5.getName(), &LevelBoardPair::p11_5},
        {LevelBoardPair::p12_1.getName(), &LevelBoardPair::p12_1},
        {LevelBoardPair::p12_2.getName(), &LevelBoardPair::p12_2},
        {LevelBoardPair::p12_3.getName(), &LevelBoardPair::p12_3},
        {LevelBoardPair::p12_4.getName(), &LevelBoardPair::p12_4},
        {LevelBoardPair::p12_5.getName(), &LevelBoardPair::p12_5},
        {LevelBoardPair::p13_1.getName(), &LevelBoardPair::p13_1},
        {LevelBoardPair::p13_2.getName(), &LevelBoardPair::p13_2},
        {LevelBoardPair::p13_3.getName(), &LevelBoardPair::p13_3},
        {LevelBoardPair::p13_4.getName(), &LevelBoardPair::p13_4},
        {LevelBoardPair::p13_5.getName(), &LevelBoardPair::p13_5},
        {LevelBoardPair::p14_1.getName(), &LevelBoardPair::p14_1},
        {LevelBoardPair::p14_2.getName(), &LevelBoardPair::p14_2},
        {LevelBoardPair::p14_3.getName(), &LevelBoardPair::p14_3},
        {LevelBoardPair::p14_4.getName(), &LevelBoardPair::p14_4},
        {LevelBoardPair::p14_5.getName(), &LevelBoardPair::p14_5},
        {LevelBoardPair::p15_1.getName(), &LevelBoardPair::p15_1},
        {LevelBoardPair::p15_2.getName(), &LevelBoardPair::p15_2},
        {LevelBoardPair::p15_3.getName(), &LevelBoardPair::p15_3},
        {LevelBoardPair::p15_4.getName(), &LevelBoardPair::p15_4},
        {LevelBoardPair::p15_5.getName(), &LevelBoardPair::p15_5},
        {LevelBoardPair::p16_1.getName(), &LevelBoardPair::p16_1},
        {LevelBoardPair::p16_2.getName(), &LevelBoardPair::p16_2},
        {LevelBoardPair::p16_3.getName(), &LevelBoardPair::p16_3},
        {LevelBoardPair::p16_4.getName(), &LevelBoardPair::p16_4},
        {LevelBoardPair::p16_5.getName(), &LevelBoardPair::p16_5},
        {LevelBoardPair::p17_1.getName(), &LevelBoardPair::p17_1},
        {LevelBoardPair::p17_2.getName(), &LevelBoardPair::p17_2},
        {LevelBoardPair::p17_3.getName(), &LevelBoardPair::p17_3},
        {LevelBoardPair::p17_4.getName(), &LevelBoardPair::p17_4},
        {LevelBoardPair::p17_5.getName(), &LevelBoardPair::p17_5},
        {LevelBoardPair::p18_1.getName(), &LevelBoardPair::p18_1},
        {LevelBoardPair::p18_2.getName(), &LevelBoardPair::p18_2},
        {LevelBoardPair::p18_3.getName(), &LevelBoardPair::p18_3},
        {LevelBoardPair::p18_4.getName(), &LevelBoardPair::p18_4},
        {LevelBoardPair::p18_5.getName(), &LevelBoardPair::p18_5},
        {LevelBoardPair::p19_1.getName(), &LevelBoardPair::p19_1},
        {LevelBoardPair::p19_2.getName(), &LevelBoardPair::p19_2},
        {LevelBoardPair::p19_3.getName(), &LevelBoardPair::p19_3},
        {LevelBoardPair::p19_4.getName(), &LevelBoardPair::p19_4},
        {LevelBoardPair::p19_5.getName(), &LevelBoardPair::p19_5},
        {LevelBoardPair::p20_1.getName(), &LevelBoardPair::p20_1},
        {LevelBoardPair::p20_2.getName(), &LevelBoardPair::p20_2},
        {LevelBoardPair::p20_3.getName(), &LevelBoardPair::p20_3},
        {LevelBoardPair::p20_4.getName(), &LevelBoardPair::p20_4},
        {LevelBoardPair::p20_5.getName(), &LevelBoardPair::p20_5},
};



