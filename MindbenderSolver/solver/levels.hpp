#pragma once

#include <cstdint>


struct Levels {
public:
    static constexpr uint8_t b1_1[36] = {1,1,0,0,1,1,1,1,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,0,0,0,1};
    static constexpr uint8_t s1_1[36] = {1,1,0,0,1,1,1,1,0,0,1,1,0,0,1,1,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,1,1};
    static constexpr uint8_t b1_2[36] = {2,3,2,2,2,2,3,3,3,3,2,2,2,2,2,2,3,2,2,2,2,2,3,2,2,3,3,3,3,2,2,3,2,2,2,2};
    static constexpr uint8_t s1_2[36] = {2,2,2,2,2,2,2,3,3,3,3,2,2,3,2,2,3,2,2,3,2,2,3,2,2,3,3,3,3,2,2,2,2,2,2,2};
    static constexpr uint8_t b1_3[36] = {4,4,5,4,4,4,4,5,5,5,5,4,4,5,5,5,5,4,4,5,4,5,5,4,4,5,4,5,5,4,4,4,5,4,4,4};
    static constexpr uint8_t s1_3[36] = {4,4,4,4,4,4,4,5,5,5,5,4,4,5,5,5,5,4,4,5,5,5,5,4,4,5,5,5,5,4,4,4,4,4,4,4};
    static constexpr uint8_t b1_4[36] = {0,4,0,0,0,0,4,4,0,0,4,0,4,0,4,4,0,0,0,0,4,4,0,0,0,4,0,0,4,0,0,0,4,0,0,0};
    static constexpr uint8_t s1_4[36] = {4,0,0,0,0,4,0,4,0,0,4,0,0,0,4,4,0,0,0,0,4,4,0,0,0,4,0,0,4,0,4,0,0,0,0,4};
    static constexpr uint8_t b1_5[36] = {7,7,7,7,7,7,7,2,2,2,7,7,7,2,7,7,2,7,7,2,7,7,2,7,2,7,2,2,2,2,7,7,7,7,7,7};
    static constexpr uint8_t s1_5[36] = {7,7,7,7,7,7,7,2,2,2,2,7,7,2,7,7,2,7,7,2,7,7,2,7,7,2,2,2,2,7,7,7,7,7,7,7};
    static constexpr uint8_t b2_1[36] = {6,0,0,0,0,6,0,0,6,6,0,0,6,6,6,6,6,0,6,6,6,6,0,6,0,0,6,6,6,0,0,0,6,6,6,0};
    static constexpr uint8_t s2_1[36] = {0,0,6,6,0,0,0,0,6,6,0,0,6,6,6,6,6,6,6,6,6,6,6,6,0,0,6,6,0,0,0,0,6,6,0,0};
    static constexpr uint8_t b2_2[36] = {0,2,2,2,2,2,2,0,0,0,2,2,0,0,3,3,0,2,2,0,3,3,2,2,2,0,0,0,2,2,0,2,2,2,2,2};
    static constexpr uint8_t s2_2[36] = {2,2,2,2,2,2,2,0,0,0,0,2,2,0,3,3,0,2,2,0,3,3,0,2,2,0,0,0,0,2,2,2,2,2,2,2};
    static constexpr uint8_t b2_3[36] = {2,6,6,2,6,2,6,2,2,6,2,6,6,6,6,2,6,2,6,2,2,6,2,6,2,6,2,2,6,2,2,6,2,6,6,2};
    static constexpr uint8_t s2_3[36] = {2,2,2,2,2,2,6,6,6,6,6,6,2,2,2,2,2,2,6,6,6,6,6,6,2,2,2,2,2,2,6,6,6,6,6,6};
    static constexpr uint8_t b2_4[36] = {1,1,5,4,1,1,1,4,4,4,1,1,1,5,4,4,4,4,4,1,4,5,4,4,1,1,4,4,1,1,4,1,5,1,1,4};
    static constexpr uint8_t s2_4[36] = {1,1,4,4,1,1,1,1,4,4,1,1,4,4,5,5,4,4,4,4,5,5,4,4,1,1,4,4,1,1,1,1,4,4,1,1};
    static constexpr uint8_t b2_5[36] = {6,0,0,0,0,6,6,6,6,0,0,0,6,6,6,0,0,0,6,0,6,6,0,0,6,6,6,6,6,0,6,6,0,6,6,6};
    static constexpr uint8_t s2_5[36] = {6,0,0,0,0,0,6,6,0,0,0,0,6,6,6,0,0,0,6,6,6,6,0,0,6,6,6,6,6,0,6,6,6,6,6,6};
    static constexpr uint8_t b3_1[36] = {6,2,2,2,2,2,2,2,2,2,7,2,6,7,6,6,6,6,6,2,6,6,6,6,7,2,7,7,7,7,7,6,7,7,7,7};
    static constexpr uint8_t s3_1[36] = {2,2,2,2,2,2,2,2,2,2,2,2,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7};
    static constexpr uint8_t b3_2[36] = {3,4,4,4,4,4,4,4,3,4,4,4,3,4,4,4,4,3,4,4,4,3,4,4,4,4,4,4,4,3,3,3,4,4,4,4};
    static constexpr uint8_t s3_2[36] = {3,4,4,4,4,3,4,4,4,4,4,4,4,4,3,3,4,4,4,4,3,3,4,4,4,4,4,4,4,4,3,4,4,4,4,3};
    static constexpr uint8_t b3_3[36] = {0,4,3,0,0,0,3,4,0,0,3,3,4,3,0,3,4,4,4,0,3,4,4,4,3,0,4,4,3,3,4,3,0,3,0,0};
    static constexpr uint8_t s3_3[36] = {0,0,0,0,0,0,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,0,0,0,0,0,0};
    static constexpr uint8_t b3_4[36] = {2,1,1,2,5,1,5,2,1,5,2,5,2,5,5,2,1,1,1,2,1,1,2,5,5,2,5,5,2,1,1,2,5,5,2,1};
    static constexpr uint8_t s3_4[36] = {1,2,5,5,2,1,1,2,5,5,2,1,1,2,5,5,2,1,1,2,5,5,2,1,1,2,5,5,2,1,1,2,5,5,2,1};
    static constexpr uint8_t b3_5[36] = {6,0,3,6,6,6,6,3,3,3,3,6,0,0,3,3,3,3,3,0,0,3,3,0,6,3,3,0,3,6,6,6,0,3,6,6};
    static constexpr uint8_t s3_5[36] = {6,6,0,0,6,6,6,3,3,3,3,6,0,3,3,3,3,0,0,3,3,3,3,0,6,3,3,3,3,6,6,6,0,0,6,6};
    static constexpr uint8_t b4_1[36] = {6,6,0,6,0,6,6,6,0,6,0,6,6,0,6,0,0,0,6,0,6,6,0,6,6,0,6,0,6,6,6,0,0,6,0,6};
    static constexpr uint8_t s4_1[36] = {6,6,6,6,6,6,6,0,0,0,0,6,6,0,6,6,0,6,6,0,6,0,0,6,6,0,6,6,6,6,6,0,0,0,0,0};
    static constexpr uint8_t b4_2[36] = {2,4,4,4,4,4,4,4,4,4,4,2,2,4,2,4,2,4,4,2,2,2,4,2,2,4,2,4,8,8,4,4,4,2,8,8};
    static constexpr uint8_t s4_2[36] = {4,4,4,4,2,4,4,2,2,2,4,4,2,4,4,2,4,4,2,4,4,2,2,4,4,2,2,2,8,8,4,4,4,4,8,8};
    static constexpr uint8_t b4_3[36] = {2,6,2,2,2,6,2,2,2,2,2,2,6,2,2,2,6,2,2,2,2,2,2,2,2,6,2,2,2,6,2,2,6,6,2,2};
    static constexpr uint8_t s4_3[36] = {2,2,2,2,2,2,2,6,2,6,2,2,2,2,6,2,6,2,2,6,2,6,2,2,2,2,6,2,6,2,2,2,2,2,2,2};
    static constexpr uint8_t b4_4[36] = {6,6,6,6,6,0,0,0,6,0,6,6,6,6,8,8,0,0,0,6,8,8,0,0,6,0,6,6,6,6,0,6,6,0,6,6};
    static constexpr uint8_t s4_4[36] = {6,6,0,0,6,6,6,0,6,6,0,6,0,6,8,8,6,0,0,6,8,8,6,0,6,0,6,6,0,6,6,6,0,0,6,6};
    static constexpr uint8_t b4_5[36] = {5,5,5,5,1,5,5,1,1,5,5,1,1,5,5,5,1,5,5,5,1,1,5,1,1,5,5,1,1,1,5,1,1,1,1,1};
    static constexpr uint8_t s4_5[36] = {5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1};
    static constexpr uint8_t b5_1[36] = {2,2,2,6,6,6,6,6,6,2,2,6,6,6,2,6,6,6,2,2,6,8,8,6,6,6,2,8,8,6,2,2,6,6,6,2};
    static constexpr uint8_t s5_1[36] = {6,6,6,6,6,6,6,2,2,2,2,6,6,2,2,2,2,6,6,2,2,8,8,6,6,2,2,8,8,6,6,6,6,6,6,6};
    static constexpr uint8_t b5_2[36] = {0,4,0,0,0,0,0,3,4,0,3,4,0,3,3,4,3,0,4,4,4,4,0,3,0,0,3,4,0,4,3,3,4,0,0,0};
    static constexpr uint8_t s5_2[36] = {0,0,3,4,0,0,0,0,3,4,0,0,3,3,3,4,3,3,4,4,4,4,4,4,0,0,3,4,0,0,0,0,3,4,0,0};
    static constexpr uint8_t b5_3[36] = {0,0,1,0,1,0,1,1,0,1,4,1,1,0,1,4,0,0,4,0,1,1,0,4,0,0,0,0,0,4,1,0,1,1,1,4};
    static constexpr uint8_t s5_3[36] = {0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,0,0,4,0,1,1,0,4,4,0,1,1,0,4,4,0,1,1,0,4};
    static constexpr uint8_t b5_4[36] = {4,4,4,1,4,4,1,1,4,4,4,1,1,1,1,4,1,4,4,4,1,4,4,1,4,4,4,4,1,4,1,4,1,4,1,1};
    static constexpr uint8_t s5_4[36] = {4,4,4,4,4,4,1,1,1,1,1,4,4,4,4,4,1,4,1,1,1,4,1,4,4,4,1,4,1,4,1,4,1,4,1,4};
    static constexpr uint8_t b5_5[36] = {3,4,4,4,0,0,4,4,4,3,0,3,0,3,3,4,3,0,3,3,0,0,0,4,0,3,0,3,4,4,0,4,3,3,4,0};
    static constexpr uint8_t s5_5[36] = {4,4,4,4,4,4,0,4,4,4,4,0,0,0,4,4,0,0,0,0,3,3,0,0,0,3,3,3,3,0,3,3,3,3,3,3};
    static constexpr uint8_t b6_1[36] = {3,4,0,4,0,3,3,0,0,3,3,3,4,0,8,8,0,4,0,3,8,8,4,3,0,0,4,4,3,0,4,3,0,0,3,3};
    static constexpr uint8_t s6_1[36] = {0,0,3,3,0,0,0,3,4,4,3,0,3,4,8,8,4,3,3,4,8,8,4,3,0,3,4,4,3,0,0,0,3,3,0,0};
    static constexpr uint8_t b6_2[36] = {2,4,6,3,3,3,4,4,3,4,6,4,6,3,8,8,3,6,3,3,8,8,2,3,4,6,4,2,4,3,2,6,6,6,3,3};
    static constexpr uint8_t s6_2[36] = {6,2,3,3,4,6,4,6,3,4,6,2,3,4,8,8,3,3,3,3,8,8,4,3,2,6,4,3,6,4,6,4,3,3,2,6};
    static constexpr uint8_t b6_3[36] = {0};
    static constexpr uint8_t s6_3[36] = {0};
    static constexpr uint8_t b6_4[36] = {0};
    static constexpr uint8_t s6_4[36] = {0};
    static constexpr uint8_t b6_5[36] = {2,6,7,6,6,6,2,6,6,2,6,7,7,6,8,8,6,6,6,6,8,8,6,7,6,6,7,2,7,7,2,2,6,2,7,2};
    static constexpr uint8_t s6_5[36] = {2,6,7,7,6,2,6,2,6,6,2,6,7,6,8,8,6,7,7,6,8,8,6,7,6,2,6,6,2,6,2,6,7,7,6,2};
    static constexpr uint8_t b7_1[36] = {2,2,2,2,6,7,2,7,6,7,2,7,6,6,7,2,6,6,2,2,2,7,2,7,6,6,7,6,2,2,7,7,2,7,6,7};
    static constexpr uint8_t s7_1[36] = {2,6,2,7,7,7,6,2,6,2,7,7,2,6,2,6,2,7,7,2,6,2,6,2,7,7,2,6,2,6,7,7,7,2,6,2};
    static constexpr uint8_t b7_2[36] = {4,4,0,3,4,4,4,4,0,3,3,4,0,4,0,0,4,4,4,4,4,3,3,4,4,0,3,0,0,3,4,4,4,4,3,4};
    static constexpr uint8_t s7_2[36] = {0,0,4,4,3,3,0,0,4,4,3,3,4,4,4,4,4,4,4,4,4,4,4,4,3,3,4,4,0,0,3,3,4,4,0,0};
    static constexpr uint8_t b7_3[36] = {1,1,6,1,1,1,6,1,6,6,6,6,6,6,1,6,6,1,1,1,6,6,6,6,1,1,1,1,6,6,6,1,1,1,6,6};
    static constexpr uint8_t s7_3[36] = {6,6,6,6,6,1,6,1,1,1,6,1,6,1,6,1,6,1,6,6,6,1,6,1,1,1,1,1,6,1,1,6,6,6,6,1};
    static constexpr uint8_t b7_4[36] = {2,4,4,2,1,2,4,4,2,1,4,4,4,4,1,2,2,1,2,1,4,1,2,1,4,1,1,1,4,2,1,2,2,4,2,1};
    static constexpr uint8_t s7_4[36] = {4,4,4,4,4,4,1,1,1,1,1,1,2,2,2,2,2,2,4,4,4,4,4,4,1,1,1,1,1,1,2,2,2,2,2,2};
    static constexpr uint8_t b7_5[36] = {0,0,0,6,6,5,0,0,5,5,0,5,0,6,0,0,0,0,6,5,0,0,5,6,0,5,0,5,6,6,5,5,0,5,5,6};
    static constexpr uint8_t s7_5[36] = {5,5,5,6,0,0,5,5,6,0,0,0,5,6,0,0,0,6,6,0,0,0,6,5,0,0,0,6,5,5,0,0,6,5,5,5};
    static constexpr uint8_t b8_1[36] = {4,1,2,2,4,1,4,2,4,2,4,4,4,4,4,1,1,2,1,2,1,4,2,1,2,4,1,1,1,2,2,1,2,1,4,2};
    static constexpr uint8_t s8_1[36] = {2,2,4,4,2,2,2,4,1,1,4,2,4,1,1,1,1,4,4,1,1,1,1,4,2,4,1,1,4,2,2,2,4,4,2,2};
    static constexpr uint8_t b8_2[36] = {6,2,6,2,2,6,6,6,2,6,6,6,6,6,8,8,2,2,2,2,8,8,6,2,6,6,6,2,6,6,6,2,6,2,6,6};
    static constexpr uint8_t s8_2[36] = {2,6,2,2,6,2,6,6,6,6,6,6,2,6,8,8,6,2,2,6,8,8,6,2,6,6,6,6,6,6,2,6,2,2,6,2};
    static constexpr uint8_t b8_3[36] = {2,6,6,6,6,2,1,1,1,6,1,6,6,6,2,6,2,6,6,1,6,2,1,1,2,2,2,6,1,2,6,2,1,1,1,1};
    static constexpr uint8_t s8_3[36] = {6,6,6,1,1,1,6,2,2,2,1,1,6,2,6,6,2,1,1,2,6,6,2,6,1,1,2,2,2,6,1,1,1,6,6,6};
    static constexpr uint8_t b8_4[36] = {3,0,0,0,0,0,3,3,3,3,3,3,0,3,8,8,3,3,0,3,8,8,0,3,3,0,0,3,0,3,3,0,0,0,0,0};
    static constexpr uint8_t s8_4[36] = {0,0,0,0,0,3,0,3,3,3,0,3,0,3,8,8,0,3,0,3,8,8,0,3,0,3,0,0,0,3,0,3,3,3,3,3};
    static constexpr uint8_t b8_5[36] = {1,1,6,6,1,6,6,6,6,1,6,1,1,6,1,1,1,6,6,1,1,1,1,6,6,6,6,1,1,6,6,1,1,1,6,6};
    static constexpr uint8_t s8_5[36] = {1,6,1,6,1,6,6,1,6,1,6,1,1,6,1,6,1,6,6,1,6,1,6,1,1,6,1,6,1,6,6,1,6,1,6,1};
    static constexpr uint8_t b9_1[36] = {2,2,2,6,2,2,2,2,6,6,2,6,6,6,8,8,2,6,2,2,8,8,2,6,2,6,2,6,2,6,2,2,6,2,2,2};
    static constexpr uint8_t s9_1[36] = {2,2,6,6,2,2,2,6,2,2,6,2,6,2,8,8,2,6,6,2,8,8,2,6,2,6,2,2,6,2,2,2,6,6,2,2};
    static constexpr uint8_t b9_2[36] = {6,0,6,0,6,6,0,0,0,2,0,0,0,6,2,2,0,6,6,6,2,6,2,2,6,2,0,0,2,0,2,2,6,2,6,2};
    static constexpr uint8_t s9_2[36] = {2,2,6,6,0,0,2,2,6,6,0,0,2,2,6,6,0,0,2,2,6,6,0,0,2,2,6,6,0,0,2,2,6,6,0,0};
    static constexpr uint8_t b9_3[36] = {0,1,1,0,1,6,1,0,3,6,4,1,0,0,6,6,0,0,6,6,0,1,1,1,3,4,1,1,0,0,1,0,1,6,6,0};
    static constexpr uint8_t s9_3[36] = {0,0,0,0,0,0,0,0,0,0,0,0,6,6,3,4,6,6,6,6,4,3,6,6,1,1,1,1,1,1,1,1,1,1,1,1};
    static constexpr uint8_t b9_4[36] = {6,6,6,2,6,6,2,2,6,2,2,6,6,6,6,2,6,6,6,6,6,2,6,2,6,6,6,6,2,6,2,6,2,6,2,6};
    static constexpr uint8_t s9_4[36] = {6,6,2,2,6,6,6,2,6,6,2,6,2,6,6,6,6,2,2,6,6,6,6,2,6,2,6,6,2,6,6,6,2,2,6,6};
    static constexpr uint8_t b9_5[36] = {6,6,2,6,0,6,6,0,6,6,6,6,0,6,6,2,2,6,6,2,6,0,2,6,6,6,6,2,6,0,6,6,0,6,6,6};
    static constexpr uint8_t s9_5[36] = {6,6,6,6,6,6,6,6,0,0,6,6,6,0,0,2,2,6,6,0,0,2,2,6,6,6,2,2,6,6,6,6,6,6,6,6};
    static constexpr uint8_t b10_1[36] = {1,4,0,0,0,1,1,1,1,4,4,0,0,4,4,0,1,0,1,1,4,1,0,4,1,4,1,0,1,1,0,4,0,4,1,1};
    static constexpr uint8_t s10_1[36] = {0,0,4,4,4,4,0,0,0,4,4,4,0,0,0,1,4,4,0,0,1,1,1,4,0,1,1,1,1,1,1,1,1,1,1,1};
    static constexpr uint8_t b10_2[36] = {3,3,3,1,4,4,4,4,3,0,0,1,3,0,3,1,0,4,0,4,1,4,1,1,3,4,0,0,1,1,3,3,1,0,0,4};
    static constexpr uint8_t s10_2[36] = {0,0,0,3,3,3,0,0,0,3,3,3,0,0,0,3,3,3,1,1,1,4,4,4,1,1,1,4,4,4,1,1,1,4,4,4};
    static constexpr uint8_t b10_3[36] = {5,5,4,0,5,5,5,5,3,3,0,0,0,3,5,4,5,0,4,5,3,5,4,0,5,5,5,4,3,5,5,3,5,5,5,4};
    static constexpr uint8_t s10_3[36] = {3,3,3,3,3,3,0,0,0,0,0,0,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5};
    static constexpr uint8_t b10_4[36] = {7,2,6,2,2,7,2,6,7,7,2,7,6,7,7,7,7,7,2,2,7,7,7,2,7,6,2,2,7,6,7,7,7,7,2,2};
    static constexpr uint8_t s10_4[36] = {7,7,7,7,7,7,7,2,2,2,2,7,7,2,6,6,2,7,7,2,6,6,2,7,7,2,2,6,2,7,7,7,7,7,2,7};
    static constexpr uint8_t b10_5[36] = {0,5,0,2,2,2,5,2,0,0,5,2,5,0,0,0,5,5,2,0,2,0,0,0,5,5,0,0,2,2,0,0,0,0,0,5};
    static constexpr uint8_t s10_5[36] = {0,0,0,2,2,2,0,0,0,2,2,2,0,0,0,2,2,2,0,0,0,5,5,5,0,0,0,5,5,5,0,0,0,5,5,5};
    static constexpr uint8_t b11_1[36] = {6,6,6,0,6,2,6,2,0,0,0,2,6,6,6,0,2,0,2,6,0,0,2,6,6,0,6,6,0,6,2,0,6,0,2,2};
    static constexpr uint8_t s11_1[36] = {2,2,2,0,0,0,2,2,2,6,6,6,2,2,2,0,0,0,6,6,6,6,6,6,0,0,0,0,0,0,6,6,6,6,6,6};
    static constexpr uint8_t b11_2[36] = {3,2,3,2,3,4,2,4,2,3,3,2,4,3,4,2,4,4,2,4,4,3,3,2,2,2,2,4,4,4,3,3,4,2,3,3};
    static constexpr uint8_t s11_2[36] = {2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4};
    static constexpr uint8_t b11_3[36] = {6,6,0,0,6,6,0,0,0,6,6,6,0,6,0,6,0,6,6,6,6,6,6,6,6,0,0,6,0,6,6,6,6,6,6,0};
    static constexpr uint8_t s11_3[36] = {6,6,6,6,6,6,6,6,0,0,6,6,6,0,0,0,0,6,6,0,0,0,0,6,6,6,0,0,6,6,6,6,6,6,6,6};
    static constexpr uint8_t b11_4[36] = {6,2,2,6,6,6,6,6,2,6,6,2,6,6,2,2,6,2,6,6,2,2,6,2,2,2,6,6,2,2,6,6,2,7,7,2};
    static constexpr uint8_t s11_4[36] = {2,6,2,2,2,2,6,6,6,7,6,6,2,6,2,2,2,2,6,7,6,6,6,6,2,2,2,2,2,2,6,6,6,6,6,6};
    static constexpr uint8_t b11_5[36] = {6,0,0,2,0,0,0,0,0,0,6,0,0,6,2,0,2,0,0,0,6,6,0,6,6,0,0,0,0,2,6,0,2,2,2,2};
    static constexpr uint8_t s11_5[36] = {6,2,0,0,2,6,2,6,0,0,6,2,0,0,0,0,0,0,0,0,0,0,0,0,2,6,0,0,6,2,6,2,0,0,2,6};
    static constexpr uint8_t b12_1[36] = {0,1,4,3,2,0,0,5,2,4,3,3,4,4,4,2,5,3,1,1,2,5,1,0,2,1,1,3,2,5,3,4,0,0,5,5};
    static constexpr uint8_t s12_1[36] = {0,0,0,0,0,0,3,3,3,3,3,3,4,4,4,4,4,4,1,1,1,1,1,1,2,2,2,2,2,2,5,5,5,5,5,5};
    static constexpr uint8_t b12_2[36] = {0}; // FAT
    static constexpr uint8_t s12_2[36] = {0}; // FAT
    static constexpr uint8_t b12_3[36] = {0}; // RRYYOO YWRORR OOYWOO ORYOYW WOYROO OYOORO
    static constexpr uint8_t s12_3[36] = {0}; // YYOORR YYOORR OOWWOO OOWWOO RROOYY RROOYY
    static constexpr uint8_t b12_4[36] = {2,7,6,7,2,2,6,7,6,6,6,2,2,2,6,2,2,7,2,7,6,2,2,7,6,6,6,2,2,2,6,2,2,2,6,2};
    static constexpr uint8_t s12_4[36] = {2,2,2,6,2,2,2,2,6,7,6,2,2,6,7,6,7,6,6,7,6,2,6,7,7,6,2,2,2,6,6,2,2,2,2,2};
    static constexpr uint8_t b12_5[36] = {0}; // BRYGGG YGBGYR ROOBOG YYBGOG GGRYRO YYORBB
    static constexpr uint8_t s12_5[36] = {0}; // ROYGBB ROYGGB ROYYGG ROYYGG ROYGGB ROYGBB
    static constexpr uint8_t b13_1[36] = {0}; // YORPOR POROOO OOYRYO OOORYR RPYOYY OOOYRY
    static constexpr uint8_t s13_1[36] = {0}; // ROOYYO OOYYOO OYYOOR YYOORR YOORRP OORRPP
    static constexpr uint8_t b13_2[36] = {1,6,6,6,4,6,6,6,6,6,4,4,4,6,4,6,6,4,6,6,1,4,4,1,1,6,4,4,6,6,4,6,6,4,6,6};
    static constexpr uint8_t s13_2[36] = {6,6,6,6,6,6,6,4,4,4,4,6,6,4,1,1,4,6,6,4,1,1,4,6,6,4,4,4,4,6,6,6,6,6,6,6};
    static constexpr uint8_t b13_3[36] = {0}; //
    static constexpr uint8_t s13_3[36] = {0}; //
    static constexpr uint8_t b13_4[36] = {0}; //
    static constexpr uint8_t s13_4[36] = {0}; //
    static constexpr uint8_t b13_5[36] = {0}; //
    static constexpr uint8_t s13_5[36] = {0}; //
    static constexpr uint8_t b14_1[36] = {0}; //
    static constexpr uint8_t s14_1[36] = {0}; //
    static constexpr uint8_t b14_2[36] = {0}; // BOBYBY YBRBOO BRBBYR BBOBYO OOBBBB YYROBY
    static constexpr uint8_t s14_2[36] = {0}; // BBYYBB BBOOBB YORROY YORROY BBOOBB BBYYBB
    static constexpr uint8_t b14_3[36] = {0,6,2,0,6,6,6,1,3,0,1,6,2,4,6,6,5,5,4,4,5,0,1,5,6,2,6,1,2,3,6,3,3,6,6,4};
    static constexpr uint8_t s14_3[36] = {0,3,4,1,2,5,0,3,4,1,2,5,6,6,6,6,6,6,6,6,6,6,6,6,5,2,1,4,3,0,5,2,1,4,3,0};
    static constexpr uint8_t b14_4[36] = {0}; //
    static constexpr uint8_t s14_4[36] = {0}; //
    static constexpr uint8_t b14_5[36] = {0}; //
    static constexpr uint8_t s14_5[36] = {0}; //
    static constexpr uint8_t b15_1[36] = {0}; //
    static constexpr uint8_t s15_1[36] = {0}; //
    static constexpr uint8_t b15_2[36] = {0}; // FAT
    static constexpr uint8_t s15_2[36] = {0}; // FAT
    static constexpr uint8_t b15_3[36] = {0}; // FAT
    static constexpr uint8_t s15_3[36] = {0}; // FAT
    static constexpr uint8_t b15_4[36] = {0}; // FAT
    static constexpr uint8_t s15_4[36] = {0}; // FAT
    static constexpr uint8_t b15_5[36] = {0}; //
    static constexpr uint8_t s15_5[36] = {0}; //
    static constexpr uint8_t b16_1[36] = {0}; // FAT
    static constexpr uint8_t s16_1[36] = {0}; // FAT
    static constexpr uint8_t b16_2[36] = {0}; //
    static constexpr uint8_t s16_2[36] = {0}; //
    static constexpr uint8_t b16_3[36] = {0}; //
    static constexpr uint8_t s16_3[36] = {0}; //
    static constexpr uint8_t b16_4[36] = {0}; //
    static constexpr uint8_t s16_4[36] = {0}; //
    static constexpr uint8_t b16_5[36] = {0}; // FAT
    static constexpr uint8_t s16_5[36] = {0}; // FAT
    static constexpr uint8_t b17_1[36] = {0}; //
    static constexpr uint8_t s17_1[36] = {0}; //
    static constexpr uint8_t b17_2[36] = {0}; // FAT
    static constexpr uint8_t s17_2[36] = {0}; // FAT
    static constexpr uint8_t b17_3[36] = {0}; //
    static constexpr uint8_t s17_3[36] = {0}; //
    static constexpr uint8_t b17_4[36] = {0}; // FAT
    static constexpr uint8_t s17_4[36] = {0}; // FAT
    static constexpr uint8_t b17_5[36] = {0}; //
    static constexpr uint8_t s17_5[36] = {0}; //
    static constexpr uint8_t b18_1[36] = {0}; // FAT
    static constexpr uint8_t s18_1[36] = {0}; // FAT
    static constexpr uint8_t b18_2[36] = {0}; // FAT
    static constexpr uint8_t s18_2[36] = {0}; // FAT
    static constexpr uint8_t b18_3[36] = {0}; //
    static constexpr uint8_t s18_3[36] = {0}; //
    static constexpr uint8_t b18_4[36] = {0}; // FAT
    static constexpr uint8_t s18_4[36] = {0}; // FAT
    static constexpr uint8_t b18_5[36] = {0}; // FAT
    static constexpr uint8_t s18_5[36] = {0}; // FAT
    static constexpr uint8_t b19_1[36] = {0}; //
    static constexpr uint8_t s19_1[36] = {0}; //
    static constexpr uint8_t b19_2[36] = {0}; // FAT
    static constexpr uint8_t s19_2[36] = {0}; // FAT
    static constexpr uint8_t b19_3[36] = {0}; //
    static constexpr uint8_t s19_3[36] = {0}; //
    static constexpr uint8_t b19_4[36] = {0}; // FAT
    static constexpr uint8_t s19_4[36] = {0}; // FAT
    static constexpr uint8_t b19_5[36] = {0}; //
    static constexpr uint8_t s19_5[36] = {0}; //
    static constexpr uint8_t b20_1[36] = {0}; //
    static constexpr uint8_t s20_1[36] = {0}; //
    static constexpr uint8_t b20_2[36] = {0}; //
    static constexpr uint8_t s20_2[36] = {0}; //
    static constexpr uint8_t b20_3[36] = {0}; // FAT
    static constexpr uint8_t s20_3[36] = {0}; // FAT
    static constexpr uint8_t b20_4[36] = {0}; //
    static constexpr uint8_t s20_4[36] = {0}; //
    static constexpr uint8_t b20_5[36] = {0}; //
    static constexpr uint8_t s20_5[36] = {0}; //
};