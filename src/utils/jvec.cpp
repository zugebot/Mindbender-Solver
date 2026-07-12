#include "jvec.hpp"

#include <cstdio>
#include <cstring>


void _JVEC_HIDDEN_PRINTF(const char* str, unsigned long long num) {
    printf(str, num);
}


void _JVEC_HIDDEN_MEMCPY(void *dst, const void *src, unsigned long long size) {
    memcpy(dst, src, size);
}