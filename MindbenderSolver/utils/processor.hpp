#pragma once

#include <cstdint>

///=============================================================================
///                      Compiler and Platform Features
///=============================================================================

#if __GNUC__
#define PREFETCH(PTR, RW, LOC) __builtin_prefetch(PTR, RW, LOC)
#define EXPECT_FALSE(COND) (__builtin_expect((COND), 0)) // [unlikely]
#define EXPECT_TRUE(COND) (__builtin_expect((COND), 1))  // [likely]
#define ATTR(...) __attribute__((__VA_ARGS__))
#else
#define PREFETCH(PTR, RW, LOC)
#define EXPECT_FALSE(COND) (COND) [[unlikely]]
#define EXPECT_TRUE(COND) (COND) [[likely]]
#define ATTR(...)
#endif

#define ND [[nodiscard]]
#define MU [[maybe_unused]]
#define MUND [[maybe_unused]] [[nodiscard]]

#define c_char const char
#define c_int const int
#define c_bool const bool
#define c_auto const auto
#define c_float const float
#define c_double const double
#define c_string const std::string

#define i8 int8_t
#define i16 int16_t
#define i32 int32_t
#define i64 int64_t

#define u8 uint8_t
#define u16 uint16_t
#define u32 uint32_t
#define u64 uint64_t

#define c_i8 const int8_t
#define c_i16 const int16_t
#define c_i32 const int32_t
#define c_i64 const int64_t

#define c_u8 const uint8_t
#define c_u16 const uint16_t
#define c_u32 const uint32_t
#define c_u64 const uint64_t
