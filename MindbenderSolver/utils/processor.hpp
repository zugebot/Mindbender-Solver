#pragma once

#include <cstdint>
#include <type_traits>


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

#define C const

#define i8 int8_t
#define i16 int16_t
#define i32 int32_t
#define i64 int64_t

#define u8 uint8_t
#define u16 uint16_t
#define u32 uint32_t
#define u64 uint64_t

template <typename T>
constexpr auto always_false = false;

#define is_instance(val, type) constexpr (std::is_same_v<decltype(val), type>)

#define not_instance(val) static_assert(always_false<decltype(val)>, "Unsupported type!")