#pragma once

#include <cstdint>
#include <type_traits>


///=============================================================================
///                      Compiler and Platform Features
///=============================================================================

#if defined(__GNUC__) || defined(__clang__)
    #define PREFETCH(PTR, RW, LOC) __builtin_prefetch(PTR, RW, LOC)
    #define EXPECT_FALSE(COND) (__builtin_expect((COND), 0)) // [unlikely]
    #define EXPECT_TRUE(COND) (__builtin_expect((COND), 1))  // [likely]
    #define ATTR(...) __attribute__((__VA_ARGS__))
#else
    #define PREFETCH(PTR, RW, LOC)
    #if __cplusplus >= 202002L
        #define EXPECT_FALSE(COND) (COND) [[unlikely]]
        #define EXPECT_TRUE(COND) (COND) [[likely]]
    #else
        #define EXPECT_FALSE(COND) (COND)
        #define EXPECT_TRUE(COND) (COND)
    #endif
    #define ATTR(...)
#endif


#if defined(_MSC_VER)
#define FORCEINLINE __forceinline
#define NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define FORCEINLINE inline __attribute__((always_inline))
#define NOINLINE __attribute__((noinline))
#else
#define FORCEINLINE inline
#define NOINLINE
#endif


#ifndef __CUDACC__
    #ifndef __host__
        #define __host__
    #endif
    #ifndef __device__
        #define __device__
    #endif
    #ifndef __constant__
        #define __constant__
    #endif
#endif

#define HD __host__ __device__


#define ND [[nodiscard]]
#define MU [[maybe_unused]]
#define MUND [[maybe_unused]] [[nodiscard]]

#define const const

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

template <typename T>
constexpr auto always_false = false;

#define is_instance(val, type) constexpr (std::is_same_v<decltype(val), type>)

#define not_instance(val) static_assert(always_false<decltype(val)>, "Unsupported type!")