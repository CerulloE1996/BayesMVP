#pragma once 


#ifndef OTHER_CONFIG_H
#define OTHER_CONFIG_H


//// Define global custom-double's (i.e., non-vectorised / non-AVX) fn typedef
typedef double (*FuncDouble)(const double);


//// Define global inlining macro
#ifdef _WIN32
    #if defined(__GNUC__) || defined(__clang__)
        #define ALWAYS_INLINE __attribute__((always_inline)) inline
    #elif defined(_MSC_VER)
        #define ALWAYS_INLINE __forceinline
    #else
        #define ALWAYS_INLINE inline
    #endif
#else
    #if defined(__GNUC__) || defined(__clang__)
        #define ALWAYS_INLINE __attribute__((always_inline)) inline
    #else
        #define ALWAYS_INLINE inline
    #endif
#endif


//// Define global inlining macro
#ifdef _WIN32
        #define MAYBE_INLINE inline // Don't force inlining on Windows as very slow compilation (+ questionable benefits)
#else //// If Linux or Mac OS (in which case we do some more aggressive inlining)
        #if defined(__GNUC__) || defined(__clang__)
            #define MAYBE_INLINE __attribute__((always_inline)) inline
        #else
            #define MAYBE_INLINE inline
        #endif
#endif

 
// #define ALWAYS_INLINE inline
// #define MAYBE_INLINE inline


#endif
 
 