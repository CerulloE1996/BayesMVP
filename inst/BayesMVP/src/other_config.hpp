#ifndef OTHER_CONFIG_H
#define OTHER_CONFIG_H
 
 
 
//// Define global inlining macro
#if defined(__GNUC__) || defined(__clang__)
    #define ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
    #define ALWAYS_INLINE __forceinline
#else
    #define ALWAYS_INLINE inline
#endif
 
//// Define global custom-double's (i.e., non-vectorised / non-AVX) fn typedef
typedef double (*FuncDouble)(const double);
 
 



#endif
 





















