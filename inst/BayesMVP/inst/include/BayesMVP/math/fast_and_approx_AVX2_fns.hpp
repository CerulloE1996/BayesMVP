
#pragma once 


#ifndef FAST_AND_APPROX_AVX2_FNS_HPP
#define FAST_AND_APPROX_AVX2_FNS_HPP


#include <immintrin.h>
#include <cmath>
  
  
// #include <R_ext/Print.h>  // For REprintf
// #include <R_ext/Utils.h>  // For R_FlushConsole
  
  
//// #if defined(__AVX2__) && !(defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)) // use AVX2 if AVX-512 not available 
#if defined(__AVX2__)   // compiled on every AVX2-capable build, including AVX-512 ones — the dispatch layer needs both kernel sets

//// -------------------------------------------------------------------------------------------------------------------------------------------------------------


// Simple test function that just multiplies vector by 2
inline __m256d  test_simple_AVX2 VECTORCALL(const __m256d x) {
   
       // // _mm256_zeroupper();  // Reset AVX state
       ALIGN32  __m256d const two = _mm256_set1_pd(2.0);
       ALIGN32  __m256d const res = _mm256_mul_pd(x, two);
       //// _mm256_zeroupper();  // Reset AVX state
       return res; 
   
}




//// -------------------------------------------------------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE __m256d   _mm256_abs_pd VECTORCALL(const __m256d x) {
 
       // _mm256_zeroupper();  // Reset AVX state
       
       ALIGN32  __m256d const x_aligned = x;
       
       ALIGN32  __m256d const sign_bit = _mm256_set1_pd(-0.0);
       ALIGN32  __m256d const x_abs = _mm256_andnot_pd(sign_bit, x_aligned);
       
       // _mm256_zeroupper();  // Reset AVX state
       
       return x_abs;
 
}




// is_finite_mask and is_not_NaN_mask for AVX2
ALWAYS_INLINE __m256d  is_finite_mask VECTORCALL(const __m256d x) {
  
       // _mm256_zeroupper();  // Reset AVX state
  
       ALIGN32  __m256d const x_aligned = x;
       
       ALIGN32  __m256d const INF = _mm256_set1_pd(std::numeric_limits<double>::infinity());
       
       ALIGN32  __m256d const abs_x = _mm256_abs_pd(x_aligned);
       
       ALIGN32  __m256d const res = _mm256_cmp_pd(abs_x, INF, _CMP_NEQ_OQ);
       
       return res; 
   
}




ALWAYS_INLINE __m256d is_not_NaN_mask VECTORCALL(const __m256d x) {
  
       // _mm256_zeroupper();  // Reset AVX state
  
       ALIGN32 __m256d const x_aligned = x;
  
       ALIGN32 __m256d const res = _mm256_cmp_pd(x_aligned, x_aligned, _CMP_EQ_OQ);
       // _mm256_zeroupper();  // Reset AVX state
       
       return res;
 
}




// bookmark (this doesnt seem to be available on g++ at all at time of writing even with all AVX256 flags!)
//#if !defined(__AVX256VL__)
#define _mm256_and_epi64(a, b) _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b))) // this uses AVX2 instructions so moree widely supported




///////////////////  fns - exp   -----------------------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE __m256d   fast_ldexp VECTORCALL(  const __m256d AVX_a,
                                                const __m256i AVX_i) {
  
       // _mm256_zeroupper();  // Reset AVX state
    
       ALIGN32  __m256d const AVX_a_aligned = AVX_a;
       ALIGN32  __m256i const AVX_i_aligned = AVX_i;
    
       ALIGN32 const uint64_t shift_val_52 = 52;
       ALIGN32  __m256i const shifted = _mm256_slli_epi64(AVX_i_aligned, shift_val_52);
       ALIGN32  __m256i const a_bits = _mm256_castpd_si256(AVX_a_aligned);
       ALIGN32  __m256i const result = _mm256_add_epi64(shifted, a_bits);
      
       ALIGN32  __m256d const res = _mm256_castsi256_pd(result);
       // _mm256_zeroupper();  // Reset AVX state
       return res;
  
}




// // Replace _mm256_srai_epi64 (AVX-512 fn) with this helper function (forAVX2)
// ALWAYS_INLINE __m256i   avx2_srai_epi64 VECTORCALL(  const __m256i x, 
//                                                      const int count) {
//   
//        // _mm256_zeroupper();  // Reset AVX state
//     
//        ALIGN32  __m256i const x_aligned = x;
//     
//        // Get the sign bits
//        ALIGN32  __m256i const sign_bits = _mm256_and_si256(_mm256_srai_epi32(_mm256_permute4x64_epi64(x_aligned, 0xF5), 31),
//                                                               _mm256_set1_epi64x(0xFFFFFFFF00000000));
//        
//        // Perform arithmetic shift on lower 32 bits
//        ALIGN32  __m256i const shift_result = _mm256_srai_epi32(x_aligned, count);
//        
//        // Combine results
//        ALIGN32  __m256i const res = _mm256_or_si256(shift_result,  
//                                          _mm256_and_si256(sign_bits, 
//                                                           _mm256_set1_epi64x(0xFFFFFFFF00000000))); 
//        // _mm256_zeroupper();  // Reset AVX state
//        return res;
//      
// }




// // Fast implementation of ldexp (multiply by power of 2) using AVX2 intrinsics
// // Computes: a * 2^i for 4 pairs of values simultaneously
// // Handles edge cases by splitting large exponents into two steps
// ALWAYS_INLINE __m256d fast_ldexp_2 VECTORCALL( const __m256d AVX_a,
//                                                const __m256i AVX_i) {
//   
//         // _mm256_zeroupper();  // Reset AVX state
//   
//         ALIGN32  __m256d const AVX_a_aligned = AVX_a;
//         ALIGN32  __m256i const AVX_i_aligned = AVX_i;
//   
//         const uint64_t shift_val_52 = 52;  // Double-precision exponent shift
//         const uint64_t shift_val_63 = 63;  // Sign bit position
//         const uint64_t shift_val_1000 = 1000;  // Safe threshold for exponents
//         const uint64_t shift_val_neg_2000 = 2000;  // Double threshold for negative case
//         const uint64_t mask = 0xF;  // 4-bit comparison mask (one bit per double)
//   
//         // Get sign mask: for each 64-bit integer, shift right by 63 bits (arithmetic shift)
//         // This replicates the sign bit across all bits: 0x0000... for positive, 0xFFFF... for negative    ;
//         ALIGN32  __m256i const neg_mask = avx2_srai_epi64(AVX_i_aligned, shift_val_63);
//   
//         // Calculate absolute value using the formula: abs(x) = (x XOR sign_mask) - sign_mask
//         // 1. XOR with sign mask flips all bits if negative, keeps same if positive
//         // 2. Subtract sign mask adds 1 for negative numbers (two's complement)
//         ALIGN32  __m256i const abs_i = _mm256_sub_epi64(_mm256_xor_si256(AVX_i_aligned, neg_mask), neg_mask);
//   
//         // Create vector with all elements = 1000 (our safe threshold for exponents)
//         ALIGN32  __m256i const threshold = _mm256_set1_epi64x(shift_val_1000);
//   
//         // Compare abs_i with threshold by subtraction
//         // If abs_i > threshold, result will be positive
//         ALIGN32  __m256i const cmp = _mm256_sub_epi64(abs_i, threshold);
//   
//         // Create comparison mask by getting sign bits of comparison result
//         // Will be 0xFFFF... where abs_i <= threshold, 0x0000... where abs_i > threshold
//         ALIGN32  __m256i const cmp_mask = avx2_srai_epi64(cmp, shift_val_63);
//   
//         // Convert comparison mask to 4-bit integer (one bit per double)
//         // If result != 0xF (not all bits set), then at least one value exceeded threshold
//         if (_mm256_movemask_pd(_mm256_castsi256_pd(cmp_mask)) != mask) {
//   
//                 // Handle large exponents by splitting into two steps
//   
//                 // Create i1 = ±1000 based on original sign:
//                 // 1. AND neg_mask with -2000 gives -2000 for negative numbers, 0 for positive
//                 // 2. XOR with threshold (1000) gives -1000 for negative numbers, 1000 for positive
//                 ALIGN32  __m256i const i1 = _mm256_xor_si256(_mm256_and_si256(neg_mask, _mm256_set1_epi64x(shift_val_neg_2000)), threshold);
//   
//                 // Calculate remaining exponent: i2 = original_i - i1
//                 ALIGN32  __m256i const i2 = _mm256_sub_epi64(AVX_i_aligned, i1);
//   
//                 // First scaling step: multiply by 2^i1
//                 // 1. Shift i1 left by 52 to position it in double's exponent field
//                 // 2. Add to bit pattern of input doubles (effectively multiplying by 2^i1)
//                 ALIGN32  __m256d const mid = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_slli_epi64(i1, shift_val_52), _mm256_castpd_si256(AVX_a)));
//   
//                 // Second scaling step: multiply intermediate result by 2^i2
//                 // Same process as above but with i2 and intermediate result
//                 // return _mm256_castsi256_pd(_mm256_add_epi64(_mm256_slli_epi64(i2, 52), _mm256_castpd_si256(mid)));
//                 ALIGN32  __m256d const res = fast_ldexp(mid, i2);
//                 return  res;
//   
//         }
//   
//         // For small exponents, perform direct scaling:
//         // 1. Shift exponent left by 52 bits to align with double's exponent field
//         // 2. Add to original number's bit pattern (effectively multiplying by 2^i)
//         // return _mm256_castsi256_pd(_mm256_add_epi64(_mm256_slli_epi64(AVX_i, 52), _mm256_castpd_si256(AVX_a)));
//         ALIGN32  __m256d const res = fast_ldexp(AVX_a_aligned, AVX_i_aligned);
//         // _mm256_zeroupper();  // Reset AVX state
//         return res;
// 
// }




 
 


// Helper function for AVX2 64-bit conversion 
// ALWAYS_INLINE __m256i avx2_cvtpd_epi64 VECTORCALL(const __m256d x) {
//   
//         ALIGN32  __m256d const x_aligned = x;
//    
//         // Extract doubles and convert to int64 one at a tim
//         ALIGN32 int64_t result[4];
//         ALIGN32 double temp[4];
//   
//         _mm256_storeu_pd(temp, x_aligned);
//        
//         for(int i = 0; i < 4; i++) {
//             #ifdef _MSC_VER
//                    result[i] = (int64_t)_mm_cvtsd_si64(_mm_load_sd(&temp[i]));  // Use SSE2 conversion
//             #else
//                    result[i] = (int64_t)std::llrint(temp[i]);     // Use llrint for proper rounding
//             #endif
//             // result[i] = (int64_t)_mm_cvtsd_si64(_mm_load_sd(&temp[i]));  // Use SSE2 conversion
//         }
//        
//         //// return _mm256_load_si256((__m256i*)result);
//         ALIGN32  __m256i const res = _mm256_load_si256(reinterpret_cast<const __m256i*>(static_cast<void*>(result))); 
//         return res;
//  
// }
// Round-to-nearest-even double -> int64 for |x| < 2^51. Matches _mm512_cvtpd_epi64 under default
// MXCSR, and matches std::llrint on the current path (|t| < ~1022 whenever the checked exp is used).
ALWAYS_INLINE __m256i avx2_cvtpd_epi64 VECTORCALL(const __m256d x) {
    
        ALIGN32 __m256d const magic = _mm256_set1_pd(6755399441055744.0);   // 1.5 * 2^52
        ALIGN32 __m256d const y     = _mm256_add_pd(x, magic);
        return _mm256_sub_epi64(_mm256_castpd_si256(y), _mm256_castpd_si256(magic));
      
}
 
 
// Adapted from: https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
// added   (optional) extra degree(s) for poly approx (oroginal float fn had 4 degrees) - using "minimaxApprox" R package to find coefficient terms
// R code:    minimaxApprox::minimaxApprox(fn = exp, lower = -0.346573590279972643113, upper = 0.346573590279972643113, degree = 5, basis ="Chebyshev")
ALWAYS_INLINE __m256d fast_exp_1_wo_checks_AVX2  VECTORCALL(const __m256d x)  { 
  
        // REprintf("Entering fast_exp_1_wo_checks_AVX2 \n");
        // R_FlushConsole();
      
        // _mm256_zeroupper();  // Reset AVX state
      
        // REprintf("Before x_aligned \n");
        // R_FlushConsole();
        ALIGN32  __m256d const x_aligned = x;
        // REprintf("After x_aligned \n");
        // R_FlushConsole();
        
        ALIGN32  __m256d const exp_l2e = _mm256_set1_pd (1.442695040888963387); /* log2(e) */
        ALIGN32  __m256d const exp_l2h = _mm256_set1_pd (-0.693145751953125); /* -log(2)_hi, exactly representable */
        ALIGN32  __m256d const exp_l2l = _mm256_set1_pd (-1.42860682030941723212e-6); /* -log(2)_lo */
        
        // /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
        ALIGN32  __m256d const exp_c0 =     _mm256_set1_pd(0.00000276479776161191821278);
        ALIGN32  __m256d const exp_c1 =     _mm256_set1_pd(0.0000248844480527491290235);
        ALIGN32  __m256d const exp_c2 =     _mm256_set1_pd(0.000198411488032534342194);
        ALIGN32  __m256d const exp_c3 =     _mm256_set1_pd(0.00138888017711994078175);
        ALIGN32  __m256d const exp_c4 =     _mm256_set1_pd(0.00833333340524595143906);
        ALIGN32  __m256d const exp_c5 =     _mm256_set1_pd(0.0416666670404215802592);
        ALIGN32  __m256d const exp_c6 =     _mm256_set1_pd(0.166666666664891632843);
        ALIGN32  __m256d const exp_c7 =     _mm256_set1_pd(0.499999999994389376923);
        ALIGN32  __m256d const exp_c8 =     _mm256_set1_pd(1.00000000000001221245);
        ALIGN32  __m256d const exp_c9 =     _mm256_set1_pd(1.00000000000001332268);
        
        ALIGN32  __m256d const input  = x_aligned;
        
        /* exp(x) = 2^i * e^f; i = rint (log2(e) * a), f = a - log(2) * i */
        ALIGN32  __m256d const t = _mm256_mul_pd(x_aligned, exp_l2e);      /* t = log2(e) * a */
        ///  const __m256i i = _mm256_cvttpd_epi32(t);       /* i = (int)rint(t) */
        // REprintf("Before avx2_cvtpd_epi64 \n");
        // R_FlushConsole();
        ALIGN32  __m256i const i = avx2_cvtpd_epi64(t);  
        // REprintf("After avx2_cvtpd_epi64 \n");
        // R_FlushConsole();
        // REprintf("Before _mm256_round_pd \n");
        // R_FlushConsole();
        // const __m256d x_2 = _mm256_round_pd(t, _MM_FROUND_TO_NEAREST_INT) ; // ((0<<4)| _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC|_MM_FROUND_NO_EXC));
        ALIGN32  __m256d const x_2 = _mm256_round_pd(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // REprintf("After _mm256_round_pd \n");
        // R_FlushConsole();
        // REprintf("Before _mm256_fmadd_pd \n");
        // R_FlushConsole();
        ALIGN32  __m256d const f0 = _mm256_fmadd_pd(x_2, exp_l2h, input);
        ALIGN32  __m256d const f =  _mm256_fmadd_pd(x_2, exp_l2l, f0);  /* a - log(2)_hi * r */    /* f = a - log(2)_hi * r - log(2)_lo * r */
        // REprintf("After _mm256_fmadd_pd \n");
        // R_FlushConsole();
        
        /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
        ALIGN32 __m256d p = exp_c0;
        p = _mm256_fmadd_pd(p, f, exp_c1);
        p = _mm256_fmadd_pd(p, f, exp_c2);
        p = _mm256_fmadd_pd(p, f, exp_c3);
        p = _mm256_fmadd_pd(p, f, exp_c4);
        p = _mm256_fmadd_pd(p, f, exp_c5);
        p = _mm256_fmadd_pd(p, f, exp_c6);
        p = _mm256_fmadd_pd(p, f, exp_c7);
        p = _mm256_fmadd_pd(p, f, exp_c8);
        p = _mm256_fmadd_pd(p, f, exp_c9);
        
        // REprintf("Before fast_ldexp \n");
        // R_FlushConsole();
        // ALIGN32  __m256d const res = fast_ldexp_2(p, i);   /* exp(x) = 2^i * p */
        ALIGN32 __m256d const res = fast_ldexp(p, i);
        // REprintf("After fast_ldexp \n");
        // R_FlushConsole();
        // _mm256_zeroupper();  // Reset AVX state
        return  res;  
  
}




// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// ALWAYS_INLINE __m256d fast_exp_1_AVX2  VECTORCALL(const __m256d a) {
//   
//         // _mm256_zeroupper();  // Reset AVX state
//         
//         ALIGN32  __m256d const   exp_bound  =   _mm256_set1_pd(708.4);
//         ALIGN32  __m256d const   pos_inf  =    _mm256_set1_pd(INFINITY);
//         ALIGN32  __m256d const   zero  =   _mm256_setzero_pd();
//         
//         ALIGN32  __m256d const res = _mm256_blendv_pd(
//                                       _mm256_blendv_pd(
//                                         _mm256_blendv_pd( 
//                                           pos_inf,
//                                           zero,
//                                           _mm256_cmp_pd(a, zero, _CMP_LT_OQ)
//                                         ),
//                                         _mm256_set1_pd(INFINITY - INFINITY),
//                                         is_not_NaN_mask(a)
//                                       ),  
//                                       fast_exp_1_wo_checks_AVX2(a),
//                                       _mm256_cmp_pd(_mm256_abs_pd(a), exp_bound, _CMP_LT_OQ)
//                                     );
//         
//         // _mm256_zeroupper();  // Reset AVX state
//         return res;
//         
// }


// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// Keep exceptional-lane libm calls out of the inlined SIMD hot path.
__attribute__((noinline, cold)) inline __m256d fast_exp_exceptional_lanes_AVX2(const __m256d input, const __m256d result, const unsigned ordinary_mask) {
    alignas(32) double input_values[4], output_values[4];
    _mm256_store_pd(input_values, input);
    _mm256_store_pd(output_values, result);
    for (int lane = 0; lane < 4; ++lane) {
        if (!(ordinary_mask & (1u << lane))) output_values[lane] = std::exp(input_values[lane]);
    }
    return _mm256_load_pd(output_values);
}

ALWAYS_INLINE __m256d fast_exp_1_AVX2 VECTORCALL(const __m256d a) {

      // 708.4 bounds the exponent-bit shortcut, NOT the domain of finite exp().
      const __m256d ordinary_lanes = _mm256_cmp_pd(_mm256_abs_pd(a), _mm256_set1_pd(708.4), _CMP_LT_OQ);
      const int ordinary_mask = _mm256_movemask_pd(ordinary_lanes);
      const __m256d result = fast_exp_1_wo_checks_AVX2(a);
      if (__builtin_expect(ordinary_mask != 0xf, 0)) return fast_exp_exceptional_lanes_AVX2(a, result, ordinary_mask);
      return result;
}
 
 











///////////////////  fns - log  (+ related e.g. log1m etc) -----------------------------------------------------------------------------------------------------------------
// ALWAYS_INLINE __m256d avx2_cvtepi64_pd VECTORCALL(const __m256i v) {
//   
//       // Extract high and low 64-bit integers
//       ALIGN32 __m128i const low =  _mm256_castsi256_si128(v);
//       ALIGN32 __m128i const high = _mm256_extracti128_si256(v, 1);
//       
//       // Convert to doubles one element at a time
//       ALIGN32 double vals[4];
//       vals[0] = (double)_mm_extract_epi64(low, 0);
//       vals[1] = (double)_mm_extract_epi64(low, 1);
//       vals[2] = (double)_mm_extract_epi64(high, 0);
//       vals[3] = (double)_mm_extract_epi64(high, 1);
//       
//       // Load back into __m256d
//       return _mm256_loadu_pd(vals);
//   
// }
////
// Only for the exponent field `e` in fast_log_1_wo_checks_AVX2 (low 52 bits are zero).
// Returns (double)e exactly; keep the following fmadd by log_c1 as-is so the structure matches AVX-512.
ALWAYS_INLINE __m256d avx2_cvtepi64_pd VECTORCALL(const __m256i e) {
  
      ALIGN32 __m256i const hi = _mm256_permutevar8x32_epi32(e, _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7)); // high dword of each lane
      ALIGN32 __m128i const E  = _mm_srai_epi32(_mm256_castsi256_si128(hi), 20);                            // == e >> 52
      ALIGN32 __m256d const Ed = _mm256_cvtepi32_pd(E);                                                     // exact
      return _mm256_mul_pd(Ed, _mm256_set1_pd(4503599627370496.0));                                        // * 2^52, exact
      
}




//// https://stackoverflow.com/a/65537754/9007125 // vectorized version of the answer by njuffa
// ALWAYS_INLINE     __m256d fast_log_1_wo_checks_AVX2 VECTORCALL(const __m256d a ) {
//   
//       ALIGN32  __m256i const log_i1 =   _mm256_set1_epi64x(0x3fe5555555555555);
//       ALIGN32  __m256i const log_i2 =   _mm256_set1_epi64x(0xFFF0000000000000);
//       
//       ALIGN32  __m256d const log_c1 =   _mm256_set1_pd(0.000000000000000222044604925031308085);
//       ALIGN32  __m256d const log_c2 =   _mm256_set1_pd(-0.13031005859375);
//       ALIGN32  __m256d const log_c3 =   _mm256_set1_pd(0.140869140625);
//       ALIGN32  __m256d const log_c4 =   _mm256_set1_pd(-0.121483256222766876221);
//       ALIGN32  __m256d const log_c5 =   _mm256_set1_pd(0.139814853668212890625);
//       ALIGN32  __m256d const log_c6 =   _mm256_set1_pd(-0.166846126317977905273);
//       ALIGN32  __m256d const log_c7 =   _mm256_set1_pd(0.200120344758033752441);
//       ALIGN32  __m256d const log_c8 =   _mm256_set1_pd(-0.249996200203895568848);
//       ALIGN32  __m256d const log_c9 =   _mm256_set1_pd(0.333331972360610961914);
//       ALIGN32  __m256d const log_c10 =  _mm256_set1_pd(-0.5);
//       ALIGN32  __m256d const log_c11 =  _mm256_set1_pd(0.693147182464599609375);
//       
//       
//       ALIGN32  __m256i const aInt = _mm256_castpd_si256(a);
//       ALIGN32  __m256i const e = _mm256_and_epi64( _mm256_sub_epi64(aInt, log_i1),  log_i2); //    e = (__double_as_int (a) - i1 )    &   i2   ;
//       ALIGN32  __m256d const i = _mm256_fmadd_pd ( avx2_cvtepi64_pd(e), log_c1, _mm256_setzero_pd()); // 0x1.0p-52
//       ALIGN32  __m256d const m = _mm256_sub_pd(_mm256_castsi256_pd( _mm256_sub_epi64(aInt, e)),  _mm256_set1_pd(1.0)) ;  //   m = __int_as_double (__double_as_int (a) - e);   //  m = _mm256_sub_pd(m, one) ; // m - 1.0;  /* m in [2/3, 4/3] */
//       ALIGN32  __m256d const s = _mm256_mul_pd(m, m);  // m = _mm256_sub_pd(m,  _mm256_set1_pd(1.0)) ; // m - 1.0;  /* m in [2/3, 4/3] */
//       
//       /* Compute log1p(m) for m in [-1/3, 1/3] */
//       ALIGN32 __m256d   r, t;
//       r =             log_c2;  // -0x1.0ae000p-3
//       t =             log_c3;  //  0x1.208000p-3
//       r = _mm256_fmadd_pd (r, s, log_c4); // -0x1.f198b2p-4
//       t = _mm256_fmadd_pd (t, s, log_c5); //  0x1.1e5740p-3
//       r = _mm256_fmadd_pd (r, s, log_c6); // -0x1.55b36cp-3
//       t = _mm256_fmadd_pd (t, s, log_c7); //  0x1.99d8b2p-3
//       r = _mm256_fmadd_pd (r, s, log_c8); // -0x1.fffe02p-3
//       r = _mm256_fmadd_pd (t, m, r);
//       r = _mm256_fmadd_pd (r, m, log_c9); //  0x1.5554fap-2
//       r = _mm256_fmadd_pd (r, m, log_c10); // -0x1.000000p-1
//       r = _mm256_fmadd_pd (r, s, m);
//       
//       ALIGN32  __m256d const res = _mm256_fmadd_pd(i,  log_c11, r); //  0x1.62e430p-1 // log(2)
//       return res;
//   
// }
//// https://stackoverflow.com/a/65537754/9007125 // vectorized version of the answer by njuffa
ALWAYS_INLINE __m256d fast_log_1_wo_checks_AVX2 VECTORCALL(const __m256d a) {
    
      ALIGN32 __m256i const log_i1  = _mm256_set1_epi64x(0x3fe5555555555555);
      ALIGN32 __m256i const log_i2  = _mm256_set1_epi64x(0xFFF0000000000000);
    
      ALIGN32 __m256d const log_c1  = _mm256_set1_pd(0.000000000000000222044604925031308085); // 2^-52
      ALIGN32 __m256d const log_c2  = _mm256_set1_pd(-0.13031005859375);                      // -0x1.0ae000p-3
      ALIGN32 __m256d const log_c3  = _mm256_set1_pd(0.140869140625);                         //  0x1.208000p-3
      ALIGN32 __m256d const log_c4  = _mm256_set1_pd(-0.121483512222766876221);               // -0x1.f198b2p-4  <-- was -0.121483256...
      ALIGN32 __m256d const log_c5  = _mm256_set1_pd(0.139814853668212890625);                //  0x1.1e5740p-3
      ALIGN32 __m256d const log_c6  = _mm256_set1_pd(-0.166846126317977905273);               // -0x1.55b36cp-3
      ALIGN32 __m256d const log_c7  = _mm256_set1_pd(0.200120344758033752441);                //  0x1.99d8b2p-3
      ALIGN32 __m256d const log_c8  = _mm256_set1_pd(-0.249996200203895568848);               // -0x1.fffe02p-3
      ALIGN32 __m256d const log_c9  = _mm256_set1_pd(0.333331972360610961914);                //  0x1.5554fap-2
      ALIGN32 __m256d const log_c10 = _mm256_set1_pd(-0.5);
      ALIGN32 __m256d const log_c11 = _mm256_set1_pd(0.693147182464599609375);                //  0x1.62e430p-1
    
      ALIGN32 __m256i const aInt = _mm256_castpd_si256(a);
      ALIGN32 __m256i const e    = _mm256_and_si256(_mm256_sub_epi64(aInt, log_i1), log_i2);   // <-- was the _mm256_and_epi64 macro
      ALIGN32 __m256d const i    = _mm256_fmadd_pd(avx2_cvtepi64_pd(e), log_c1, _mm256_setzero_pd());
      ALIGN32 __m256d const m    = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_sub_epi64(aInt, e)), _mm256_set1_pd(1.0)); // m in [-1/3, 1/3]
      ALIGN32 __m256d const s    = _mm256_mul_pd(m, m);
    
      ALIGN32 __m256d r, t;
      r = log_c2;
      t = log_c3;
      r = _mm256_fmadd_pd(r, s, log_c4);
      t = _mm256_fmadd_pd(t, s, log_c5);
      r = _mm256_fmadd_pd(r, s, log_c6);
      t = _mm256_fmadd_pd(t, s, log_c7);
      r = _mm256_fmadd_pd(r, s, log_c8);
      r = _mm256_fmadd_pd(t, m, r);
      r = _mm256_fmadd_pd(r, m, log_c9);
      r = _mm256_fmadd_pd(r, m, log_c10);
      r = _mm256_fmadd_pd(r, s, m);
    
      return _mm256_fmadd_pd(i, log_c11, r);
  
}



ALWAYS_INLINE __m256d fast_log1p_1_wo_checks_AVX2 VECTORCALL(const __m256d x) {
    
      ALIGN32  __m256d const small =  _mm256_set1_pd(1e-4);
      ALIGN32  __m256d const minus_one_half  =  _mm256_set1_pd(-0.50);
    
      ALIGN32  __m256d const mask = _mm256_cmp_pd(_mm256_abs_pd(x), small, _CMP_GT_OQ);     // Create mask using _mm256_cmp_pd
      
      ALIGN32  __m256d const res =  _mm256_blendv_pd(
                                    _mm256_mul_pd(_mm256_fmadd_pd(minus_one_half, x,  _mm256_set1_pd(1.0)), x),
                                    fast_log_1_wo_checks_AVX2(_mm256_add_pd(x,  _mm256_set1_pd(1.0))),
                                    mask
                                  );
      
      return res;
  
}




ALWAYS_INLINE __m256d fast_log1m_1_wo_checks_AVX2 VECTORCALL(const __m256d x) {
  
     ALIGN32  __m256d const res =  fast_log1p_1_wo_checks_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x));
     return res;
  
} 




ALWAYS_INLINE __m256d fast_log1p_exp_1_wo_checks_AVX2 VECTORCALL(const __m256d x) {
  
       ALIGN32  __m256d const neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
  
       ALIGN32  __m256d const mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ);     // Create mask using _mm256_cmp_pd
      
       ALIGN32  __m256d const res =  _mm256_blendv_pd(
                                    fast_log1p_1_wo_checks_AVX2(fast_exp_1_wo_checks_AVX2(x)),
                                    _mm256_add_pd(x, fast_log1p_1_wo_checks_AVX2(fast_exp_1_wo_checks_AVX2(neg_x))),
                                    mask
                                  );
      
      return res;
  
}




// //https://stackoverflow.com/a/65537754/9007125
// Isolate rare subnormal/domain repairs from the normal-positive SIMD path.
__attribute__((noinline, cold)) inline __m256d fast_log_exceptional_lanes_AVX2(const __m256d input, const __m256d result, const unsigned ordinary_mask) {
    alignas(32) double input_values[4], output_values[4];
    _mm256_store_pd(input_values, input);
    _mm256_store_pd(output_values, result);
    for (int lane = 0; lane < 4; ++lane) {
        if (!(ordinary_mask & (1u << lane))) output_values[lane] = std::log(input_values[lane]);
    }
    return _mm256_load_pd(output_values);
}

ALWAYS_INLINE __m256d fast_log_1_AVX2 VECTORCALL(const __m256d a) {

      // The mantissa/exponent shortcut assumes a positive NORMAL double.
      const __m256d ordinary_lanes = _mm256_and_pd(
          _mm256_cmp_pd(a, _mm256_set1_pd(std::numeric_limits<double>::min()), _CMP_GE_OQ),
          _mm256_cmp_pd(a, _mm256_set1_pd(INFINITY), _CMP_LT_OQ));
      const int ordinary_mask = _mm256_movemask_pd(ordinary_lanes);
      const __m256d result = fast_log_1_wo_checks_AVX2(a);
      if (__builtin_expect(ordinary_mask != 0xf, 0)) return fast_log_exceptional_lanes_AVX2(a, result, ordinary_mask);
      return result;
  
}




// //  compute log(1+x) without losing precision for small values of x.
// //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
ALWAYS_INLINE __m256d fast_log1p_1_AVX2 VECTORCALL(const __m256d x) {
  
      ALIGN32  __m256d const minus_1 =  _mm256_set1_pd(-1.0);
      ALIGN32  __m256d const neg_half = _mm256_set1_pd(-0.5);
      ALIGN32  __m256d const sign_bit = _mm256_set1_pd(-0.0);
      
      ALIGN32  __m256d const res =  _mm256_blendv_pd(
                                    _mm256_blendv_pd(
                                      _mm256_mul_pd(_mm256_fmadd_pd(neg_half, x,  _mm256_set1_pd(1.0)), x),
                                      fast_log_1_AVX2(_mm256_add_pd(x,  _mm256_set1_pd(1.0))),
                                      _mm256_cmp_pd(_mm256_andnot_pd(sign_bit, x), _mm256_set1_pd(1e-4), _CMP_GT_OQ)
                                    ), 
                                    fast_log_1_AVX2(_mm256_add_pd(x,  _mm256_set1_pd(1.0))),
                                    _mm256_cmp_pd(x, minus_1, _CMP_LE_OQ)
                                  );
      
      return res;
  
} 




// //  compute log(1-x) without losing precision for small values of x.
ALWAYS_INLINE __m256d fast_log1m_1_AVX2 VECTORCALL(const __m256d x) {
  
      ALIGN32  __m256d const res =  fast_log1p_1_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x));
      return res;
  
}  




// //  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
ALWAYS_INLINE __m256d fast_log1p_exp_1_AVX2 VECTORCALL(const __m256d x) {
  
      ALIGN32  __m256d const neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
      
      ALIGN32  __m256d const res =  _mm256_blendv_pd(
                                      fast_log1p_1_AVX2(fast_exp_1_AVX2(x)),
                                      _mm256_add_pd(x, fast_log1p_1_AVX2(fast_exp_1_AVX2(neg_x))), 
                                      _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ)
                                    );
      
      return res;
   
}




ALWAYS_INLINE     __m256d fast_logit_wo_checks_AVX2 VECTORCALL(const __m256d x)   {
  
      ALIGN32  __m256d const log_x = fast_log_1_AVX2(x);
      ALIGN32  __m256d const log_1m_x =  fast_log1m_1_AVX2(x);
      ALIGN32  __m256d const res =   log_x - log_1m_x;
      return res;
  
} 




ALWAYS_INLINE     __m256d fast_logit_AVX2 VECTORCALL(const __m256d x)   {
  
      ALIGN32  __m256d const log_x = fast_log_1_AVX2(x);
      ALIGN32  __m256d const log_1m_x =  fast_log1m_1_AVX2(x);
      ALIGN32  __m256d const res =   log_x - log_1m_x;
      return res;
  
} 




// //https://stackoverflow.com/a/65537754/9007125
// inline      __m256d fast_log_1_AVX2(const __m256d a ) {
//   
//             const __mmask8 is_a_finite_and_gr0_mask =  is_finite_mask(a) &  _mm256_cmp_pd_mask(a, _mm256_setzero_pd(), _CMP_GT_OQ);
//             
//             return  _mm256_mask_blend_pd(is_a_finite_and_gr0_mask,
//                                          _mm256_mask_blend_pd(_mm256_cmp_pd_mask(_mm256_setzero_pd(), a, _CMP_EQ_OQ),     // cond is "if a = 0 "
//                                                               _mm256_sub_pd(_mm256_set1_pd(INFINITY), _mm256_set1_pd(INFINITY)),  // if a =/= 0  (which implies "if a < 0.0" !!!!!)
//                                                               _mm256_set1_pd(-INFINITY)),  // if a = 0 (i.e. cond=TRUE)
//                                                               fast_log_1_wo_checks_AVX2(a));  // if TRUE
//             
// }
// 
// //  compute log(1+x) without losing precision for small values of x.
// //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
// inline     __m256d fast_log1p_1_AVX2(const __m256d x)   {
//   
//   const __mmask8 is_x_le_or_eq_to_m1_mask =     _mm256_cmp_pd_mask(x,  _mm256_set1_pd(-1.0), _CMP_LE_OQ);
//   const __m256d sign_bit = _mm256_set1_pd(-0.0);
//   const __m256d x_abs = _mm256_andnot_pd(sign_bit, x);
//   
//   return   _mm256_mask_blend_pd(is_x_le_or_eq_to_m1_mask, // is x <= -1? (i.e., is x+1 <= 0?)
//                                 _mm256_mask_blend_pd( _mm256_cmp_pd_mask(x_abs,  _mm256_set1_pd(1e-4), _CMP_GT_OQ), // if "is_x_le_or_eq_to_m1_mask" is FALSE (i.e., x+1 > 0)
//                                                      _mm256_mul_pd(_mm256_fmadd_pd( _mm256_set1_pd(-0.5), x,  _mm256_set1_pd(1.0)), x) ,
//                                                      fast_log_1_AVX2( _mm256_add_pd(x,  _mm256_set1_pd(1.0)) ) ), // if x > 1e-4 - evaluate using normal log fn!
//                                  fast_log_1_AVX2( _mm256_add_pd(x,  _mm256_set1_pd(1.0)) ) ) ;  // if FALSE (i.e., x+1  > 0 ) then still pass onto normal log fn as this will handle NaNs etc!
//   
// }
// 
// //  compute log(1-x) without losing precision for small values of x.
// inline     __m256d fast_log1m_1_AVX2(const __m256d x)   {
//   
//   return fast_log1p_1_AVX2( _mm256_sub_pd(_mm256_setzero_pd() , x)) ; // -x);
//   
// }
// 
// //  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
// inline      __m256d fast_log1p_exp_1_AVX2(const __m256d x)   {
//   
//   const __m256d neg_x =  _mm256_sub_pd(_mm256_setzero_pd(), x);
//   const __mmask8 is_x_gr_0_mask =      _mm256_cmp_pd_mask(x,  _mm256_setzero_pd(), _CMP_GT_OQ);
//   
//   return       _mm256_mask_blend_pd(is_x_gr_0_mask,
//                                     fast_log1p_1_AVX2(fast_exp_1_AVX2(x)),
//                                     _mm256_add_pd(x, fast_log1p_1_AVX2(fast_exp_1_AVX2(neg_x))));
//   
// }




///////////////////  fns - inv_logit   ----------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE     __m256d  fast_inv_logit_for_x_pos_AVX2 VECTORCALL(const __m256d x )  {
  
          ALIGN32  __m256d const  exp_m_x =  fast_exp_1_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x)) ;
          ALIGN32  __m256d const res =    _mm256_div_pd( _mm256_set1_pd(1.0), _mm256_add_pd( _mm256_set1_pd(1.0), exp_m_x))  ;
          return res;
}




ALWAYS_INLINE     __m256d  fast_inv_logit_for_x_neg_AVX2 VECTORCALL(const __m256d x )  {
  
          ALIGN32  __m256d const  log_eps  =   _mm256_set1_pd(-18.420680743952367);
          ALIGN32  __m256d const  exp_x =  fast_exp_1_AVX2(x) ;
          
          ALIGN32  __m256d const res = _mm256_blendv_pd(    exp_x,
                                      _mm256_div_pd(exp_x, _mm256_add_pd( _mm256_set1_pd(1.0), exp_x)), 
                                      _mm256_cmp_pd(x,  log_eps, _CMP_GT_OQ));
          return res;
}




// ALWAYS_INLINE     __m256d  fast_inv_logit_AVX2 VECTORCALL(const __m256d x )  {
//   
//           ALIGN32  __m256d const res =  _mm256_blendv_pd(  fast_inv_logit_for_x_neg_AVX2(x),
//                                        fast_inv_logit_for_x_pos_AVX2(x), 
//                                        _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ));  // x > 0
//           return res;
//   
// }
ALWAYS_INLINE __m256d fast_inv_logit_AVX2 VECTORCALL(const __m256d x) {
    
          ALIGN32 __m256d const result = _mm256_blendv_pd(fast_inv_logit_for_x_neg_AVX2(x),
                                                           fast_inv_logit_for_x_pos_AVX2(x),
                                                           _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ));
          return _mm256_min_pd(_mm256_set1_pd(1.0), _mm256_max_pd(_mm256_setzero_pd(), result));
      
}




ALWAYS_INLINE      __m256d  fast_inv_logit_for_x_pos_wo_checks_AVX2 VECTORCALL(const __m256d x )  {

          ALIGN32  __m256d const  exp_m_x =  fast_exp_1_wo_checks_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x)) ;
          ALIGN32  __m256d const res =    _mm256_div_pd( _mm256_set1_pd(1.0), _mm256_add_pd( _mm256_set1_pd(1.0), exp_m_x))  ;
          return res;
  
}




ALWAYS_INLINE     __m256d  fast_inv_logit_for_x_neg_wo_checks_AVX2 VECTORCALL(const __m256d x )  {
  
          ALIGN32  __m256d const  log_eps  =   _mm256_set1_pd(-18.420680743952367);
          ALIGN32  __m256d const  exp_x =  fast_exp_1_wo_checks_AVX2(x) ;
          
          ALIGN32  __m256d const res = _mm256_blendv_pd(    exp_x,
                                      _mm256_div_pd(exp_x, _mm256_add_pd( _mm256_set1_pd(1.0), exp_x)), 
                                      _mm256_cmp_pd(x,  log_eps, _CMP_GT_OQ));
          return res;
  
}




// ALWAYS_INLINE      __m256d  fast_inv_logit_wo_checks_AVX2 VECTORCALL(const __m256d x )  {
//   
//           ALIGN32  __m256d const res =  _mm256_blendv_pd(    fast_inv_logit_for_x_neg_wo_checks_AVX2(x),
//                                        fast_inv_logit_for_x_pos_wo_checks_AVX2(x), 
//                                        _mm256_cmp_pd(x,  _mm256_setzero_pd(), _CMP_GT_OQ));  // x > 0
//           return res;
//       
// }
ALWAYS_INLINE __m256d fast_inv_logit_wo_checks_AVX2 VECTORCALL(const __m256d x) {
      
          ALIGN32 __m256d const result = _mm256_blendv_pd(fast_inv_logit_for_x_neg_wo_checks_AVX2(x),
                                                           fast_inv_logit_for_x_pos_wo_checks_AVX2(x),
                                                           _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ));
          return _mm256_min_pd(_mm256_set1_pd(1.0), _mm256_max_pd(_mm256_setzero_pd(), result));
      
}




///////////////////  fns - log_inv_logit   ----------------------------------------------------------------------------------------------------------------
// inline     __m256d  fast_log_inv_logit_for_x_pos_AVX2 VECTORCALL(const __m256d x )  {
//    
//       ALIGN32 const    __m256d   m_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
//     
//       ALIGN32  __m256d const res =   _mm256_sub_pd(_mm256_setzero_pd(), fast_log1p_exp_1_AVX2(fast_exp_1_AVX2(m_x)));
//       return res;
//     
// }
inline __m256d fast_log_inv_logit_for_x_pos_AVX2 VECTORCALL(const __m256d x) {
  
      ALIGN32 __m256d const m_x     = _mm256_sub_pd(_mm256_setzero_pd(), x);   // -x
      ALIGN32 __m256d const exp_m_x = fast_exp_1_AVX2(m_x);                    // exp(-x)
      ALIGN32 __m256d const log_sum = fast_log1p_1_AVX2(exp_m_x);              // log(1 + exp(-x))
      return _mm256_sub_pd(_mm256_setzero_pd(), log_sum);                      // -log(1 + exp(-x))
      
}




inline     __m256d  fast_log_inv_logit_for_x_neg_AVX2 VECTORCALL(const __m256d x )  {

      ALIGN32  __m256d const  log_eps  =   _mm256_set1_pd(-18.420680743952367);
    
      ALIGN32  __m256d const res = _mm256_blendv_pd(    x,
                                  _mm256_sub_pd(x,   fast_log1p_1_AVX2(fast_exp_1_AVX2(x))),
                                  _mm256_cmp_pd(x,  log_eps, _CMP_GT_OQ));
      return res;
    
}




inline     __m256d  fast_log_inv_logit_AVX2 VECTORCALL(const __m256d x )  {
  
      ALIGN32  __m256d const res =  _mm256_blendv_pd(    fast_log_inv_logit_for_x_neg_AVX2(x),
                                     fast_log_inv_logit_for_x_pos_AVX2(x), 
                                     _mm256_cmp_pd(x,  _mm256_setzero_pd(), _CMP_GE_OQ));  // x > 0
      return res;
      
}




////////////
inline     __m256d  fast_log_inv_logit_for_x_pos_wo_checks_AVX2 VECTORCALL(const __m256d x )  {
      
      ALIGN32  __m256d const res =   _mm256_sub_pd(_mm256_setzero_pd(), fast_log1p_exp_1_wo_checks_AVX2((_mm256_sub_pd(_mm256_setzero_pd(), x)) ) );
      return res;
   
}




inline     __m256d  fast_log_inv_logit_for_x_neg_wo_checks_AVX2 VECTORCALL(const __m256d x )  {
  
      ALIGN32  __m256d const  log_eps  =   _mm256_set1_pd(-18.420680743952367);
    
      ALIGN32  __m256d const res = _mm256_blendv_pd( x,
                                  _mm256_sub_pd(x,   fast_log1p_1_wo_checks_AVX2(fast_exp_1_wo_checks_AVX2(x))),
                                  _mm256_cmp_pd(x,  log_eps, _CMP_GT_OQ));
      return res;
    
}




inline     __m256d  fast_log_inv_logit_wo_checks_AVX2 VECTORCALL(const __m256d x )  {
  
      ALIGN32  __m256d const res =  _mm256_blendv_pd(    fast_log_inv_logit_for_x_neg_wo_checks_AVX2(x),
                                     fast_log_inv_logit_for_x_pos_wo_checks_AVX2(x),
                                     _mm256_cmp_pd(x,  _mm256_setzero_pd(), _CMP_GE_OQ));  // x > 0
      return res;
  
}




/////////////////////  fns - Phi_approx   ----------------------------------------------------------------------------------------------------------------
inline     __m256d  fast_Phi_approx_wo_checks_AVX2 VECTORCALL(const __m256d x)  {
  
      ALIGN32  __m256d const a =     _mm256_set1_pd(0.07056);
      ALIGN32  __m256d const b =     _mm256_set1_pd(1.5976);
      
      ALIGN32  __m256d const   x_sq =  _mm256_mul_pd(x, x);
      ALIGN32  __m256d const   a_x_sq_plus_b = _mm256_fmadd_pd(a, x_sq, b);
      ALIGN32  __m256d const   stuff_to_inv_logit =  _mm256_mul_pd(x, a_x_sq_plus_b);
      
      ALIGN32  __m256d const res = fast_inv_logit_wo_checks_AVX2(stuff_to_inv_logit);
      return res;
  
}




inline     __m256d  fast_Phi_approx_AVX2 VECTORCALL(const __m256d x )  {
  
      ALIGN32  __m256d const a =     _mm256_set1_pd(0.07056);
      ALIGN32  __m256d const b =     _mm256_set1_pd(1.5976);
      
      ALIGN32  __m256d const  x_sq =  _mm256_mul_pd(x, x);
      ALIGN32  __m256d const  a_x_sq_plus_b = _mm256_fmadd_pd(a, x_sq, b);
      ALIGN32  __m256d const   stuff_to_inv_logit =  _mm256_mul_pd(x, a_x_sq_plus_b);
      
      ALIGN32  __m256d const res = fast_inv_logit_AVX2(stuff_to_inv_logit);
      return res;
  
}




///////////////////  fns - inv_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
inline      __m256d  fast_inv_Phi_approx_wo_checks_AVX2 VECTORCALL(const __m256d x )  {
  
      ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(-0.34176618822627863);
      ALIGN32  __m256d const  inv_Phi_approx_c2 = _mm256_set1_pd(2.7472242570765295);
      ALIGN32  __m256d const  one_third =  _mm256_set1_pd(0.33333333333333331483);
      
      ALIGN32  __m256d const  m_logit_p = fast_log_1_wo_checks_AVX2( _mm256_sub_pd(_mm256_div_pd( _mm256_set1_pd(1.0),  x),  _mm256_set1_pd(1.0) ) ); // logit first
      ALIGN32  __m256d const  x_i =  _mm256_mul_pd(m_logit_p, inv_Phi_approx_c1) ; // -0.34176618822627863*m_logit_p;
      ALIGN32  __m256d const  asinh_stuff_div_3 =  _mm256_mul_pd( one_third,  fast_log_1_wo_checks_AVX2(  _mm256_add_pd(x_i, (  _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ) ;          // now do arc_sinh part
      ALIGN32  __m256d const  exp_x_i = fast_exp_1_wo_checks_AVX2(asinh_stuff_div_3);
      
      ALIGN32  __m256d const res = _mm256_mul_pd(  inv_Phi_approx_c2,   _mm256_div_pd(_mm256_sub_pd(_mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;    //   now do sinh parth part
      return res;
  
}




inline     __m256d  fast_inv_Phi_approx_AVX2 VECTORCALL(const __m256d x )  {

      // Avoid 1 / p overflowing for valid subnormal probabilities. The usual path below is unchanged.
      const __m256d ordinary_lanes = _mm256_and_pd(
          _mm256_cmp_pd(x, _mm256_set1_pd(std::numeric_limits<double>::min()), _CMP_GE_OQ),
          _mm256_cmp_pd(x, _mm256_set1_pd(1.0), _CMP_LT_OQ));
      const int ordinary_mask = _mm256_movemask_pd(ordinary_lanes);
      if (ordinary_mask != 0xf) {
          alignas(32) double input_values[4], output_values[4];
          _mm256_store_pd(input_values, x);
          const __m256d safe_input = _mm256_blendv_pd(_mm256_set1_pd(0.5), x, ordinary_lanes);
          _mm256_store_pd(output_values, fast_inv_Phi_approx_AVX2(safe_input));
          for (int lane = 0; lane < 4; ++lane) {
              if (!(ordinary_mask & (1u << lane))) {
                  const double logit_probability = std::log(input_values[lane]) - std::log1p(-input_values[lane]);
                  output_values[lane] = 5.494448514153059 * std::sinh(std::asinh(0.34176618822627863 * logit_probability) / 3.0);
              }
          }
          return _mm256_load_pd(output_values);
      }
  
      ALIGN32  __m256d const  one_third =  _mm256_set1_pd(0.33333333333333331483);
      
      ALIGN32  __m256d const m_logit_p = fast_log_1_AVX2(_mm256_div_pd(_mm256_sub_pd(_mm256_set1_pd(1.0), x), x)); // log((1-p)/p), stable near 1
      ALIGN32  __m256d const x_i =  _mm256_mul_pd(m_logit_p, _mm256_set1_pd(-0.34176618822627863)) ; // -0.34176618822627863*m_logit_p;
      ALIGN32  __m256d const asinh_stuff_div_3 =  _mm256_mul_pd(  one_third,  fast_log_1_AVX2(  _mm256_add_pd(x_i, (  _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ) ;          // now do arc_sinh part
      ALIGN32  __m256d const exp_x_i = fast_exp_1_AVX2(asinh_stuff_div_3);
      
      ALIGN32  __m256d const res = _mm256_mul_pd(  _mm256_set1_pd(2.7472242570765295),   _mm256_div_pd( _mm256_sub_pd(_mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;    //   now do sinh parth part
      return res;
      
}




// need to add citation to this (slight modification from a forum post)
inline      __m256d  fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX2 VECTORCALL(const __m256d logit_p )  {
  
      // ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(-0.34176618822627863);
      ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(0.34176618822627863);
      ALIGN32  __m256d const  inv_Phi_approx_c2 = _mm256_set1_pd(2.7472242570765295);
      ALIGN32  __m256d const  one_third =  _mm256_set1_pd(0.33333333333333331483);
      
      ALIGN32  __m256d const  x_i =  _mm256_mul_pd(logit_p,  inv_Phi_approx_c1) ;
      ALIGN32  __m256d const  asinh_stuff_div_3 =  _mm256_mul_pd(  one_third,  fast_log_1_wo_checks_AVX2( _mm256_add_pd(x_i  , _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ;          // now do arc_sinh part
      ALIGN32  __m256d const  exp_x_i = fast_exp_1_wo_checks_AVX2(asinh_stuff_div_3);
      
      ALIGN32  __m256d const res = _mm256_mul_pd(inv_Phi_approx_c2,   _mm256_div_pd(_mm256_sub_pd( _mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;   //   now do sinh parth part
      return res;
  
  
}




inline     __m256d  fast_inv_Phi_approx_from_logit_prob_AVX2 VECTORCALL(const __m256d logit_p )  {
  
      // ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(-0.34176618822627863);
      ALIGN32  __m256d const  inv_Phi_approx_c1 = _mm256_set1_pd(0.34176618822627863);
      ALIGN32  __m256d const  inv_Phi_approx_c2 = _mm256_set1_pd(2.7472242570765295);
      ALIGN32  __m256d const  one_third =  _mm256_set1_pd(0.33333333333333331483);
      
      ALIGN32  __m256d const  x_i =  _mm256_mul_pd(logit_p,  inv_Phi_approx_c1) ;
      ALIGN32  __m256d const  asinh_stuff_div_3 =  _mm256_mul_pd( one_third,  fast_log_1_AVX2( _mm256_add_pd(x_i  , _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ;          // now do arc_sinh part
      ALIGN32  __m256d const  exp_x_i = fast_exp_1_AVX2(asinh_stuff_div_3);
      
      ALIGN32  __m256d const res = _mm256_mul_pd(inv_Phi_approx_c2,   _mm256_div_pd(_mm256_sub_pd( _mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;   //   now do sinh parth part
      return res;
  
}




///////////////////  fns - log_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------
inline      __m256d  fast_log_Phi_approx_wo_checks_AVX2 VECTORCALL(const __m256d x )  {
  
      ALIGN32  __m256d const a =     _mm256_set1_pd(0.07056);
      ALIGN32  __m256d const b =     _mm256_set1_pd(1.5976);
      
      ALIGN32  __m256d const  x_sq =  _mm256_mul_pd(x, x);
      ALIGN32  __m256d const  a_x_sq_plus_b = _mm256_fmadd_pd(a, x_sq, b);
      ALIGN32  __m256d const  stuff_to_inv_logit =  _mm256_mul_pd(x, a_x_sq_plus_b);
      
      ALIGN32  __m256d const res = fast_log_inv_logit_wo_checks_AVX2(stuff_to_inv_logit);
      return res;
  
}




inline __m256d fast_log_Phi_approx_AVX2 VECTORCALL(const __m256d x) {
  
      ALIGN32  __m256d const a =     _mm256_set1_pd(0.07056);
      ALIGN32  __m256d const b =     _mm256_set1_pd(1.5976);
      ALIGN32  __m256d const Phi_upper_bound =  _mm256_set1_pd(8.25);
      ALIGN32  __m256d const Phi_lower_bound =  _mm256_set1_pd(-37.5);
      
      ALIGN32  __m256d const x_sq = _mm256_mul_pd(x, x);
      ALIGN32  __m256d const x_cubed = _mm256_mul_pd(x_sq, x);
      ALIGN32  __m256d const result = _mm256_fmadd_pd(a, x_cubed, _mm256_mul_pd(b, x));
      
      ALIGN32  __m256d const res = fast_log_inv_logit_AVX2(result);
      return res;
  
} 




//////////////////// ------------- tanh  --------------------------------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE      __m256d    fast_tanh_AVX2 VECTORCALL( const  __m256d x  )   {
  
  
      ALIGN32  __m256d const  two =     _mm256_set1_pd(2.0);
      ALIGN32  __m256d const  m_two =   _mm256_set1_pd(-2.0);
      
      ALIGN32  __m256d const res = _mm256_sub_pd( _mm256_div_pd(two,   _mm256_add_pd( _mm256_set1_pd(1.0), fast_exp_1_AVX2(  _mm256_mul_pd(x, m_two)   ) ) ),  _mm256_set1_pd(1.0) ) ;
      return res;
  
}




ALWAYS_INLINE      __m256d    fast_tanh_wo_checks_AVX2 VECTORCALL(const   __m256d x  )   {
  
      ALIGN32  __m256d const  two =     _mm256_set1_pd(2.0);
      ALIGN32  __m256d const  m_two =   _mm256_set1_pd(-2.0);
      
      ALIGN32  __m256d const res  =  _mm256_sub_pd( _mm256_div_pd(two,   _mm256_add_pd( _mm256_set1_pd(1.0), fast_exp_1_wo_checks_AVX2(  _mm256_mul_pd(x, m_two)   ) ) ),  _mm256_set1_pd(1.0) ) ;
      return res;
      
}




// from: https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
ALWAYS_INLINE    __m256d CopySign VECTORCALL(  const __m256d srcSign, 
                                               const __m256d srcValue) {
        
      ALIGN32  __m256d const mask0 = _mm256_set1_pd(-0.);
      
      ALIGN32  __m256d const tmp0 = _mm256_and_pd(srcSign, mask0); // Extract the signed bit from srcSign
      ALIGN32  __m256d const tmp1 = _mm256_andnot_pd(mask0, srcValue); // Extract the number without sign of srcValue (abs(srcValue))
      
      ALIGN32  __m256d const res =  _mm256_or_pd(tmp0, tmp1);  // Merge signed bit with number and return
      return res;
  
}




///////////////////  fns - Phi functions   ------------------------------------------------------------------------------------------------------------------------------------
// inline    __m256d fast_Phi_wo_checks_AVX2(const __m256d x) {
//   
//     const __m256d sqrt_2_recip = _mm256_set1_pd(0.707106781186547461715);
//   
//     return _mm256_mul_pd(one_half, _mm256_add_pd(one, fast_erf_wo_checks_AVX2( _mm256_mul_pd(x, sqrt_2_recip)) ) ) ;
//     
// }
//  
//  
//  
// 
// // AVX2 version with consistent mask logic (i.e., consistent with AVX-256 logic)
// inline __m256d fast_Phi_AVX2(const __m256d x) {
//  
//   
//   const __m256d is_x_gr_lower =  _mm256_cmp_pd(x, Phi_lower_bound, _CMP_GT_OQ); // i.e. if x > -37.5  
//   const __m256d is_x_lt_upper =  _mm256_cmp_pd(x, Phi_upper_bound, _CMP_LT_OQ); // i.e. if x < 8.25
//   const __m256d is_x_in_range =  _mm256_and_pd( is_x_gr_lower ,  is_x_lt_upper);
//   
//    
//    return    _mm256_blendv_pd(     
//                              _mm256_blendv_pd(zero,   // if x NOT in range and x NOT < -37.5 i.e. x must be <-37.5
//                                               one,  // if x NOT in range and x > 8.25
//                                               _mm256_cmp_pd(x, Phi_lower_bound, _CMP_LT_OQ)),  // if x NOT in range and x < -37.5
//                             fast_Phi_wo_checks_AVX2(x), // i.e. if x in range ?
//                             is_x_in_range); 
//   
// }
// 
// 
// 
//




//// based on Abramowitz-Stegun polynomial approximation for Phi
ALWAYS_INLINE __m256d fast_Phi_wo_checks_AVX2 VECTORCALL(const __m256d x) {
   
       ALIGN32  __m256d const a =  _mm256_set1_pd(0.2316419);
       ALIGN32  __m256d const b1 = _mm256_set1_pd(0.31938153);
       ALIGN32  __m256d const b2 = _mm256_set1_pd(-0.356563782);
       ALIGN32  __m256d const b3 = _mm256_set1_pd(1.781477937);
       ALIGN32  __m256d const b4 = _mm256_set1_pd(-1.821255978);
       ALIGN32  __m256d const b5 = _mm256_set1_pd(1.330274429);
       ALIGN32  __m256d const rsqrt_2pi = _mm256_set1_pd(0.3989422804014327);
       
       ALIGN32  __m256d const z = _mm256_abs_pd(x); /////  std::fabs(x);
       
       ALIGN32  __m256d const denom_t = _mm256_fmadd_pd(a, z, _mm256_set1_pd(1.0));
       
       ALIGN32  __m256d const t = _mm256_div_pd(_mm256_set1_pd(1.0), denom_t);  //// double t = 1.0 / (1.0 + a * z);
       ALIGN32  __m256d const t_2 = _mm256_mul_pd(t, t);
       ALIGN32  __m256d const t_3 = _mm256_mul_pd(t_2, t);
       ALIGN32  __m256d const t_4 = _mm256_mul_pd(t_2, t_2);
       ALIGN32  __m256d const t_5 = _mm256_mul_pd(t_3, t_2);
       
       /////  double poly = b1 * t     +   b2 * t * t    +     b3 * t * t * t   +   b4 * t * t * t * t      +       b5 * t * t * t * t * t;
       ALIGN32  __m256d const poly_term_1 = _mm256_mul_pd(b1, t);
       ALIGN32  __m256d const poly_term_2 = _mm256_mul_pd(b2, t_2);
       ALIGN32  __m256d const poly_term_3 = _mm256_mul_pd(b3, t_3);
       ALIGN32  __m256d const poly_term_4 = _mm256_mul_pd(b4, t_4);
       ALIGN32  __m256d const poly_term_5 = _mm256_mul_pd(b5, t_5);
       
       ALIGN32  __m256d const poly = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(poly_term_1, poly_term_2),  _mm256_add_pd(poly_term_3, poly_term_4)), poly_term_5);
       
       ALIGN32  __m256d const is_x_gr_0 = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ);
       
       ALIGN32  __m256d const exp_stuff = fast_exp_1_AVX2(_mm256_mul_pd(_mm256_set1_pd(-0.50), _mm256_mul_pd(z, z)) ); 
       ALIGN32  __m256d const val = _mm256_mul_pd(_mm256_mul_pd(rsqrt_2pi, exp_stuff), poly);
       ALIGN32  __m256d const one_m_res = _mm256_sub_pd(_mm256_set1_pd(1.0), val);
       
       ALIGN32  __m256d const res = _mm256_blendv_pd(    val,  //// if x < 0
                                                        one_m_res, //// if x > 0
                                                        is_x_gr_0); 
       return res;
   
}




ALWAYS_INLINE __m256d fast_Phi_AVX2 VECTORCALL(const __m256d x) {
  
      
      ALIGN32  __m256d const Phi_upper_bound =  _mm256_set1_pd(8.25);
      ALIGN32  __m256d const Phi_lower_bound =  _mm256_set1_pd(-37.5);
      
      ALIGN32  __m256d const is_x_gr_lower = _mm256_cmp_pd(x, Phi_lower_bound, _CMP_GE_OQ); // equality must not fall through to 1
      ALIGN32  __m256d const is_x_lt_upper = _mm256_cmp_pd(x, Phi_upper_bound, _CMP_LT_OQ); // true where x < 8.25 (in range)
      ALIGN32  __m256d const is_x_in_range = _mm256_and_pd(is_x_gr_lower, is_x_lt_upper);
      
      // Calculate the main Phi function for in-range values
      ALIGN32  __m256d const val = _mm256_blendv_pd(
                                    _mm256_blendv_pd( _mm256_set1_pd(1.0),    // if x NOT in range and x NOT < -37.5 i.e. x must be > 8.25
                                                      _mm256_setzero_pd(), // if x NOT in range and x < -37.5
                                                      _mm256_cmp_pd(x, Phi_lower_bound, _CMP_LT_OQ)),  
                                                      fast_Phi_wo_checks_AVX2(x), // if x is in range
                                                      is_x_in_range
      );
      
      // Clamp results between 0 and 1
      ALIGN32  __m256d const res = _mm256_min_pd(_mm256_set1_pd(1.0), _mm256_max_pd(_mm256_setzero_pd(), val));
      // Preserve NaN input rather than turning it into a valid probability through min/max.
      return _mm256_blendv_pd(res, x, _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
  
}




///////////////////  fns - inv_Phi functions   --------------------------------------------------------------------------------------------------------------------------------
//// vectorised, AVX-256 version of Stan fn provided by Sean Pinkney:  https://github.com/stan-dev/math/issues/2555 
ALWAYS_INLINE __m256d fast_inv_Phi_wo_checks_case_2a_AVX2 VECTORCALL( const __m256d p,
                                                                      __m256d r) { ///  CASE 2(a): if abs(q) > 0.425  AND   if r <= 5.0 
   
       r = _mm256_add_pd(r, _mm256_set1_pd(-1.60));
       
       ALIGN32 __m256d numerator = _mm256_set1_pd(0.00077454501427834140764);
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(0.0227238449892691845833));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(0.24178072517745061177));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(1.27045825245236838258));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(3.64784832476320460504));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(5.7694972214606914055));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(4.6303378461565452959));
       numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(1.42343711074968357734)); 
       
       ALIGN32 __m256d  denominator = _mm256_set1_pd(0.00000000105075007164441684324);
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.0005475938084995344946));
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.0151986665636164571966));
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.14810397642748007459));
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.68976733498510000455));
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(1.6763848301838038494));
       denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(2.05319162663775882187));
       denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(1.0));
       
       ALIGN32  __m256d const val = _mm256_div_pd(numerator, denominator);
       return val;

}  




ALWAYS_INLINE __m256d fast_inv_Phi_wo_checks_case_2b_AVX2 VECTORCALL( const __m256d p,
                                                                      __m256d r) { ///  CASE 2(a): if abs(q) > 0.425  AND   if r > 5.0 
        
      ALIGN32  __m256d const one =   _mm256_set1_pd(1.0);
      
      r = _mm256_add_pd(r, _mm256_set1_pd(-5.0));
      
      ALIGN32 __m256d numerator =  _mm256_set1_pd(0.000000201033439929228813265);
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.0000271155556874348757815));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.0012426609473880784386));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.026532189526576123093));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.29656057182850489123));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(1.7848265399172913358));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(5.4637849111641143699));
      numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(6.6579046435011037772));
      
      ALIGN32 __m256d denominator =  _mm256_set1_pd(0.00000000000000204426310338993978564);
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.00000014215117583164458887));
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.000018463183175100546818));
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.0007868691311456132591));
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.0148753612908506148525));
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.13692988092273580531)); 
      denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.59983220655588793769));
      denominator = _mm256_fmadd_pd(r, denominator,  one);
      
      ALIGN32  __m256d const val = _mm256_div_pd(numerator, denominator);
      return val;

} 




// ALWAYS_INLINE __m256d fast_inv_Phi_wo_checks_case_2_AVX2 VECTORCALL( const __m256d p,
//                                                                      const __m256d q) {  /// CASE 2 (i.e. one of case 2(a) or 2(b) depending on value of r)
//   
//   
//       ALIGN32  __m256d const zero =  _mm256_setzero_pd();
//       ALIGN32  __m256d const one =   _mm256_set1_pd(1.0);
//       ALIGN32  __m256d const five =  _mm256_set1_pd(5.0);
//       
//       ALIGN32  __m256d const is_q_gr_0 = _mm256_cmp_pd(q, zero, _CMP_GT_OQ);
//        
//       ////   compute r 
//       ALIGN32 __m256d r_if_q_gr_0 =  _mm256_sub_pd(one, p);
//       ALIGN32 __m256d r_if_q_lt_0 = p;
//       ALIGN32 __m256d r = _mm256_blendv_pd(r_if_q_gr_0, r_if_q_lt_0, is_q_gr_0);
//       
//       r = _mm256_sqrt_pd(_mm256_sub_pd(zero, fast_log_1_AVX2(r)));
//       
//       /// then call either case 2(a) fn (if r <= 5.0) or case 2(b) fn (if r > 5.0)  
//       ALIGN32  __m256d const is_r_gr_5 = _mm256_cmp_pd(r, five, _CMP_GT_OQ);
//       ALIGN32 __m256d val =  _mm256_blendv_pd(     fast_inv_Phi_wo_checks_case_2b_AVX2(p, r),  /// if r > 5.0
//                                            fast_inv_Phi_wo_checks_case_2a_AVX2(p, r),
//                                            is_r_gr_5);
//       
//       // Flip the sign if q is negative
//       ALIGN32  __m256d const is_q_lt_zero = _mm256_cmp_pd(q, zero, _CMP_LT_OQ); // _mm256_cmplt_pd_mask(q, _mm256_set1_pd(0.0));
//       
//       val = _mm256_blendv_pd(    _mm256_sub_pd(zero, val),   //// if q < 0
//                                  val,
//                                  is_q_lt_zero); //// if   q => 0 
//       return val;
// 
// }
ALWAYS_INLINE __m256d fast_inv_Phi_wo_checks_case_2_AVX2 VECTORCALL( const __m256d p,
                                                                     const __m256d q) {
  
      ALIGN32 __m256d const zero = _mm256_setzero_pd();
      ALIGN32 __m256d const one  = _mm256_set1_pd(1.0);
      ALIGN32 __m256d const five = _mm256_set1_pd(5.0);

      ALIGN32 __m256d const q_pos = _mm256_cmp_pd(q, zero, _CMP_GT_OQ);
      ALIGN32 __m256d r = _mm256_blendv_pd(p, _mm256_sub_pd(one, p), q_pos);            // q>0 -> 1-p, else p
      r = _mm256_sqrt_pd(_mm256_sub_pd(zero, fast_log_1_AVX2(r)));

      ALIGN32 __m256d const r_gr_5 = _mm256_cmp_pd(r, five, _CMP_GT_OQ);
      ALIGN32 __m256d val = fast_inv_Phi_wo_checks_case_2a_AVX2(p, r);                 // r <= 5
      // No need to evaluate the extreme-tail polynomial unless a lane will actually use it.
      if (_mm256_movemask_pd(r_gr_5) != 0) val = _mm256_blendv_pd(val, fast_inv_Phi_wo_checks_case_2b_AVX2(p, r), r_gr_5);

      ALIGN32 __m256d const q_neg = _mm256_cmp_pd(q, zero, _CMP_LT_OQ);
      return _mm256_blendv_pd(val, _mm256_sub_pd(zero, val), q_neg);                    // q<0 -> -val
      
}




ALWAYS_INLINE __m256d fast_inv_Phi_wo_checks_case_1_AVX2 VECTORCALL( const __m256d p,
                                                                     const __m256d q) { ///  CASE 1: if abs(q) <= 0.425
   
      ALIGN32  __m256d const q_sq = _mm256_mul_pd(q, q);
      ALIGN32  __m256d const r = _mm256_sub_pd(_mm256_set1_pd(0.180625), q_sq);
      
      ALIGN32 __m256d numerator = _mm256_fmadd_pd(r, _mm256_set1_pd(2509.0809287301226727), _mm256_set1_pd(33430.575583588128105));
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(67265.770927008700853));
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(45921.953931549871457));
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(13731.693765509461125));
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(1971.5909503065514427)); 
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(133.14166789178437745));
      numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(3.387132872796366608));
      
      ALIGN32 __m256d denominator = _mm256_fmadd_pd(r, _mm256_set1_pd(5226.495278852854561), _mm256_set1_pd(28729.085735721942674));
      denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(39307.89580009271061));
      denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(21213.794301586595867));
      denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(5394.1960214247511077));
      denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(687.1870074920579083));
      denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(42.313330701600911252)); 
      denominator = _mm256_fmadd_pd(denominator, r,  _mm256_set1_pd(1.0));
      
      ALIGN32 __m256d val = _mm256_div_pd(numerator, denominator); 
      val = _mm256_mul_pd(q, val);
      return val;

}   




// ALWAYS_INLINE __m256d fast_inv_Phi_wo_checks_AVX2 VECTORCALL(const __m256d p) {
//   
//       ALIGN32  __m256d const one_half =  _mm256_set1_pd(0.50);
//       ALIGN32  __m256d const point_425 = _mm256_set1_pd(0.425);
//       
//       ALIGN32  __m256d const q = _mm256_sub_pd(p, one_half);
//       ALIGN32  __m256d const is_q_le_threshold = _mm256_cmp_pd(_mm256_abs_pd(q), point_425, _CMP_LE_OQ);
//       
//      
//       ALIGN32 __m256d res;
//       
//       if (_mm256_movemask_pd(is_q_le_threshold) == 0xF) {  // 0xF is binary 1111 (4 bits)  ///  if (_mm256_kortestc(is_q_le_threshold, is_q_le_threshold)) { 
//         
//         res = fast_inv_Phi_wo_checks_case_2_AVX2(p, q);
//         
//       } else { 
//         
//         res = fast_inv_Phi_wo_checks_case_1_AVX2(p, q);
//         
//       }
//       
//       return res;
// 
// }
ALWAYS_INLINE __m256d fast_inv_Phi_wo_checks_AVX2 VECTORCALL(const __m256d p) {
  
      ALIGN32 __m256d const q       = _mm256_sub_pd(p, _mm256_set1_pd(0.5));
      ALIGN32 __m256d const central = _mm256_cmp_pd(_mm256_abs_pd(q), _mm256_set1_pd(0.425), _CMP_LE_OQ);
      ALIGN32 __m256d const v1 = fast_inv_Phi_wo_checks_case_1_AVX2(p, q);
      // The tail result would be discarded in every lane. Avoid its log and two rational polynomials.
      if (_mm256_movemask_pd(central) == 0xf) return v1;
      ALIGN32 __m256d const v2 = fast_inv_Phi_wo_checks_case_2_AVX2(p, q);
      return _mm256_blendv_pd(v2, v1, central);   // central lanes -> v1
      
}




// =============================================================================================
//  AVX2   (identical logic; blendv(a, b, m) == mask_blend(m, a, b), so argument order is unchanged)
// =============================================================================================
ALWAYS_INLINE __m256d fast_mills_ratio_tail_AVX2 VECTORCALL(const __m256d x) {
      __m256d f = x;
      for (int k = MILLS_CF_DEPTH; k >= 1; --k) {
          f = _mm256_add_pd(x, _mm256_div_pd(_mm256_set1_pd(static_cast<double>(k)), f));
      }
      return _mm256_div_pd(_mm256_set1_pd(1.0), f);
}




ALWAYS_INLINE __m256d fast_log_Phi_AVX2 VECTORCALL(const __m256d x) {
      const __m256d thresh   = _mm256_set1_pd(NORMAL_TAIL_THRESH);
      const __m256d ax       = _mm256_abs_pd(x);
      const __m256d R        = fast_mills_ratio_tail_AVX2(_mm256_max_pd(ax, thresh));
      const __m256d log_tail = _mm256_add_pd(_mm256_fnmadd_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(x, x), _mm256_set1_pd(NEG_HALF_LOG_2PI)),
                                             fast_log_1_AVX2(R));
      const __m256d log_body = fast_log_1_AVX2(fast_Phi_AVX2(x));
      const __m256d is_tail  = _mm256_cmp_pd(x, _mm256_sub_pd(_mm256_setzero_pd(), thresh), _CMP_LT_OQ);
      return _mm256_blendv_pd(log_body, log_tail, is_tail);
}




ALWAYS_INLINE __m256d fast_log_1m_Phi_AVX2 VECTORCALL(const __m256d x) {
      return fast_log_Phi_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x));
}




ALWAYS_INLINE __m256d fast_dlog_Phi_dx_AVX2 VECTORCALL(const __m256d x) {
      const __m256d thresh  = _mm256_set1_pd(NORMAL_TAIL_THRESH);
      const __m256d ax      = _mm256_abs_pd(x);
      const __m256d R       = fast_mills_ratio_tail_AVX2(_mm256_max_pd(ax, thresh));
      const __m256d d_tail  = _mm256_div_pd(_mm256_set1_pd(1.0), R);
      const __m256d phi     = _mm256_mul_pd(_mm256_set1_pd(INV_SQRT_2PI),
                                            fast_exp_1_AVX2(_mm256_mul_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(x, x))));
      const __m256d d_body  = _mm256_div_pd(phi, fast_Phi_AVX2(x));
      const __m256d is_tail = _mm256_cmp_pd(x, _mm256_sub_pd(_mm256_setzero_pd(), thresh), _CMP_LT_OQ);
      return _mm256_blendv_pd(d_body, d_tail, is_tail);
}




ALWAYS_INLINE __m256d fast_inv_Phi_from_log_p_wo_checks_AVX2 VECTORCALL( const __m256d log_p, 
                                                                         const __m256d log_1m_p) {
      const __m256d zero    = _mm256_setzero_pd();
      const __m256d p       = fast_exp_1_AVX2(log_p);
      const __m256d q       = _mm256_sub_pd(p, _mm256_set1_pd(0.5));
      const __m256d central = _mm256_cmp_pd(_mm256_abs_pd(q), _mm256_set1_pd(0.425), _CMP_LE_OQ);
      const __m256d log_min = _mm256_min_pd(log_p, log_1m_p);
      const __m256d r       = _mm256_sqrt_pd(_mm256_sub_pd(zero, log_min));
      const __m256d r_gr_5  = _mm256_cmp_pd(r, _mm256_set1_pd(5.0), _CMP_GT_OQ);
      __m256d v_tail = _mm256_blendv_pd(fast_inv_Phi_wo_checks_case_2a_AVX2(p, r),
                                        fast_inv_Phi_wo_checks_case_2b_AVX2(p, r),
                                        r_gr_5);
      const __m256d lower   = _mm256_cmp_pd(log_p, log_1m_p, _CMP_LT_OQ);
      v_tail = _mm256_blendv_pd(v_tail, _mm256_sub_pd(zero, v_tail), lower);
      const __m256d v_central = fast_inv_Phi_wo_checks_case_1_AVX2(p, q);
      return _mm256_blendv_pd(v_tail, v_central, central);
}




ALWAYS_INLINE __m256d fast_dZ_dv_from_log_p_AVX2 VECTORCALL( const __m256d Z, 
                                                             const __m256d log_p, 
                                                             const __m256d log_1m_p,
                                                             const __m256d dlog_p_dv, 
                                                             const __m256d dlog_1m_p_dv) {
  
      const __m256d log_inv_phi = _mm256_fmadd_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(Z, Z), _mm256_set1_pd(-NEG_HALF_LOG_2PI));
      const __m256d cap    = _mm256_set1_pd(700.0);
      const __m256d e_lo   = _mm256_min_pd(_mm256_add_pd(log_p,    log_inv_phi), cap);
      const __m256d e_hi   = _mm256_min_pd(_mm256_add_pd(log_1m_p, log_inv_phi), cap);
      const __m256d via_lo = _mm256_mul_pd(fast_exp_1_AVX2(e_lo), dlog_p_dv);
      const __m256d via_hi = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), fast_exp_1_AVX2(e_hi)), dlog_1m_p_dv);
      const __m256d Z_neg  = _mm256_cmp_pd(Z, _mm256_setzero_pd(), _CMP_LT_OQ);
      return _mm256_blendv_pd(via_hi, via_lo, Z_neg);
      
}




#endif




#endif
 
 
 
 
 
