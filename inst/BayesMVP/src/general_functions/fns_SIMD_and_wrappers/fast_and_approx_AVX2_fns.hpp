#pragma once 


#ifndef FAST_AND_APPROX_AVX256_AVX2_FNS_HPP
#define FAST_AND_APPROX_AVX256_AVX2_FNS_HPP

 
 
 
 
 
#include <immintrin.h>
#include <cmath>
  
  
  
#if defined(__AVX2__) && ( !(defined(__AVX256VL__) && defined(__AVX256F__)  && defined(__AVX256DQ__)) ) // use AVX2 if AVX-256 not available 
 

//// -------------------------------------------------------------------------------------------------------------------------------------------------------------

 
 
 
 
 

 
 
 // is_finite_mask and is_not_NaN_mask for AVX2
 inline __m256d is_finite_mask(__m256d x) {
       
       const __m256d sign_bit = _mm256_set1_pd(-0.0);
       
       const __m256d INF = _mm256_set1_pd(std::numeric_limits<double>::infinity());
       const __m256d abs_x = _mm256_andnot_pd(sign_bit, x);
       
       return _mm256_cmp_pd(abs_x, INF, _CMP_NEQ_OQ); 
   
 }

inline __m256d is_not_NaN_mask(__m256d x) {
 
       return _mm256_cmp_pd(x, x, _CMP_EQ_OQ);
 
}


 
 
 
 
// bookmark (this doesnt seem to be available on g++ at all at time of writing even with all AVX256 flags!)
//#if !defined(__AVX256VL__)
#define _mm256_and_epi64(a, b) _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b))) // this uses AVX2 instructions so moree widely supported
 
 
 
 
 
inline    __m256d  _mm256_abs_pd(const __m256d x) {
  
    const __m256d sign_bit = _mm256_set1_pd(-0.0);
    const __m256d x_abs = _mm256_andnot_pd(sign_bit, x);
    
    return x_abs;
  
}
 



  
  
 



///////////////////  fns - exp   -----------------------------------------------------------------------------------------------------------------------------
 

inline    __m256d fast_ldexp(const __m256d AVX_a,
                             const __m256i AVX_i) {
   
  return  _mm256_castsi256_pd (_mm256_add_epi64 (_mm256_slli_epi64 (AVX_i, 52ULL), _mm256_castpd_si256(AVX_a))); /* AVX_a = p * 2^AVX_i */
  
  
}

 

// Fast implementation of ldexp (multiply by power of 2) using AVX2 intrinsics
// Computes: a * 2^i for 4 pairs of values simultaneously
// Handles edge cases by splitting large exponents into two steps
 
inline __m256d fast_ldexp_2(const __m256d AVX_a, 
                            const __m256i AVX_i) {
  
      // Get sign mask: for each 64-bit integer, shift right by 63 bits (arithmetic shift)
      // This replicates the sign bit across all bits: 0x0000... for positive, 0xFFFF... for negative
      const __m256i neg_mask = _mm256_srai_epi64(AVX_i, 63);
      
      // Calculate absolute value using the formula: abs(x) = (x XOR sign_mask) - sign_mask
      // 1. XOR with sign mask flips all bits if negative, keeps same if positive
      // 2. Subtract sign mask adds 1 for negative numbers (two's complement)
      const __m256i abs_i = _mm256_sub_epi64(_mm256_xor_si256(AVX_i, neg_mask), neg_mask);
      
      // Create vector with all elements = 1000 (our safe threshold for exponents)
      const __m256i threshold = _mm256_set1_epi64x(1000);
      
      // Compare abs_i with threshold by subtraction
      // If abs_i > threshold, result will be positive
      const __m256i cmp = _mm256_sub_epi64(abs_i, threshold);
      
      // Create comparison mask by getting sign bits of comparison result
      // Will be 0xFFFF... where abs_i <= threshold, 0x0000... where abs_i > threshold
      const __m256i cmp_mask = _mm256_srai_epi64(cmp, 63);
      
      // Convert comparison mask to 4-bit integer (one bit per double)
      // If result != 0xF (not all bits set), then at least one value exceeded threshold
      if (_mm256_movemask_pd(_mm256_castsi256_pd(cmp_mask)) != 0xF) {
              
              // Handle large exponents by splitting into two steps
              
              // Create i1 = ±1000 based on original sign:
              // 1. AND neg_mask with -2000 gives -2000 for negative numbers, 0 for positive
              // 2. XOR with threshold (1000) gives -1000 for negative numbers, 1000 for positive
              const __m256i i1 = _mm256_xor_si256(_mm256_and_si256(neg_mask, _mm256_set1_epi64x(-2000)), threshold);
              
              // Calculate remaining exponent: i2 = original_i - i1
              const __m256i i2 = _mm256_sub_epi64(AVX_i, i1);
              
              // First scaling step: multiply by 2^i1
              // 1. Shift i1 left by 52 to position it in double's exponent field
              // 2. Add to bit pattern of input doubles (effectively multiplying by 2^i1)
              const __m256d mid = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_slli_epi64(i1, 52), _mm256_castpd_si256(AVX_a))); 
              
              // Second scaling step: multiply intermediate result by 2^i2
              // Same process as above but with i2 and intermediate result
              // return _mm256_castsi256_pd(_mm256_add_epi64(_mm256_slli_epi64(i2, 52), _mm256_castpd_si256(mid)));
              return  fast_ldexp(mid, i2); 
        
      }
      
      // For small exponents, perform direct scaling:
      // 1. Shift exponent left by 52 bits to align with double's exponent field
      // 2. Add to original number's bit pattern (effectively multiplying by 2^i)
      // return _mm256_castsi256_pd(_mm256_add_epi64(_mm256_slli_epi64(AVX_i, 52), _mm256_castpd_si256(AVX_a)));
      return  fast_ldexp(AVX_a, AVX_i); 
  
}

 



 
 
 


// Helper function for AVX2 64-bit conversion
inline __m256i avx2_cvtpd_epi64(__m256d x) {
 
     // Extract doubles and convert to int64 one at a time
     alignas(32) int64_t result[4];
     alignas(32) double temp[4];
     _mm256_store_pd(temp, x);
     
     for(int i = 0; i < 4; i++) {
       result[i] = (int64_t)std::llrint(temp[i]);  // Use llrint for proper rounding
     } 
     
     return _mm256_load_si256((__m256i*)result);
 
}

 
 
// Adapted from: https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
// added   (optional) extra degree(s) for poly approx (oroginal float fn had 4 degrees) - using "minimaxApprox" R package to find coefficient terms
// R code:    minimaxApprox::minimaxApprox(fn = exp, lower = -0.346573590279972643113, upper = 0.346573590279972643113, degree = 5, basis ="Chebyshev")
inline    __m256d fast_exp_1_wo_checks_AVX2(const __m256d x)  {
  
  
  
  const __m256d exp_l2e = _mm256_set1_pd (1.442695040888963387); /* log2(e) */
  const __m256d exp_l2h = _mm256_set1_pd (-0.693145751999999948367); /* -log(2)_hi */
  const __m256d exp_l2l = _mm256_set1_pd (-0.00000142860676999999996193); /* -log(2)_lo */
  
  // /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
  const __m256d exp_c0 =     _mm256_set1_pd(0.00000276479776161191821278);
  const __m256d exp_c1 =     _mm256_set1_pd(0.0000248844480527491290235);
  const __m256d exp_c2 =     _mm256_set1_pd(0.000198411488032534342194);
  const __m256d exp_c3 =     _mm256_set1_pd(0.00138888017711994078175);
  const __m256d exp_c4 =     _mm256_set1_pd(0.00833333340524595143906);
  const __m256d exp_c5 =     _mm256_set1_pd(0.0416666670404215802592);
  const __m256d exp_c6 =     _mm256_set1_pd(0.166666666664891632843);
  const __m256d exp_c7 =     _mm256_set1_pd(0.499999999994389376923);
  const __m256d exp_c8 =     _mm256_set1_pd(1.00000000000001221245);
  const __m256d exp_c9 =     _mm256_set1_pd(1.00000000000001332268);
  
  const __m256d input  = x;
  
  /* exp(x) = 2^i * e^f; i = rint (log2(e) * a), f = a - log(2) * i */
  const __m256d t = _mm256_mul_pd(x, exp_l2e);      /* t = log2(e) * a */
  ///  const __m256i i = _mm256_cvttpd_epi32(t);       /* i = (int)rint(t) */
  const __m256i i = avx2_cvtpd_epi64(t);  
  const __m256d x_2 = _mm256_round_pd(t, _MM_FROUND_TO_NEAREST_INT) ; // ((0<<4)| _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC|_MM_FROUND_NO_EXC));
  const __m256d f = _mm256_fmadd_pd(x_2, exp_l2l, _mm256_fmadd_pd (x_2, exp_l2h, input));  /* a - log(2)_hi * r */    /* f = a - log(2)_hi * r - log(2)_lo * r */
  
  /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
  __m256d p = exp_c0;
  p = _mm256_fmadd_pd(p, f, exp_c1);
  p = _mm256_fmadd_pd(p, f, exp_c2);
  p = _mm256_fmadd_pd(p, f, exp_c3);
  p = _mm256_fmadd_pd(p, f, exp_c4);
  p = _mm256_fmadd_pd(p, f, exp_c5);
  p = _mm256_fmadd_pd(p, f, exp_c6);
  p = _mm256_fmadd_pd(p, f, exp_c7);
  p = _mm256_fmadd_pd(p, f, exp_c8);
  p = _mm256_fmadd_pd(p, f, exp_c9);
  
  return  fast_ldexp(p, i) ;    /* exp(x) = 2^i * p */
  
}
 


 
 

 
// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
inline __m256d fast_exp_1_AVX2(const __m256d a) {
  
  const __m256d   exp_bound  =   _mm256_set1_pd(708.4);
  const __m256d   pos_inf  =    _mm256_set1_pd(INFINITY);
  
  return _mm256_blendv_pd(
    _mm256_blendv_pd(
      _mm256_blendv_pd( 
        pos_inf,
        _mm256_setzero_pd(),
        _mm256_cmp_pd(a, _mm256_setzero_pd(), _CMP_LT_OQ)
      ),
      _mm256_set1_pd(INFINITY - INFINITY),
      is_not_NaN_mask(a)
    ),  
    fast_exp_1_wo_checks_AVX2(a),
    _mm256_cmp_pd(_mm256_abs_pd(a), exp_bound, _CMP_LT_OQ)
  );
  
}
 


 
 











///////////////////  fns - log  (+ related e.g. log1m etc) -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



 /// const __m256d i = _mm256_fmadd_pd(_mm256_cvtepi64_pd(e), log_c1, _mm256_setzero_pd()); // Replace as needs AVX-256
 /// replace with manual fn:

 
inline __m256d avx2_cvtepi64_pd(__m256i v) {

  
      // Extract high and low 64-bit integers
      __m128i low = _mm256_castsi256_si128(v);
      __m128i high = _mm256_extracti128_si256(v, 1);
      
      // Convert to doubles one element at a time
      double vals[4];
      vals[0] = (double)_mm_extract_epi64(low, 0);
      vals[1] = (double)_mm_extract_epi64(low, 1);
      vals[2] = (double)_mm_extract_epi64(high, 0);
      vals[3] = (double)_mm_extract_epi64(high, 1);
      
      // Load back into __m256d
      return _mm256_loadu_pd(vals);
  
}



 

//https://stackoverflow.com/a/65537754/9007125 // vectorized version of the answer by njuffa
inline     __m256d fast_log_1_wo_checks_AVX2(const __m256d a ) {
  
  
  
  const __m256i log_i1 =   _mm256_set1_epi64x(0x3fe5555555555555);
  const __m256i log_i2 =   _mm256_set1_epi64x(0xFFF0000000000000);
  
  const __m256d log_c1 =   _mm256_set1_pd(0.000000000000000222044604925031308085);
  const __m256d log_c2 =   _mm256_set1_pd(-0.13031005859375);
  const __m256d log_c3 =   _mm256_set1_pd(0.140869140625);
  const __m256d log_c4 =   _mm256_set1_pd(-0.121483256222766876221);
  const __m256d log_c5 =   _mm256_set1_pd(0.139814853668212890625);
  const __m256d log_c6 =   _mm256_set1_pd(-0.166846126317977905273);
  const __m256d log_c7 =   _mm256_set1_pd(0.200120344758033752441);
  const __m256d log_c8 =   _mm256_set1_pd(-0.249996200203895568848);
  const __m256d log_c9 =   _mm256_set1_pd(0.333331972360610961914);
  const __m256d log_c10 =  _mm256_set1_pd(-0.5);
  const __m256d log_c11 =  _mm256_set1_pd(0.693147182464599609375);
  
  
  const __m256i aInt = _mm256_castpd_si256(a);
  const __m256i e = _mm256_and_epi64( _mm256_sub_epi64(aInt, log_i1),  log_i2); //    e = (__double_as_int (a) - i1 )    &   i2   ;
  const __m256d i = _mm256_fmadd_pd ( avx2_cvtepi64_pd(e), log_c1, _mm256_setzero_pd()); // 0x1.0p-52
  const __m256d m = _mm256_sub_pd(_mm256_castsi256_pd( _mm256_sub_epi64(aInt, e)),  _mm256_set1_pd(1.0)) ;  //   m = __int_as_double (__double_as_int (a) - e);   //  m = _mm256_sub_pd(m, one) ; // m - 1.0;  /* m in [2/3, 4/3] */
  const __m256d s = _mm256_mul_pd(m, m);  // m = _mm256_sub_pd(m,  _mm256_set1_pd(1.0)) ; // m - 1.0;  /* m in [2/3, 4/3] */
  
  /* Compute log1p(m) for m in [-1/3, 1/3] */
  __m256d   r, t;
  r =             log_c2;  // -0x1.0ae000p-3
  t =             log_c3;  //  0x1.208000p-3
  r = _mm256_fmadd_pd (r, s, log_c4); // -0x1.f198b2p-4
  t = _mm256_fmadd_pd (t, s, log_c5); //  0x1.1e5740p-3
  r = _mm256_fmadd_pd (r, s, log_c6); // -0x1.55b36cp-3
  t = _mm256_fmadd_pd (t, s, log_c7); //  0x1.99d8b2p-3
  r = _mm256_fmadd_pd (r, s, log_c8); // -0x1.fffe02p-3
  r = _mm256_fmadd_pd (t, m, r);
  r = _mm256_fmadd_pd (r, m, log_c9); //  0x1.5554fap-2
  r = _mm256_fmadd_pd (r, m, log_c10); // -0x1.000000p-1
  r = _mm256_fmadd_pd (r, s, m);
  
  return _mm256_fmadd_pd (i,  log_c11, r); //  0x1.62e430p-1 // log(2)
  
}


inline __m256d fast_log1p_1_wo_checks_AVX2(const __m256d x) {
  
    const __m256d small =  _mm256_set1_pd(1e-4);
    const __m256d minus_one_half  =  _mm256_set1_pd(-0.50);
  
    const __m256d mask = _mm256_cmp_pd(_mm256_abs_pd(x), small, _CMP_GT_OQ);     // Create mask using _mm256_cmp_pd
    
    return _mm256_blendv_pd(
      _mm256_mul_pd(_mm256_fmadd_pd(minus_one_half, x,  _mm256_set1_pd(1.0)), x),
      fast_log_1_wo_checks_AVX2(_mm256_add_pd(x,  _mm256_set1_pd(1.0))),
      mask
    );
  
}

inline __m256d fast_log1m_1_wo_checks_AVX2(const __m256d x) {
  
     return fast_log1p_1_wo_checks_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x));
  
} 

inline __m256d fast_log1p_exp_1_wo_checks_AVX2(const __m256d x) {
  
      const __m256d neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
  
      const __m256d mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ);     // Create mask using _mm256_cmp_pd
      
      return _mm256_blendv_pd(
        fast_log1p_1_wo_checks_AVX2(fast_exp_1_wo_checks_AVX2(x)),
        _mm256_add_pd(x, fast_log1p_1_wo_checks_AVX2(fast_exp_1_wo_checks_AVX2(neg_x))),
        mask
      );
  
}






// //https://stackoverflow.com/a/65537754/9007125
inline __m256d fast_log_1_AVX2(const __m256d a) {
  
  const __m256d  pos_inf  =    _mm256_set1_pd(INFINITY);
  const __m256d  neg_inf  =    _mm256_set1_pd(-INFINITY);
  
  return _mm256_blendv_pd(
    _mm256_blendv_pd(
      _mm256_sub_pd(pos_inf, pos_inf),
      neg_inf,
      _mm256_cmp_pd(_mm256_setzero_pd(), a, _CMP_EQ_OQ)
    ),
    fast_log_1_wo_checks_AVX2(a),
    _mm256_and_pd( 
      is_finite_mask(a),
      _mm256_cmp_pd(a, _mm256_setzero_pd(), _CMP_GT_OQ)
    )  
  );
  
}   



// //  compute log(1+x) without losing precision for small values of x.
// //  see: https://www.johndcook.com/cpp_log_one_plus_x.html


inline __m256d fast_log1p_1_AVX2(const __m256d x) {
  
  const __m256d minus_1 =  _mm256_set1_pd(-1.0);
  const __m256d neg_half = _mm256_set1_pd(-0.5);
  const __m256d sign_bit = _mm256_set1_pd(-0.0);
  
  return _mm256_blendv_pd(
    _mm256_blendv_pd(
      _mm256_mul_pd(_mm256_fmadd_pd(neg_half, x,  _mm256_set1_pd(1.0)), x),
      fast_log_1_AVX2(_mm256_add_pd(x,  _mm256_set1_pd(1.0))),
      _mm256_cmp_pd(_mm256_andnot_pd(sign_bit, x), _mm256_set1_pd(1e-4), _CMP_GT_OQ)
    ), 
    fast_log_1_AVX2(_mm256_add_pd(x,  _mm256_set1_pd(1.0))),
    _mm256_cmp_pd(x, minus_1, _CMP_LE_OQ)
  );
  
} 



// //  compute log(1-x) without losing precision for small values of x.
inline __m256d fast_log1m_1_AVX2(const __m256d x) {
  
  return fast_log1p_1_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x));
  
}  


// //  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
inline __m256d fast_log1p_exp_1_AVX2(const __m256d x) {
  
  const __m256d neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
  
  return _mm256_blendv_pd(
    fast_log1p_1_AVX2(fast_exp_1_AVX2(x)),
    _mm256_add_pd(x, fast_log1p_1_AVX2(fast_exp_1_AVX2(neg_x))), 
    _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ)
  );
   
}



inline     __m256d fast_logit_wo_checks_AVX2(const __m256d x)   {
  
  const __m256d log_x = fast_log_1_AVX2(x);
  const __m256d log_1m_x =  fast_log1m_1_AVX2(x);
  return  log_x - log_1m_x;
  
} 



inline     __m256d fast_logit_AVX2(const __m256d x)   {
  
  const __m256d log_x = fast_log_1_AVX2(x);
  const __m256d log_1m_x =  fast_log1m_1_AVX2(x);
  return  log_x - log_1m_x;
  
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

 

inline     __m256d  fast_inv_logit_for_x_pos_AVX2(const __m256d x )  {
  
          const __m256d  exp_m_x =  fast_exp_1_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x)) ;
          
          return    _mm256_div_pd( _mm256_set1_pd(1.0), _mm256_add_pd( _mm256_set1_pd(1.0), exp_m_x))  ;
}



inline     __m256d  fast_inv_logit_for_x_neg_AVX2(const __m256d x )  {
  
          const __m256d  log_eps  =   _mm256_set1_pd(-18.420680743952367);
          const __m256d  exp_x =  fast_exp_1_AVX2(x) ;
          
          return _mm256_blendv_pd(    exp_x,
                                      _mm256_div_pd(exp_x, _mm256_add_pd( _mm256_set1_pd(1.0), exp_x)), 
                                      _mm256_cmp_pd(x,  log_eps, _CMP_GT_OQ));
}

inline     __m256d  fast_inv_logit_AVX2(const __m256d x )  {
  
    return  _mm256_blendv_pd(  fast_inv_logit_for_x_neg_AVX2(x),
                               fast_inv_logit_for_x_pos_AVX2(x), 
                               _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ));  // x > 0
  
}

 

 

 
 
inline      __m256d  fast_inv_logit_for_x_pos_wo_checks_AVX2(const __m256d x )  {

  const __m256d  exp_m_x =  fast_exp_1_wo_checks_AVX2(_mm256_sub_pd(_mm256_setzero_pd(), x)) ;
  
  return    _mm256_div_pd( _mm256_set1_pd(1.0), _mm256_add_pd( _mm256_set1_pd(1.0), exp_m_x))  ;
  
}

inline     __m256d  fast_inv_logit_for_x_neg_wo_checks_AVX2(const __m256d x )  {
  
  const __m256d  log_eps  =   _mm256_set1_pd(-18.420680743952367);
  const __m256d  exp_x =  fast_exp_1_wo_checks_AVX2(x) ;
  
  return _mm256_blendv_pd(    exp_x,
                              _mm256_div_pd(exp_x, _mm256_add_pd( _mm256_set1_pd(1.0), exp_x)), 
                              _mm256_cmp_pd(x,  log_eps, _CMP_GT_OQ));
  
}

inline      __m256d  fast_inv_logit_wo_checks_AVX2(const __m256d x )  {
  
  return  _mm256_blendv_pd(    fast_inv_logit_for_x_neg_wo_checks_AVX2(x),
                               fast_inv_logit_for_x_pos_wo_checks_AVX2(x), 
                               _mm256_cmp_pd(x,  _mm256_setzero_pd(), _CMP_GT_OQ));  // x > 0
  
}

 


 



///////////////////  fns - log_inv_logit   ----------------------------------------------------------------------------------------------------------------


////////////

inline     __m256d  fast_log_inv_logit_for_x_pos_AVX2(const __m256d x )  {
  
    const    __m256d   m_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
  
    return   _mm256_sub_pd(_mm256_setzero_pd(), fast_log1p_exp_1_AVX2(fast_exp_1_AVX2(m_x)));
    
}

inline     __m256d  fast_log_inv_logit_for_x_neg_AVX2(const __m256d x )  {

    const __m256d  log_eps  =   _mm256_set1_pd(-18.420680743952367);
  
    return _mm256_blendv_pd(    x,
                                _mm256_sub_pd(x,   fast_log1p_1_AVX2(fast_exp_1_AVX2(x))),
                                _mm256_cmp_pd(x,  log_eps, _CMP_GT_OQ));
  
}

inline     __m256d  fast_log_inv_logit_AVX2(const __m256d x )  {
  
    return  _mm256_blendv_pd(    fast_log_inv_logit_for_x_neg_AVX2(x),
                                 fast_log_inv_logit_for_x_pos_AVX2(x), 
                                 _mm256_cmp_pd(x,  _mm256_setzero_pd(), _CMP_GE_OQ));  // x > 0
}
 



 



 
////////////
inline     __m256d  fast_log_inv_logit_for_x_pos_wo_checks_AVX2(const __m256d x )  {
  
   return   _mm256_sub_pd(_mm256_setzero_pd(), fast_log1p_exp_1_wo_checks_AVX2((_mm256_sub_pd(_mm256_setzero_pd(), x)) ) );
   
}

inline     __m256d  fast_log_inv_logit_for_x_neg_wo_checks_AVX2(const __m256d x )  {
  
    const __m256d  log_eps  =   _mm256_set1_pd(-18.420680743952367);
  
    return _mm256_blendv_pd(    x,
                                _mm256_sub_pd(x,   fast_log1p_1_wo_checks_AVX2(fast_exp_1_wo_checks_AVX2(x))),
                                _mm256_cmp_pd(x,  log_eps, _CMP_GT_OQ));
    
}

inline     __m256d  fast_log_inv_logit_wo_checks_AVX2(const __m256d x )  {
  
    return  _mm256_blendv_pd(    fast_log_inv_logit_for_x_neg_wo_checks_AVX2(x),
                                 fast_log_inv_logit_for_x_pos_wo_checks_AVX2(x),
                                 _mm256_cmp_pd(x,  _mm256_setzero_pd(), _CMP_GE_OQ));  // x > 0
  
}
 

 


 
/////////////////////  fns - Phi_approx   ----------------------------------------------------------------------------------------------------------------
 
 


 
inline     __m256d  fast_Phi_approx_wo_checks_AVX2(const __m256d x)  {
  
  const __m256d a =     _mm256_set1_pd(0.07056);
  const __m256d b =     _mm256_set1_pd(1.5976);
  
  const __m256d   x_sq =  _mm256_mul_pd(x, x);
  const __m256d   a_x_sq_plus_b = _mm256_fmadd_pd(a, x_sq, b);
  const __m256d   stuff_to_inv_logit =  _mm256_mul_pd(x, a_x_sq_plus_b);
  
  return fast_inv_logit_wo_checks_AVX2(stuff_to_inv_logit);
  
}


 
 
inline     __m256d  fast_Phi_approx_AVX2(const __m256d x )  {
  
  const __m256d a =     _mm256_set1_pd(0.07056);
  const __m256d b =     _mm256_set1_pd(1.5976);
  
  const __m256d  x_sq =  _mm256_mul_pd(x, x);
  const __m256d  a_x_sq_plus_b = _mm256_fmadd_pd(a, x_sq, b);
  const __m256d   stuff_to_inv_logit =  _mm256_mul_pd(x, a_x_sq_plus_b);
  
  return fast_inv_logit_AVX2(stuff_to_inv_logit);
  
}

 


///////////////////  fns - inv_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

   


inline      __m256d  fast_inv_Phi_approx_wo_checks_AVX2(const __m256d x )  {
  
  const __m256d  inv_Phi_approx_c1 = _mm256_set1_pd(-0.3418);
  const __m256d  inv_Phi_approx_c2 = _mm256_set1_pd(2.74699999999999988631);
  const __m256d  one_third =  _mm256_set1_pd(0.33333333333333331483);
  
  const __m256d  m_logit_p = fast_log_1_wo_checks_AVX2( _mm256_sub_pd(_mm256_div_pd( _mm256_set1_pd(1.0),  x),  _mm256_set1_pd(1.0) ) ); // logit first
  const __m256d  x_i =  _mm256_mul_pd(m_logit_p, inv_Phi_approx_c1) ; // -0.3418*m_logit_p;
  const __m256d  asinh_stuff_div_3 =  _mm256_mul_pd( one_third,  fast_log_1_wo_checks_AVX2(  _mm256_add_pd(x_i, (  _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ) ;          // now do arc_sinh part
  const __m256d  exp_x_i = fast_exp_1_wo_checks_AVX2(asinh_stuff_div_3);
  
  return _mm256_mul_pd(  inv_Phi_approx_c2,   _mm256_div_pd(_mm256_sub_pd(_mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;    //   now do sinh parth part
  
}
 



 


 

inline     __m256d  fast_inv_Phi_approx_AVX2(const __m256d x )  {
  
  const __m256d  one_third =  _mm256_set1_pd(0.33333333333333331483);
  
  const __m256d m_logit_p = fast_log_1_AVX2( _mm256_sub_pd(_mm256_div_pd( _mm256_set1_pd(1.0),  x),  _mm256_set1_pd(1.0) ) ); // logit first
  const __m256d x_i =  _mm256_mul_pd(m_logit_p, _mm256_set1_pd(-0.3418)) ; // -0.3418*m_logit_p;
  const __m256d asinh_stuff_div_3 =  _mm256_mul_pd(  one_third,  fast_log_1_AVX2(  _mm256_add_pd(x_i, (  _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ) ;          // now do arc_sinh part
  const __m256d exp_x_i = fast_exp_1_AVX2(asinh_stuff_div_3);
  return _mm256_mul_pd(  _mm256_set1_pd(2.74699999999999988631),   _mm256_div_pd( _mm256_sub_pd(_mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;    //   now do sinh parth part
  
}


 
 




 

 
 


// need to add citation to this (slight modification from a forum post)
inline      __m256d  fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX2(const __m256d logit_p )  {
  
  const __m256d  inv_Phi_approx_c1 = _mm256_set1_pd(-0.3418);
  const __m256d  inv_Phi_approx_c2 = _mm256_set1_pd(2.74699999999999988631);
  const __m256d  one_third =  _mm256_set1_pd(0.33333333333333331483);
  
  const __m256d  x_i =  _mm256_mul_pd(logit_p,  inv_Phi_approx_c1) ;
  const __m256d  asinh_stuff_div_3 =  _mm256_mul_pd(  one_third,  fast_log_1_wo_checks_AVX2( _mm256_add_pd(x_i  , _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ;          // now do arc_sinh part
  const __m256d  exp_x_i = fast_exp_1_wo_checks_AVX2(asinh_stuff_div_3);
  
  return _mm256_mul_pd(inv_Phi_approx_c2,   _mm256_div_pd(_mm256_sub_pd( _mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;   //   now do sinh parth part
  
  
}





inline     __m256d  fast_inv_Phi_approx_from_logit_prob_AVX2(const __m256d logit_p )  {
  
  const __m256d  inv_Phi_approx_c1 = _mm256_set1_pd(-0.3418);
  const __m256d  inv_Phi_approx_c2 = _mm256_set1_pd(2.74699999999999988631);
  const __m256d  one_third =  _mm256_set1_pd(0.33333333333333331483);
  
  const __m256d  x_i =  _mm256_mul_pd(logit_p,  inv_Phi_approx_c1) ;
  const __m256d  asinh_stuff_div_3 =  _mm256_mul_pd( one_third,  fast_log_1_AVX2( _mm256_add_pd(x_i  , _mm256_sqrt_pd(  _mm256_fmadd_pd(x_i, x_i,  _mm256_set1_pd(1.0)) ) )  ) ) ;          // now do arc_sinh part
  const __m256d  exp_x_i = fast_exp_1_AVX2(asinh_stuff_div_3);
  
  return _mm256_mul_pd(inv_Phi_approx_c2,   _mm256_div_pd(_mm256_sub_pd( _mm256_mul_pd(exp_x_i, exp_x_i),  _mm256_set1_pd(1.0)), exp_x_i) ) ;   //   now do sinh parth part
  
}

 

 




///////////////////  fns - log_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------


 
 
inline      __m256d  fast_log_Phi_approx_wo_checks_AVX2(const __m256d x )  {
  
  const __m256d a =     _mm256_set1_pd(0.07056);
  const __m256d b =     _mm256_set1_pd(1.5976);
  
 const __m256d  x_sq =  _mm256_mul_pd(x, x);
 const __m256d  a_x_sq_plus_b = _mm256_fmadd_pd(a, x_sq, b);
 const __m256d  stuff_to_inv_logit =  _mm256_mul_pd(x, a_x_sq_plus_b);
  
  return fast_log_inv_logit_wo_checks_AVX2(stuff_to_inv_logit);
  
}
 




 

 


inline __m256d fast_log_Phi_approx_AVX2(const __m256d x) {
  
  const __m256d a =     _mm256_set1_pd(0.07056);
  const __m256d b =     _mm256_set1_pd(1.5976);
  const __m256d Phi_upper_bound =  _mm256_set1_pd(8.25);
  const __m256d Phi_lower_bound =  _mm256_set1_pd(-37.5);
  
  const __m256d x_sq = _mm256_mul_pd(x, x);
  const __m256d x_cubed = _mm256_mul_pd(x_sq, x);
  const __m256d result = _mm256_fmadd_pd(a, x_cubed, _mm256_mul_pd(b, x));
  
  return fast_log_inv_logit_AVX2(result);
  
} 

 

 


//////////////////// ------------- tanh  --------------------------------------------------------------------------------------------------------------------------------------


 


inline      __m256d    fast_tanh_AVX2( const  __m256d x  )   {
  
  
  const __m256d  two =     _mm256_set1_pd(2.0);
  const __m256d  m_two =   _mm256_set1_pd(-2.0);
  
  return        _mm256_sub_pd( _mm256_div_pd(two,   _mm256_add_pd( _mm256_set1_pd(1.0), fast_exp_1_AVX2(  _mm256_mul_pd(x, m_two)   ) ) ),  _mm256_set1_pd(1.0) ) ;
  
}
 
 
 
inline      __m256d    fast_tanh_wo_checks_AVX2(const   __m256d x  )   {
  
  const __m256d  two =     _mm256_set1_pd(2.0);
  const __m256d  m_two =   _mm256_set1_pd(-2.0);
  
  return        _mm256_sub_pd( _mm256_div_pd(two,   _mm256_add_pd( _mm256_set1_pd(1.0), fast_exp_1_wo_checks_AVX2(  _mm256_mul_pd(x, m_two)   ) ) ),  _mm256_set1_pd(1.0) ) ;
  
}
 


// from: https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
inline    __m256d CopySign(const __m256d srcSign, 
                           const __m256d srcValue) {
    
    const __m256d mask0 = _mm256_set1_pd(-0.);
    
    __m256d tmp0 = _mm256_and_pd(srcSign, mask0); // Extract the signed bit from srcSign
    __m256d tmp1 = _mm256_andnot_pd(mask0, srcValue); // Extract the number without sign of srcValue (abs(srcValue))
    
    return _mm256_or_pd(tmp0, tmp1);  // Merge signed bit with number and return
  
}


  
 
  
 





///////////////////  fns - Phi functions   ----------------------------------------------------------------------------------------------------------------------------------------


 


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
 inline __m256d fast_Phi_wo_checks_AVX2(const __m256d x) {
   
   const __m256d a =  _mm256_set1_pd(0.2316419);
   const __m256d b1 = _mm256_set1_pd(0.31938153);
   const __m256d b2 = _mm256_set1_pd(-0.356563782);
   const __m256d b3 = _mm256_set1_pd(1.781477937);
   const __m256d b4 = _mm256_set1_pd(-1.821255978);
   const __m256d b5 = _mm256_set1_pd(1.330274429);
   const __m256d rsqrt_2pi = _mm256_set1_pd(0.3989422804014327);
   
   const __m256d z = _mm256_abs_pd(x); /////  std::fabs(x);
   
   const __m256d denom_t = _mm256_fmadd_pd(a, z, _mm256_set1_pd(1.0));
   
   const __m256d t = _mm256_div_pd(_mm256_set1_pd(1.0), denom_t);  //// double t = 1.0 / (1.0 + a * z);
   const __m256d t_2 = _mm256_mul_pd(t, t);
   const __m256d t_3 = _mm256_mul_pd(t_2, t);
   const __m256d t_4 = _mm256_mul_pd(t_2, t_2);
   const __m256d t_5 = _mm256_mul_pd(t_3, t_2);
   
   /////  double poly = b1 * t     +   b2 * t * t    +     b3 * t * t * t   +   b4 * t * t * t * t      +       b5 * t * t * t * t * t;
   const __m256d poly_term_1 = _mm256_mul_pd(b1, t);
   const __m256d poly_term_2 = _mm256_mul_pd(b2, t_2);
   const __m256d poly_term_3 = _mm256_mul_pd(b3, t_3);
   const __m256d poly_term_4 = _mm256_mul_pd(b4, t_4);
   const __m256d poly_term_5 = _mm256_mul_pd(b5, t_5);
   
   const __m256d poly = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(poly_term_1, poly_term_2),  _mm256_add_pd(poly_term_3, poly_term_4)), poly_term_5);
   
   const __m256d is_x_gr_0 = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ);
   
   const __m256d exp_stuff = fast_exp_1_AVX2(_mm256_mul_pd(_mm256_set1_pd(-0.50), _mm256_mul_pd(z, z)) ); 
   const __m256d res = _mm256_mul_pd(_mm256_mul_pd(rsqrt_2pi, exp_stuff), poly);
   const __m256d one_m_res = _mm256_sub_pd(_mm256_set1_pd(1.0), res);
   
   return _mm256_blendv_pd(    res,  //// if x < 0
                               one_m_res, //// if x > 0
                               is_x_gr_0); 
   
 }



 


inline __m256d fast_Phi_AVX2(const __m256d x) {
  
  
  const __m256d Phi_upper_bound =  _mm256_set1_pd(8.25);
  const __m256d Phi_lower_bound =  _mm256_set1_pd(-37.5);
  
  const __m256d is_x_gr_lower = _mm256_cmp_pd(x, Phi_lower_bound, _CMP_GT_OQ); // true where x > -37.5 (in range)
  const __m256d is_x_lt_upper = _mm256_cmp_pd(x, Phi_upper_bound, _CMP_LT_OQ); // true where x < 8.25 (in range)
  const __m256d is_x_in_range = _mm256_and_pd(is_x_gr_lower, is_x_lt_upper);
  
  // Calculate the main Phi function for in-range values
  __m256d result = _mm256_blendv_pd(
          _mm256_blendv_pd( _mm256_set1_pd(1.0),    // if x NOT in range and x NOT < -37.5 i.e. x must be > 8.25
                            _mm256_setzero_pd(), // if x NOT in range and x < -37.5
                            _mm256_cmp_pd(x, Phi_lower_bound, _CMP_LT_OQ)),  
                            fast_Phi_wo_checks_AVX2(x), // if x is in range
                            is_x_in_range
  );
  
  // Clamp results between 0 and 1
  result = _mm256_min_pd(_mm256_set1_pd(1.0), _mm256_max_pd(_mm256_setzero_pd(), result));
  
  return result;
  
}












 

///////////////////  fns - inv_Phi functions   ----------------------------------------------------------------------------------------------------------------------------------------

 
 
 
 
 
 
 
 
 /// vectorised, AVX-256 version of Stan fn provided by Sean Pinkney:  https://github.com/stan-dev/math/issues/2555 
 
 
 inline __m256d fast_inv_Phi_wo_checks_case_2a_AVX2(const __m256d p,
                                                      __m256d r) { ///  CASE 2(a): if abs(q) > 0.425  AND   if r <= 5.0 
   
   // std::cout << "calling  fast_Phi_wo_checks_case_2a_AVX256"   << "\n"; 
   //std::cout << "r = " <<  r  << "\n";
   
   r = _mm256_add_pd(r, _mm256_set1_pd(-1.60));
   
   __m256d numerator = _mm256_set1_pd(0.00077454501427834140764);
   numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(0.0227238449892691845833));
   numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(0.24178072517745061177));
   numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(1.27045825245236838258));
   numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(3.64784832476320460504));
   numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(5.7694972214606914055));
   numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(4.6303378461565452959));
   numerator = _mm256_fmadd_pd(r, numerator, _mm256_set1_pd(1.42343711074968357734)); 
   
   __m256d  denominator = _mm256_set1_pd(0.00000000105075007164441684324);
   denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.0005475938084995344946));
   denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.0151986665636164571966));
   denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.14810397642748007459));
   denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(0.68976733498510000455));
   denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(1.6763848301838038494));
   denominator = _mm256_fmadd_pd(r,  denominator, _mm256_set1_pd(2.05319162663775882187));
   denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(1.0));
   
   const __m256d val = _mm256_div_pd(numerator, denominator);
   
   return val;
   
 }  


inline __m256d fast_inv_Phi_wo_checks_case_2b_AVX2(const __m256d p,
                                                         __m256d r) { ///  CASE 2(a): if abs(q) > 0.425  AND   if r > 5.0 
  
  // std::cout << "calling  fast_Phi_wo_checks_case_2b_AVX256"   << "\n"; 
  //std::cout << "r = " <<  r  << "\n";
  
  r = _mm256_add_pd(r, _mm256_set1_pd(-5.0));
  
  __m256d numerator =  _mm256_set1_pd(0.000000201033439929228813265);
  numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.0000271155556874348757815));
  numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.0012426609473880784386));
  numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.026532189526576123093));
  numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(0.29656057182850489123));
  numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(1.7848265399172913358));
  numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(5.4637849111641143699));
  numerator = _mm256_fmadd_pd(r, numerator,  _mm256_set1_pd(6.6579046435011037772));
  
  __m256d denominator =  _mm256_set1_pd(0.00000000000000204426310338993978564);
  denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.00000014215117583164458887));
  denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.000018463183175100546818));
  denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.0007868691311456132591));
  denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.0148753612908506148525));
  denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.13692988092273580531)); 
  denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(0.59983220655588793769));
  denominator = _mm256_fmadd_pd(r, denominator,  _mm256_set1_pd(1.0));
  
  const __m256d val = _mm256_div_pd(numerator, denominator);
  
  return val;
  
  
} 
 


inline __m256d fast_inv_Phi_wo_checks_case_2_AVX2(const __m256d p,
                                                  const __m256d q) {  /// CASE 2 (i.e. one of case 2(a) or 2(b) depending on value of r)
  
  const __m256d is_q_gr_0 = _mm256_cmp_pd(q, _mm256_setzero_pd(), _CMP_GT_OQ);
   
  ////   compute r 
  __m256d r_if_q_gr_0 =  _mm256_sub_pd(_mm256_set1_pd(1.0), p);
  __m256d r_if_q_lt_0 = p;
  __m256d r = _mm256_blendv_pd(r_if_q_gr_0, r_if_q_lt_0, is_q_gr_0);
  
  r = _mm256_sqrt_pd(_mm256_sub_pd(_mm256_setzero_pd(), fast_log_1_AVX2(r)));
  
  /// then call either case 2(a) fn (if r <= 5.0) or case 2(b) fn (if r > 5.0)  
  const __m256d is_r_gr_5 = _mm256_cmp_pd(r, _mm256_set1_pd(5.0), _CMP_GT_OQ);
  __m256d val =  _mm256_blendv_pd(     fast_inv_Phi_wo_checks_case_2b_AVX2(p, r),  /// if r > 5.0
                                       fast_inv_Phi_wo_checks_case_2a_AVX2(p, r),
                                       is_r_gr_5);
  
  // Flip the sign if q is negative
  const __m256d is_q_lt_zero = _mm256_cmp_pd(q, _mm256_setzero_pd(), _CMP_LT_OQ); // _mm256_cmplt_pd_mask(q, _mm256_set1_pd(0.0));
  
  val = _mm256_blendv_pd(    _mm256_sub_pd(_mm256_setzero_pd(), val),   //// if q < 0
                             val,
                             is_q_lt_zero); //// if   q => 0 
  
  return val;
  
  
} 




inline __m256d fast_inv_Phi_wo_checks_case_1_AVX2(const __m256d p,
                                                  const __m256d q) { ///  CASE 1: if abs(q) <= 0.425
  
  
  const __m256d q_sq = _mm256_mul_pd(q, q);
  const __m256d r = _mm256_sub_pd(_mm256_set1_pd(0.180625), q_sq);
  
  __m256d numerator = _mm256_fmadd_pd(r, _mm256_set1_pd(2509.0809287301226727), _mm256_set1_pd(33430.575583588128105));
  numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(67265.770927008700853));
  numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(45921.953931549871457));
  numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(13731.693765509461125));
  numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(1971.5909503065514427)); 
  numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(133.14166789178437745));
  numerator = _mm256_fmadd_pd(numerator, r, _mm256_set1_pd(3.387132872796366608));
  
  __m256d denominator = _mm256_fmadd_pd(r, _mm256_set1_pd(5226.495278852854561), _mm256_set1_pd(28729.085735721942674));
  denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(39307.89580009271061));
  denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(21213.794301586595867));
  denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(5394.1960214247511077));
  denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(687.1870074920579083));
  denominator = _mm256_fmadd_pd(denominator, r, _mm256_set1_pd(42.313330701600911252)); 
  denominator = _mm256_fmadd_pd(denominator, r,  _mm256_set1_pd(1.0));
  
  __m256d val = _mm256_div_pd(numerator, denominator); 
  val = _mm256_mul_pd(q, val);
  
  return val;
  
}   
 

 

inline __m256d fast_inv_Phi_wo_checks_AVX2(const __m256d p) {
  
  const __m256d q = _mm256_sub_pd(p, _mm256_set1_pd(0.50));
  const __m256d is_q_le_threshold = _mm256_cmp_pd(_mm256_abs_pd(q), _mm256_set1_pd(0.425), _CMP_LE_OQ);
  
 
  if (_mm256_movemask_pd(is_q_le_threshold) == 0xF) {  // 0xF is binary 1111 (4 bits)  ///  if (_mm256_kortestc(is_q_le_threshold, is_q_le_threshold)) { 
    
    return fast_inv_Phi_wo_checks_case_2_AVX2(p, q);
    
  } else { 
    
    return fast_inv_Phi_wo_checks_case_1_AVX2(p, q);
    
  }
  
}  



 



#endif



 
 
 
#endif
 
 
 
 
 


