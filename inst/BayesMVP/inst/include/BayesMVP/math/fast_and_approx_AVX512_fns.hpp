
#pragma once 


#ifndef FAST_AND_APPROX_AVX512_AVX2_FNS_HPP
#define FAST_AND_APPROX_AVX512_AVX2_FNS_HPP


#include <immintrin.h>
#include <cmath>


#if defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)  /// if have AVX-512
 
//// -------------------------------------------------------------------------------------------------------------------------------------------------------------
 

// Simple test function that just multiplies vector by 2
inline __m512d  test_simple_AVX512 VECTORCALL(const __m512d x) {
     
       ALIGN64  __m512d const two = _mm512_set1_pd(2.0);
       ALIGN64  __m512d const res = _mm512_mul_pd(x, two);
       return res; 
 
}








// // =============================================================================================
// //  SELF-TEST  — call once from an Rcpp export or a throwaway main(); prints max errors.
// //  Reference: log Phi(x) = log(0.5*erfc(-x/sqrt2)) is exact to ~1e-15 for x >= -37.
// //  Passing:  err_logPhi <~ 1e-8 (limited by fast_log), err_dlogPhi <~ 1e-8,
// //            err_roundtrip (inv_Phi_from_log_p(log Phi(x), log Phi(-x)) - x) <~ 1e-12.
// // =============================================================================================
// inline void normal_tail_self_test() {
//       double err_logPhi = 0, err_dlogPhi = 0, err_roundtrip = 0, worst_x = 0;
//       for (double x = -37.0; x <= 8.0; x += 0.01) {
//           const double ref_logPhi  = std::log(0.5 * std::erfc(-x * M_SQRT1_2));
//           const double ref_dlogPhi = INV_SQRT_2PI * std::exp(-0.5 * x * x) / (0.5 * std::erfc(-x * M_SQRT1_2));
//           const double e1 = std::fabs(fast_log_Phi(x) - ref_logPhi) / std::max(1.0, std::fabs(ref_logPhi));
//           const double e2 = std::fabs(fast_dlog_Phi_dx(x) - ref_dlogPhi) / std::max(1e-300, std::fabs(ref_dlogPhi));
//           const double lp = fast_log_Phi(x), l1mp = fast_log_Phi(-x);
//           const double e3 = std::fabs(fast_inv_Phi_from_log_p(lp, l1mp) - x) / std::max(1.0, std::fabs(x));
//           if (e1 > err_logPhi)    err_logPhi = e1;
//           if (e2 > err_dlogPhi)   err_dlogPhi = e2;
//           if (e3 > err_roundtrip) { err_roundtrip = e3; worst_x = x; }
//       }
//       std::printf("normal_tail_self_test: max rel err  logPhi=%.3e  dlogPhi/dx=%.3e  inv_Phi roundtrip=%.3e (worst at x=%.2f)\n",
//                   err_logPhi, err_dlogPhi, err_roundtrip, worst_x);
//       //// the SIMD kernels must agree with the scalar ones to ~1e-12 — check a few lanes:
// #if defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)
//       {
//           alignas(64) double xs[8]  = {-30.0, -12.0, -6.0, -3.0, -2.6, -1.0, 0.5, 4.0};
//           alignas(64) double out[8];
//           _mm512_store_pd(out, fast_log_Phi_AVX512(_mm512_load_pd(xs)));
//           double e = 0; for (int i = 0; i < 8; ++i) e = std::max(e, std::fabs(out[i] - fast_log_Phi(xs[i])) / std::max(1.0, std::fabs(out[i])));
//           std::printf("  AVX512 vs scalar log_Phi: max rel diff %.3e\n", e);
//       }
// #endif
// }
// 
// 
// 


//// ------------------------------------------------------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE    __mmask8 is_NaN_mask(const __m512d x) {
       
       const __mmask8 mask_is_NaN =_mm512_cmp_pd_mask(x, x, _CMP_NEQ_OQ); // comparing value to itself checks for  NaN. is NaN then this = FALSE
       //  __mmask8 mask_is_NaN = f(mask_is_not_NaN); // there is no "NOT" or "complement" mask  !!!!
       
       return  mask_is_NaN;
 
}



ALWAYS_INLINE    __mmask8 is_not_NaN_mask(const __m512d x)  {
 
       const __mmask8 mask_is_not_NaN =_mm512_cmp_pd_mask(x, x, _CMP_EQ_OQ); // comparing value to itself checks for  NaN. is NaN then this = FALSE
       //  __mmask8 mask_is_NaN = f(mask_is_not_NaN); // there is no "NOT" or "complement" mask  !!!!
       
       return  mask_is_not_NaN;
 
}



// adapted from: https://stackoverflow.com/questions/30674291/how-to-check-inf-for-avx-intrinsic-m256
ALWAYS_INLINE    __mmask8 is_finite_mask(const __m512d x) {
  
        const __m512d INF = _mm512_set1_pd(std::numeric_limits<double>::infinity());
        const __m512d sign_bit = _mm512_set1_pd(-0.0);
       
        //x = _mm512_andnot_pd(sign_bit, x); // x  = NOT -0.0 AND x
        return _mm512_cmp_pd_mask(_mm512_andnot_pd(sign_bit, x), INF, _CMP_NEQ_OQ);
 
}



// adapted from: https://stackoverflow.com/questions/30674291/how-to-check-inf-for-avx-intrinsic-m256
ALWAYS_INLINE    __mmask8 is_infinity_mask(const __m512d x) {
  
        const __m512d INF = _mm512_set1_pd(std::numeric_limits<double>::infinity());
        const __m512d sign_bit = _mm512_set1_pd(-0.0);
       
        // x = _mm512_andnot_pd(sign_bit, x);
        return _mm512_cmp_pd_mask(_mm512_andnot_pd(sign_bit, x), INF, _CMP_EQ_OQ);
 
}




ALWAYS_INLINE    __mmask8 is_positive_mask(const __m512d x) {
 
        return _mm512_cmp_pd_mask(_mm512_setzero_pd(), x, _CMP_LT_OQ);
 
}



ALWAYS_INLINE    __mmask8 is_negative_mask(const __m512d x) {
 
        return _mm512_cmp_pd_mask(_mm512_setzero_pd(), x, _CMP_GT_OQ);
 
}




///////////////////  fns - exp   -----------------------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE    __m512d fast_ldexp( const __m512d AVX_a,
                                     const __m512i AVX_i) {
  
       return _mm512_castsi512_pd (_mm512_add_epi64 (_mm512_slli_epi64 (AVX_i, 52ULL), _mm512_castpd_si512 (AVX_a))); /* AVX_a = p * 2^AVX_i */
   
}




// ALWAYS_INLINE __m512d fast_ldexp_2( const __m512d AVX_a, 
//                                     const __m512i AVX_i) {
//   
//         const __m512i neg_mask = _mm512_srai_epi64(AVX_i, 63);
//         const __m512i abs_i = _mm512_sub_epi64(_mm512_xor_si512(AVX_i, neg_mask), neg_mask);
//         const __m512i threshold = _mm512_set1_epi64(1000);
//         const __mmask8 cmp_mask = _mm512_cmpgt_epi64_mask(abs_i, threshold); 
//         
//         if (cmp_mask) {
//           
//               const __m512i i1 = _mm512_xor_si512(_mm512_and_si512(neg_mask, _mm512_set1_epi64(-2000)), threshold);
//               const __m512i i2 = _mm512_sub_epi64(AVX_i, i1);
//               const __m512d mid = _mm512_castsi512_pd(_mm512_add_epi64(_mm512_slli_epi64(i1, 52), _mm512_castpd_si512(AVX_a)));
//               
//               // return _mm512_castsi512_pd(_mm512_add_epi64(_mm512_slli_epi64(i2, 52), _mm512_castpd_si512(mid)));
//               return  fast_ldexp(mid, i2); 
//           
//         }
//         
//         // return _mm512_castsi512_pd(_mm512_add_epi64(_mm512_slli_epi64(AVX_i, 52), _mm512_castpd_si512(AVX_a)));
//         return  fast_ldexp(AVX_a, AVX_i); 
//   
// }




// Adapted from: https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
// added   (optional) extra degree(s) for poly approx (oroginal float fn had 4 degrees) - using "minimaxApprox" R package to find coefficient terms
// R code:    minimaxApprox::minimaxApprox(fn = exp, lower = -0.346573590279972643113, upper = 0.346573590279972643113, degree = 5, basis ="Chebyshev")
ALWAYS_INLINE    __m512d fast_exp_1_wo_checks_AVX512(const __m512d x)  {
  
        const __m512d exp_l2e = _mm512_set1_pd (1.442695040888963387); /* log2(e) */
        const __m512d exp_l2h = _mm512_set1_pd (-0.693145751953125); /* -log(2)_hi, exactly representable */
        const __m512d exp_l2l = _mm512_set1_pd (-1.42860682030941723212e-6); /* -log(2)_lo */
        
        // /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
        const __m512d exp_c0 =     _mm512_set1_pd(0.00000276479776161191821278);
        const __m512d exp_c1 =     _mm512_set1_pd(0.0000248844480527491290235);
        const __m512d exp_c2 =     _mm512_set1_pd(0.000198411488032534342194);
        const __m512d exp_c3 =     _mm512_set1_pd(0.00138888017711994078175);
        const __m512d exp_c4 =     _mm512_set1_pd(0.00833333340524595143906);
        const __m512d exp_c5 =     _mm512_set1_pd(0.0416666670404215802592);
        const __m512d exp_c6 =     _mm512_set1_pd(0.166666666664891632843);
        const __m512d exp_c7 =     _mm512_set1_pd(0.499999999994389376923);
        const __m512d exp_c8 =     _mm512_set1_pd(1.00000000000001221245);
        const __m512d exp_c9 =     _mm512_set1_pd(1.00000000000001332268);
    
        const __m512d input  = x;
    
        /* exp(x) = 2^i * e^f; i = rint (log2(e) * a), f = a - log(2) * i */
        const __m512d t = _mm512_mul_pd(x, exp_l2e);      /* t = log2(e) * a */
        const __m512i i = _mm512_cvtpd_epi64(t);       /* i = (int)rint(t) */
        const __m512d x_2 = _mm512_roundscale_pd(t, _MM_FROUND_TO_NEAREST_INT) ; // ((0<<4)| _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC|_MM_FROUND_NO_EXC));
        const __m512d f = _mm512_fmadd_pd(x_2, exp_l2l, _mm512_fmadd_pd (x_2, exp_l2h, input));  /* a - log(2)_hi * r */    /* f = a - log(2)_hi * r - log(2)_lo * r */
        
        /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
        // polynomial approximation
        __m512d p = exp_c0;
        p = _mm512_fmadd_pd(p, f, exp_c1);
        p = _mm512_fmadd_pd(p, f, exp_c2);
        p = _mm512_fmadd_pd(p, f, exp_c3);
        p = _mm512_fmadd_pd(p, f, exp_c4);
        p = _mm512_fmadd_pd(p, f, exp_c5);
        p = _mm512_fmadd_pd(p, f, exp_c6);
        p = _mm512_fmadd_pd(p, f, exp_c7);
        p = _mm512_fmadd_pd(p, f, exp_c8);
        p = _mm512_fmadd_pd(p, f, exp_c9);
        
        return  fast_ldexp(p, i) ;    /* exp(x) = 2^i * p */
      
}




//// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// Keep exceptional-lane libm calls out of the inlined SIMD hot path.
__attribute__((noinline, cold)) inline __m512d fast_exp_exceptional_lanes_AVX512(const __m512d input, const __m512d result, const unsigned ordinary_mask) {
    alignas(64) double input_values[8], output_values[8];
    _mm512_store_pd(input_values, input);
    _mm512_store_pd(output_values, result);
    for (int lane = 0; lane < 8; ++lane) {
        if (!(ordinary_mask & (1u << lane))) output_values[lane] = std::exp(input_values[lane]);
    }
    return _mm512_load_pd(output_values);
}

ALWAYS_INLINE    __m512d fast_exp_1_AVX512(const __m512d a) {

        // 708.4 bounds the exponent-bit shortcut, NOT the domain of finite exp().
        // Preserve the ordinary SIMD path; only exceptional lanes need libm's full exponent range.
        const __mmask8 ordinary_lanes = _mm512_cmp_pd_mask(_mm512_abs_pd(a), _mm512_set1_pd(708.4), _CMP_LT_OQ);
        const __m512d result = fast_exp_1_wo_checks_AVX512(a);
        if (__builtin_expect(ordinary_lanes != 0xff, 0)) return fast_exp_exceptional_lanes_AVX512(a, result, ordinary_lanes);
        return result;

}

// // see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// inline __m512d fast_exp_1_AVX512(const __m512d a) {
//   
//   return _mm512_mask_blend_pd(
//     _mm512_cmp_pd_mask(_mm512_abs_pd(a), exp_bound, _CMP_LT_OQ),
//     _mm512_mask_blend_pd(
//       is_not_NaN_mask(a),
//       _mm512_set1_pd(INFINITY - INFINITY),
//       _mm512_mask_blend_pd(
//         _mm512_cmp_pd_mask(a, zero, _CMP_LT_OQ),
//         pos_inf,
//         zero
//       ) 
//     ),
//     fast_exp_1_wo_checks_AVX512(a)
//   );
//   
// } 
 
















///////////////////  fns - log  (+ related e.g. log1m etc) ----------------------------------------------------------------------------------------------------


////// ------  log fn  --------   https://stackoverflow.com/a/65537754/9007125 // vectorized version of the answer by njuffa
ALWAYS_INLINE       __m512d fast_log_1_wo_checks_AVX512(const __m512d a ) {
  
          const __m512i log_i1 =   _mm512_set1_epi64(0x3fe5555555555555);
          const __m512i log_i2 =   _mm512_set1_epi64(0xFFF0000000000000);
          
          const __m512d log_c1 =   _mm512_set1_pd(0.000000000000000222044604925031308085);
          const __m512d log_c2 =   _mm512_set1_pd(-0.13031005859375);
          const __m512d log_c3 =   _mm512_set1_pd(0.140869140625);
          const __m512d log_c4 =   _mm512_set1_pd(-0.121483512222766876221);           
          const __m512d log_c5 =   _mm512_set1_pd(0.139814853668212890625);
          const __m512d log_c6 =   _mm512_set1_pd(-0.166846126317977905273);
          const __m512d log_c7 =   _mm512_set1_pd(0.200120344758033752441);
          const __m512d log_c8 =   _mm512_set1_pd(-0.249996200203895568848);
          const __m512d log_c9 =   _mm512_set1_pd(0.333331972360610961914);
          const __m512d log_c10 =  _mm512_set1_pd(-0.5);
          const __m512d log_c11 =  _mm512_set1_pd(0.693147182464599609375);
          
          const __m512i aInt = _mm512_castpd_si512(a);
  
          const __m512i e = _mm512_and_epi64( _mm512_sub_epi64(aInt, log_i1),  log_i2); //    e = (__double_as_int (a) - i1 )    &   i2   ;
          const __m512d i = _mm512_fmadd_pd ( _mm512_cvtepi64_pd(e), log_c1, _mm512_setzero_pd()); // 0x1.0p-52
          const __m512d m = _mm512_sub_pd(_mm512_castsi512_pd( _mm512_sub_epi64(aInt, e)), _mm512_set1_pd(1.0)) ; 
          //   m = __int_as_double (__double_as_int (a) - e);  
          //  m = _mm256_sub_pd(m, one) ; // m - 1.0;  /* m in [2/3, 4/3] */
          const __m512d s = _mm512_mul_pd(m, m);  // m = _mm512_sub_pd(m, one) ; // m - 1.0;  /* m in [2/3, 4/3] */
  
          /* Compute log1p(m) for m in [-1/3, 1/3] */
          __m512d r =             log_c2;  // -0x1.0ae000p-3
          __m512d t =             log_c3;  //  0x1.208000p-3
          r = _mm512_fmadd_pd (r, s, log_c4); // -0x1.f198b2p-4
          t = _mm512_fmadd_pd (t, s, log_c5); //  0x1.1e5740p-3
          r = _mm512_fmadd_pd (r, s, log_c6); // -0x1.55b36cp-3
          t = _mm512_fmadd_pd (t, s, log_c7); //  0x1.99d8b2p-3
          r = _mm512_fmadd_pd (r, s, log_c8); // -0x1.fffe02p-3
          r = _mm512_fmadd_pd (t, m, r);
          r = _mm512_fmadd_pd (r, m, log_c9); //  0x1.5554fap-2
          r = _mm512_fmadd_pd (r, m, log_c10); // -0x1.000000p-1
          r = _mm512_fmadd_pd (r, s, m);
  
          return _mm512_fmadd_pd(i,  log_c11, r); //  0x1.62e430p-1 // log(2)

}
 
 
///// compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
ALWAYS_INLINE  __m512d fast_log1p_1_wo_checks_AVX512(const __m512d x)   {
  
          const __m512d small =  _mm512_set1_pd(1e-4);
          const __m512d minus_one_half  =  _mm512_set1_pd(-0.50);
  
          const __mmask8 is_abs_x_gr_1e_m_4 =      _mm512_cmp_pd_mask(_mm512_abs_pd(x), small, _CMP_GT_OQ); /// if abs(x) > small
          
          return       _mm512_mask_blend_pd(is_abs_x_gr_1e_m_4,
                                            _mm512_mul_pd(_mm512_fmadd_pd( minus_one_half, x,  _mm512_set1_pd(1.0)), x) , 
                                            fast_log_1_wo_checks_AVX512( _mm512_add_pd(x, _mm512_set1_pd(1.0)) ) ) ;  ////  if fabs(x) > small
          
}

// ///////  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
ALWAYS_INLINE      __m512d fast_log1m_1_wo_checks_AVX512(const __m512d x)   {
  
          const __m512d neg_x = _mm512_sub_pd(_mm512_setzero_pd(), x);
          return fast_log1p_1_wo_checks_AVX512(neg_x) ; // -x);
  
}


ALWAYS_INLINE      __m512d fast_log1p_exp_1_wo_checks_AVX512(const __m512d x)   {
  
          const __m512d neg_x =  _mm512_sub_pd(_mm512_setzero_pd(), x);
          const __mmask8 is_x_gr_0_mask =      _mm512_cmp_pd_mask(x,  _mm512_setzero_pd(), _CMP_GT_OQ);
          
          return       _mm512_mask_blend_pd(is_x_gr_0_mask,
                                            fast_log1p_1_wo_checks_AVX512(fast_exp_1_wo_checks_AVX512(x)),  /// if x < 0
                                            _mm512_add_pd(x, fast_log1p_1_wo_checks_AVX512(fast_exp_1_wo_checks_AVX512(neg_x)))); //// if x > 0 
  
}






//////https://stackoverflow.com/a/65537754/9007125
// Isolate rare subnormal/domain repairs from the normal-positive SIMD path.
__attribute__((noinline, cold)) inline __m512d fast_log_exceptional_lanes_AVX512(const __m512d input, const __m512d result, const unsigned ordinary_mask) {
    alignas(64) double input_values[8], output_values[8];
    _mm512_store_pd(input_values, input);
    _mm512_store_pd(output_values, result);
    for (int lane = 0; lane < 8; ++lane) {
        if (!(ordinary_mask & (1u << lane))) output_values[lane] = std::log(input_values[lane]);
    }
    return _mm512_load_pd(output_values);
}

ALWAYS_INLINE      __m512d fast_log_1_AVX512(const __m512d a ) {

        // The mantissa/exponent shortcut assumes a positive NORMAL double.
        // A single unsigned interval test covers positive normal doubles, excluding infinity/NaN.
        const __mmask8 ordinary_lanes = _mm512_cmp_epu64_mask(
            _mm512_sub_epi64(_mm512_castpd_si512(a), _mm512_set1_epi64(0x0010000000000000)),
            _mm512_set1_epi64(0x7fe0000000000000), _MM_CMPINT_LT);
        const __m512d result = fast_log_1_wo_checks_AVX512(a);
        if (__builtin_expect(ordinary_lanes != 0xff, 0)) return fast_log_exceptional_lanes_AVX512(a, result, ordinary_lanes);
        return result;
}


///////  compute log(1+x) without losing precision for small values of x.
/////see: https://www.johndcook.com/cpp_log_one_plus_x.html 
ALWAYS_INLINE      __m512d fast_log1p_1_AVX512(const __m512d x)   {
  
  const __m512d minus_one  =  _mm512_set1_pd(-1.0);
  const __m512d minus_one_half  =  _mm512_set1_pd(-0.50);
  const __m512d small =  _mm512_set1_pd(1e-4);
  
  const __mmask8 is_x_le_or_eq_to_m1_mask =     _mm512_cmp_pd_mask(x,  minus_one, _CMP_LE_OQ); /// if x <= -1.0
  const __mmask8 is_abs_x_gr_1e_m_4_mask  =     _mm512_cmp_pd_mask(_mm512_abs_pd(x), small, _CMP_GT_OQ);  //// if abs(x) > small
  
  return  _mm512_mask_blend_pd(is_x_le_or_eq_to_m1_mask,              // is x <= -1? (i.e., is x+1 <= 0?)
                               _mm512_mask_blend_pd(is_abs_x_gr_1e_m_4_mask, // if "is_x_le_or_eq_to_m1_mask" is FALSE (i.e., x+1 > 0)
                                                    _mm512_mul_pd(_mm512_fmadd_pd(minus_one_half, x,  _mm512_set1_pd(1.0)), x),
                                                    fast_log_1_AVX512(_mm512_add_pd(x,  _mm512_set1_pd(1.0)))), // if x > 1e-4 - evaluate using normal log fn!
                               fast_log_1_AVX512(_mm512_add_pd(x,  _mm512_set1_pd(1.0)))) ;  // if FALSE (i.e., x+1  > 0 ) then still pass onto normal log fn as this will handle NaNs etc!
  
}

// //  compute log(1+x) without losing precision for small values of x.
// //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
ALWAYS_INLINE     __m512d fast_log1m_1_AVX512(const __m512d x)   {
   
  const __m512d neg_x =  _mm512_sub_pd(_mm512_setzero_pd(), x);
  return fast_log1p_1_AVX512(neg_x) ;  
  
} 


ALWAYS_INLINE      __m512d fast_log1p_exp_1_AVX512(const  __m512d x)   {
   
  const __m512d neg_x =  _mm512_sub_pd(_mm512_setzero_pd(), x);
  const __mmask8 is_x_gr_0_mask =      _mm512_cmp_pd_mask(x,  _mm512_setzero_pd(), _CMP_GT_OQ);
   
  return       _mm512_mask_blend_pd(is_x_gr_0_mask, /// if x > 0
                                    fast_log1p_1_AVX512(fast_exp_1_AVX512(x)), /// if x < 0  ----log1p(exp(x))
                                    _mm512_add_pd(x, fast_log1p_1_AVX512(fast_exp_1_AVX512(neg_x)))); /// if x > 0 ---- x + log1p(exp(-x))
   
}



// //  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
// inline   double fast_log1p_exp_1(const double x)   {
//   
//   if (x > 0.0) return x + fast_log1p_1(fast_exp_1(-x));
//   return fast_log1p_1(fast_exp_1(x));
//   
// }


 

 

ALWAYS_INLINE     __m512d fast_logit_wo_checks_AVX512(const __m512d x)   {
   
   const __m512d log_x = fast_log_1_AVX512(x);
   const __m512d log_1m_x =  fast_log1m_1_AVX512(x);
   return  log_x - log_1m_x;
   
} 



ALWAYS_INLINE     __m512d fast_logit_AVX512(const __m512d x)   {
   
   const __m512d log_x = fast_log_1_AVX512(x);
   const __m512d log_1m_x =  fast_log1m_1_AVX512(x);
   return  log_x - log_1m_x;
   
} 
 












 







///////////////////  fns - inv_logit   ----------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE     __m512d  fast_inv_logit_for_x_pos_AVX512(const __m512d x )  {

      const __m512d  exp_m_x =  fast_exp_1_AVX512(_mm512_sub_pd(_mm512_setzero_pd(), x)) ;
      return    _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_add_pd(_mm512_set1_pd(1.0), exp_m_x))  ;
          
}




ALWAYS_INLINE     __m512d  fast_inv_logit_for_x_neg_AVX512(const __m512d x )  {
  
      const __m512d  log_eps  =   _mm512_set1_pd(-18.420680743952367);
      const __m512d exp_x =  fast_exp_1_AVX512(x) ;
      
      return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  log_eps, _CMP_GT_OQ),
                                  exp_x,
                                  _mm512_div_pd(exp_x, _mm512_add_pd(_mm512_set1_pd(1.0), exp_x)));
  
}




ALWAYS_INLINE     __m512d  fast_inv_logit_AVX512(const __m512d x )  {
  
  __m512d result =   _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  _mm512_setzero_pd(), _CMP_GE_OQ),
                                   fast_inv_logit_for_x_neg_AVX512(x),
                                   fast_inv_logit_for_x_pos_AVX512(x));
  
  return  _mm512_min_pd(_mm512_set1_pd(1.0), _mm512_max_pd(_mm512_setzero_pd(), result));
  
}

 






 

 






ALWAYS_INLINE     __m512d  fast_inv_logit_for_x_pos_wo_checks_AVX512(const __m512d x )  {
  
   const __m512d  exp_m_x =  fast_exp_1_wo_checks_AVX512(_mm512_sub_pd(_mm512_setzero_pd(), x)) ;
   return    _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_add_pd(_mm512_set1_pd(1.0), exp_m_x))  ;
   
}




ALWAYS_INLINE      __m512d  fast_inv_logit_for_x_neg_wo_checks_AVX512(const __m512d x )  {
  
  const __m512d  log_eps  =   _mm512_set1_pd(-18.420680743952367);
  const __m512d  exp_x =  fast_exp_1_wo_checks_AVX512(x) ;
  
  return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  log_eps, _CMP_GT_OQ),
                              exp_x,
                              _mm512_div_pd(exp_x, _mm512_add_pd(_mm512_set1_pd(1.0), exp_x)));
  
}




ALWAYS_INLINE     __m512d  fast_inv_logit_wo_checks_AVX512(const __m512d x )  {
  
  __m512d result =   _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  _mm512_setzero_pd(), _CMP_GT_OQ),
                                  fast_inv_logit_for_x_neg_wo_checks_AVX512(x),
                                  fast_inv_logit_for_x_pos_wo_checks_AVX512(x));
  
  return  _mm512_min_pd(_mm512_set1_pd(1.0), _mm512_max_pd(_mm512_setzero_pd(), result));
  
}







 

 




///////////////////  fns - log_inv_logit   ----------------------------------------------------------------------------------------------------------------

 

////////////


// inline     __m512d  fast_log_inv_logit_for_x_pos_AVX512(const __m512d x )  {
//   
//            const __m512d   m_x = _mm512_sub_pd(zero, x);
//            return   _mm512_sub_pd(zero, fast_log1p_exp_1_AVX512(fast_exp_1_AVX512(m_x))); ///  - log1p(exp(-x))
//            
// }
// 
// inline      __m512d  fast_log_inv_logit_for_x_neg_AVX512(const __m512d x )  {
//   
//           return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  log_eps, _CMP_GT_OQ), /// if x > log_eps = -18.420680743952367
//                                       x,  /// if x < log_eps
//                                       _mm512_sub_pd(x,   fast_log1p_1_AVX512(fast_exp_1_AVX512(x))) ); /// if x > log_eps = -18.420680743952367
//   
// }
// 
// inline      __m512d  fast_log_inv_logit_AVX512(const __m512d x )  {
//   
//           return  _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  zero, _CMP_GE_OQ),  // x > 0
//                                        fast_log_inv_logit_for_x_neg_AVX512(x),   /// x < 0
//                                        fast_log_inv_logit_for_x_pos_AVX512(x));  // x > 0
//   
// }


// For positive x: log(1/(1 + exp(-x))) = -log(1 + exp(-x))
inline __m512d fast_log_inv_logit_for_x_pos_AVX512(const __m512d x) {
  
      const __m512d m_x = _mm512_sub_pd(_mm512_setzero_pd(), x);  // -x
      const __m512d exp_m_x = fast_exp_1_AVX512(m_x);             // exp(-x)
      const __m512d log_sum = fast_log1p_1_AVX512(exp_m_x);       // log(1 + exp(-x))
      return _mm512_sub_pd(_mm512_setzero_pd(), log_sum);         // -log(1 + exp(-x))
  
}




// For negative x: log(exp(x)/(1 + exp(x))) = x - log(1 + exp(x))
inline __m512d fast_log_inv_logit_for_x_neg_AVX512(const __m512d x) {
  
      const __m512d log_eps = _mm512_set1_pd(-18.420680743952367);
      const __m512d exp_x = fast_exp_1_AVX512(x);                 // exp(x)
      const __m512d log_sum = fast_log1p_1_AVX512(exp_x);        // log(1 + exp(x))
      const __m512d result = _mm512_sub_pd(x, log_sum);          // x - log(1 + exp(x))
      
      // Blend based on x > log_eps 
      __mmask8 mask = _mm512_cmp_pd_mask(x, log_eps, _CMP_GT_OQ);
      return _mm512_mask_blend_pd(mask,
                                  x,        // if x <= log_eps
                                  result);  // if x > log_eps
  
}




inline __m512d fast_log_inv_logit_AVX512(const __m512d x) {
  
      __mmask8 is_pos = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_GE_OQ);
      
      return _mm512_mask_blend_pd(is_pos,
                                  fast_log_inv_logit_for_x_neg_AVX512(x),  // if x < 0 
                                  fast_log_inv_logit_for_x_pos_AVX512(x)); // if x >= 0
  
}












// 
// inline   double  fast_log_inv_logit_for_x_pos(const double x )  {
//   return    - fast_log1p_exp_1(-x);  // return    - fast_log1p_1(fast_exp_1(-x));
// }
// 
// inline   double  fast_log_inv_logit_for_x_neg(const double x )  {
//   const double log_eps = -18.420680743952367;
//   if (x > log_eps) return  x - fast_log1p_1(fast_exp_1(x));
//   return x;
// }
//  
// inline   double  fast_log_inv_logit(const double x )  {
//   if (x > 0.0) return  fast_log_inv_logit_for_x_pos(x);
//   return  fast_log_inv_logit_for_x_neg(x);
// } 
// 
//  
// 





 



////////////
inline     __m512d  fast_log_inv_logit_for_x_pos_wo_checks_AVX512(const __m512d x )  {
 
        return   _mm512_sub_pd(_mm512_setzero_pd(), fast_log1p_exp_1_wo_checks_AVX512((_mm512_sub_pd(_mm512_setzero_pd(), x)) ) );
  
}




inline     __m512d  fast_log_inv_logit_for_x_neg_wo_checks_AVX512(const __m512d x )  {
        
        const __m512d  log_eps  =   _mm512_set1_pd(-18.420680743952367);
        
        return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  log_eps, _CMP_GT_OQ), /// if x > log_eps = -18.420680743952367
                                    x,
                                    _mm512_sub_pd(x,   fast_log1p_1_wo_checks_AVX512(fast_exp_1_wo_checks_AVX512(x))));
        
}




inline     __m512d  fast_log_inv_logit_wo_checks_AVX512(const __m512d x )  {
  
        return  _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x,  _mm512_setzero_pd(), _CMP_GE_OQ),  // x > 0
                                     fast_log_inv_logit_for_x_neg_wo_checks_AVX512(x),
                                     fast_log_inv_logit_for_x_pos_wo_checks_AVX512(x));  // x > 0
  
}




/////////////////////  fns - Phi_approx   ----------------------------------------------------------------------------------------------------------------
inline      __m512d  fast_Phi_approx_wo_checks_AVX512( const __m512d x)  {
        
        const __m512d a =     _mm512_set1_pd(0.07056);
        const __m512d b =     _mm512_set1_pd(1.5976);
      
        const __m512d  x_sq =  _mm512_mul_pd(x, x);
        const __m512d  a_x_sq_plus_b = _mm512_fmadd_pd(a, x_sq, b);
        const __m512d  stuff_to_inv_logit =  _mm512_mul_pd(x, a_x_sq_plus_b);
        
        return    fast_inv_logit_wo_checks_AVX512(stuff_to_inv_logit);
  
}




inline     __m512d  fast_Phi_approx_AVX512(const __m512d x )  {
  
        const __m512d a =     _mm512_set1_pd(0.07056);
        const __m512d b =     _mm512_set1_pd(1.5976);
        
        const __m512d  x_sq =  _mm512_mul_pd(x, x);
        const __m512d  a_x_sq_plus_b = _mm512_fmadd_pd(a, x_sq, b);
        const __m512d  stuff_to_inv_logit =  _mm512_mul_pd(x, a_x_sq_plus_b);
        
        return fast_inv_logit_AVX512(stuff_to_inv_logit);
  
}




///////////////////  fns - inv_Phi_approx   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
inline     __m512d  fast_inv_Phi_approx_wo_checks_AVX512(const __m512d x )  {
  
        const __m512d  inv_Phi_approx_c1 = _mm512_set1_pd(-0.34176618822627863);
        const __m512d  inv_Phi_approx_c2 = _mm512_set1_pd(2.7472242570765295);
        const __m512d  one_third =  _mm512_set1_pd(0.33333333333333331483);
        
        const __m512d  m_logit_p = fast_log_1_wo_checks_AVX512( _mm512_sub_pd(_mm512_div_pd(_mm512_set1_pd(1.0),  x), _mm512_set1_pd(1.0) ) ); // logit first
        const __m512d  x_i =  _mm512_mul_pd(m_logit_p, inv_Phi_approx_c1) ; // -0.34176618822627863*m_logit_p;
        const __m512d  asinh_stuff_div_3 =  _mm512_mul_pd(  one_third,  fast_log_1_wo_checks_AVX512(  _mm512_add_pd(x_i, (  _mm512_sqrt_pd(  _mm512_fmadd_pd(x_i, x_i, _mm512_set1_pd(1.0)) ) )  ) ) ) ;          // now do arc_sinh part
        const __m512d  exp_x_i = fast_exp_1_wo_checks_AVX512(asinh_stuff_div_3);
        
        return _mm512_mul_pd(inv_Phi_approx_c2,   _mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(exp_x_i, exp_x_i), _mm512_set1_pd(1.0)), exp_x_i) );
  
}




inline     __m512d  fast_inv_Phi_approx_AVX512(const __m512d x )  {

        // Avoid 1 / p overflowing for valid subnormal probabilities. The usual path below is unchanged.
        const __mmask8 ordinary_lanes = _mm512_cmp_pd_mask(x, _mm512_set1_pd(std::numeric_limits<double>::min()), _CMP_GE_OQ)
                                     & _mm512_cmp_pd_mask(x, _mm512_set1_pd(1.0), _CMP_LT_OQ);
        if (ordinary_lanes != 0xff) {
            alignas(64) double input_values[8], output_values[8];
            _mm512_store_pd(input_values, x);
            const __m512d safe_input = _mm512_mask_blend_pd(ordinary_lanes, _mm512_set1_pd(0.5), x);
            _mm512_store_pd(output_values, fast_inv_Phi_approx_AVX512(safe_input));
            for (int lane = 0; lane < 8; ++lane) {
                if (!(ordinary_lanes & (1u << lane))) {
                    const double logit_probability = std::log(input_values[lane]) - std::log1p(-input_values[lane]);
                    output_values[lane] = 5.494448514153059 * std::sinh(std::asinh(0.34176618822627863 * logit_probability) / 3.0);
                }
            }
            return _mm512_load_pd(output_values);
        }
        
        const __m512d  inv_Phi_approx_c1 = _mm512_set1_pd(-0.34176618822627863);
        const __m512d  inv_Phi_approx_c2 = _mm512_set1_pd(2.7472242570765295);
        const __m512d  one_third =  _mm512_set1_pd(0.33333333333333331483);
        
        const __m512d m_logit_p = fast_log_1_AVX512(_mm512_div_pd(_mm512_sub_pd(_mm512_set1_pd(1.0), x), x)); // log((1-p)/p), stable near 1
        const __m512d x_i =  _mm512_mul_pd(m_logit_p, inv_Phi_approx_c1) ; // -0.34176618822627863*m_logit_p;
        const __m512d asinh_stuff_div_3 =  _mm512_mul_pd( one_third,  fast_log_1_AVX512(  _mm512_add_pd(x_i, (  _mm512_sqrt_pd(  _mm512_fmadd_pd(x_i, x_i, _mm512_set1_pd(1.0)) ) )  ) ) ) ;          // now do arc_sinh part
        const __m512d exp_x_i = fast_exp_1_AVX512(asinh_stuff_div_3);
        
        return    _mm512_mul_pd(inv_Phi_approx_c2,   _mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(exp_x_i, exp_x_i), _mm512_set1_pd(1.0)), exp_x_i) );
        
        // __mmask8 is_x_valid_prob =  _mm512_cmp_pd_mask(x,  zero, _CMP_GT_OQ) & _mm512_cmp_pd_mask(x,  one, _CMP_LT_OQ); // if x in (0, 1)
        // return   _mm512_mask_blend_pd(is_x_valid_prob,
        //                               _mm512_mask_blend_pd( (_mm512_cmp_pd_mask(x,  one, _CMP_GT_OQ) | _mm512_cmp_pd_mask(x,  zero, _CMP_GT_OQ)),  //  if x > 1 or x < 0 (i.e. = NaN / throw error)
        //                                                    _mm512_mask_blend_pd( _mm512_cmp_pd_mask(x,  zero, _CMP_EQ_OQ),
        //                                                                          _mm512_set1_pd(INFINITY), // if x = 1
        //                                                                          _mm512_set1_pd(-INFINITY)),  // if x = 0
        //                                                    _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN()) ),
        //                               _mm512_mul_pd(_mm512_set1_pd(2.7472242570765295),   _mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(exp_x_i, exp_x_i), one), exp_x_i) ));
  
}




inline      __m512d  fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX512(const __m512d logit_p )  {
        
        // const __m512d  inv_Phi_approx_c1 = _mm512_set1_pd(-0.34176618822627863);
        const __m512d  inv_Phi_approx_c1 = _mm512_set1_pd(0.34176618822627863);
        const __m512d  inv_Phi_approx_c2 = _mm512_set1_pd(2.7472242570765295);
        const __m512d  one_third =  _mm512_set1_pd(0.33333333333333331483);
       
        const __m512d  x_i =   _mm512_mul_pd(logit_p, inv_Phi_approx_c1) ;
        const __m512d  asinh_stuff_div_3 =  _mm512_mul_pd(one_third,  fast_log_1_wo_checks_AVX512(  _mm512_add_pd(x_i, (  _mm512_sqrt_pd(  _mm512_fmadd_pd(x_i, x_i, _mm512_set1_pd(1.0)) ) )  ) ) ) ;          // now do arc_sinh part
        const __m512d  exp_x_i = fast_exp_1_wo_checks_AVX512(asinh_stuff_div_3);
        
        return _mm512_mul_pd(inv_Phi_approx_c2,   _mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(exp_x_i, exp_x_i), _mm512_set1_pd(1.0)), exp_x_i) );
  
}




inline      __m512d  fast_inv_Phi_approx_from_logit_prob_AVX512(const __m512d logit_p )  {
        
        // const __m512d  inv_Phi_approx_c1 = _mm512_set1_pd(-0.34176618822627863);
        const __m512d  inv_Phi_approx_c1 = _mm512_set1_pd(0.34176618822627863);
        const __m512d  inv_Phi_approx_c2 = _mm512_set1_pd(2.7472242570765295);
        const __m512d  one_third =  _mm512_set1_pd(0.33333333333333331483);
       
        const __m512d  x_i =   _mm512_mul_pd(logit_p,  inv_Phi_approx_c1) ;
        const __m512d  asinh_stuff_div_3 =  _mm512_mul_pd(one_third,  fast_log_1_AVX512(  _mm512_add_pd(x_i, (  _mm512_sqrt_pd(  _mm512_fmadd_pd(x_i, x_i, _mm512_set1_pd(1.0)) ) )  ) ) ) ;          // now do arc_sinh part
        const __m512d  exp_x_i = fast_exp_1_AVX512(asinh_stuff_div_3);
        
        return _mm512_mul_pd(inv_Phi_approx_c2,   _mm512_div_pd(_mm512_sub_pd(_mm512_mul_pd(exp_x_i, exp_x_i), _mm512_set1_pd(1.0)), exp_x_i) );
  
}




///////////////////  fns - log_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------
inline      __m512d  fast_log_Phi_approx_wo_checks_AVX512(const __m512d x )  {
    
        const __m512d a =     _mm512_set1_pd(0.07056);
        const __m512d b =     _mm512_set1_pd(1.5976);
        
        const __m512d  x_sq =  _mm512_mul_pd(x, x);
        const __m512d  a_x_sq_plus_b = _mm512_fmadd_pd(a, x_sq, b);
        const __m512d  stuff_to_inv_logit =  _mm512_mul_pd(x, a_x_sq_plus_b);
        
        return fast_log_inv_logit_wo_checks_AVX512(stuff_to_inv_logit);
  
}




inline __m512d fast_log_Phi_approx_AVX512(const __m512d x) {
      
        const __m512d Phi_upper_bound =  _mm512_set1_pd(8.25);
        const __m512d Phi_lower_bound =  _mm512_set1_pd(-37.5);
        
        const __m512d a =     _mm512_set1_pd(0.07056);
        const __m512d b =     _mm512_set1_pd(1.5976);
    
        const __m512d x_sq = _mm512_mul_pd(x, x);
        const __m512d x_cubed = _mm512_mul_pd(x_sq, x);
        const __m512d result = _mm512_fmadd_pd(a, x_cubed, _mm512_mul_pd(b, x));
        
        return fast_log_inv_logit_AVX512(result);
  
}




//////////////////// ------------- tanh  --------------------------------------------------------------------------------------------------------------------------------------
ALWAYS_INLINE      __m512d    fast_tanh_AVX512( const  __m512d  x  )   {
        
        const __m512d  two =     _mm512_set1_pd(2.0);
        const __m512d  m_two =   _mm512_set1_pd(-2.0);
        
        return        _mm512_sub_pd( _mm512_div_pd(two,   _mm512_add_pd(_mm512_set1_pd(1.0), fast_exp_1_AVX512(  _mm512_mul_pd(x, m_two)   ) ) ), _mm512_set1_pd(1.0) ) ;
  
}




ALWAYS_INLINE       __m512d    fast_tanh_wo_checks_AVX512( const  __m512d x  )   {
  
        const __m512d  two =     _mm512_set1_pd(2.0);
        const __m512d  m_two =   _mm512_set1_pd(-2.0);
        
        return        _mm512_sub_pd( _mm512_div_pd(two,   _mm512_add_pd(_mm512_set1_pd(1.0), fast_exp_1_wo_checks_AVX512(  _mm512_mul_pd(x, m_two)   ) ) ), _mm512_set1_pd(1.0) ) ;
  
}




// from: https://stackoverflow.com/questions/57870896/writing-a-portable-sse-avx-version-of-stdcopysign
ALWAYS_INLINE     __m512d CopySign( const __m512d srcSign, 
                                    const __m512d srcValue) {
  
        const __m512d mask0 = _mm512_set1_pd(-0.);
        
        __m512d tmp0 = _mm512_and_pd(srcSign, mask0); // Extract the signed bit from srcSign
        __m512d tmp1 = _mm512_andnot_pd(mask0, srcValue); // Extract the number without sign of srcValue (abs(srcValue))
        
        return _mm512_or_pd(tmp0, tmp1);  // Merge signed bit with number and return
  
}
  

  
  
  
  
  
///////////////////  fns - error functions   ----------------------------------------------------------------------------------------------------------------------------------------

 
  




// 
// 
// static const __m512d erf_part_1_c0 =      _mm512_set1_pd(-5.6271698391213282e-18);
// static const __m512d erf_part_1_c1 =      _mm512_set1_pd(4.8565951797366214e-16);
// static const __m512d erf_part_1_c2 =      _mm512_set1_pd(-1.9912968283386570e-14);
// static const __m512d erf_part_1_c3 =      _mm512_set1_pd(5.1614612434698227e-13);
// static const __m512d erf_part_1_c4 =      _mm512_set1_pd(-9.4934693745934645e-12);
// static const __m512d erf_part_1_c5 =      _mm512_set1_pd(1.3183034417605052e-10);
// static const __m512d erf_part_1_c6 =      _mm512_set1_pd(-1.4354030030292210e-09);
// static const __m512d erf_part_1_c7 =      _mm512_set1_pd(1.2558925114413972e-08);
// static const __m512d erf_part_1_c8 =      _mm512_set1_pd(-8.9719702096303798e-08);
// static const __m512d erf_part_1_c9 =      _mm512_set1_pd(5.2832013824348913e-07);
// static const __m512d erf_part_1_c10 =     _mm512_set1_pd(-2.5730580226082933e-06);
// static const __m512d erf_part_1_c11 =     _mm512_set1_pd(1.0322052949676148e-05);
// static const __m512d erf_part_1_c12 =     _mm512_set1_pd(-3.3555264836700767e-05);
// static const __m512d erf_part_1_c13 =     _mm512_set1_pd(8.4667486930266041e-05);
// static const __m512d erf_part_1_c14 =     _mm512_set1_pd(-1.4570926486271945e-04);
// static const __m512d erf_part_1_c15 =     _mm512_set1_pd(7.1877160107954648e-05);
// static const __m512d erf_part_1_c16 =     _mm512_set1_pd(4.9486959714661590e-04);
// static const __m512d erf_part_1_c17 =     _mm512_set1_pd(-1.6221099717135270e-03);
// static const __m512d erf_part_1_c18 =     _mm512_set1_pd(1.6425707149019379e-04);
// static const __m512d erf_part_1_c19 =     _mm512_set1_pd(1.9148914196620660e-02);
// static const __m512d erf_part_1_c20 =     _mm512_set1_pd(-1.0277918343487560e-1);
// static const __m512d erf_part_1_c21 =     _mm512_set1_pd(-6.3661844223699315e-1);
// static const __m512d erf_part_1_c22 =     _mm512_set1_pd(-1.2837929411398119e-1);
// 
// 
// ///// see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function   // max ulp error = 0.97749 (USE_EXPM1 = 1); 1.05364 (USE_EXPM1 = 0)
// inline    __m512d fast_erf_wo_checks_part_1_upper_AVX512(const __m512d a,
//                                                          const __m512d t,
//                                                          const __m512d s) {
// 
//       __m512d r, u;
// 
//       r = _mm512_fmadd_pd (erf_part_1_c0, t, erf_part_1_c1); // -0x1.9f363ba3b515dp-58, 0x1.17f6b1d68f44bp-51
//       u = _mm512_fmadd_pd (erf_part_1_c2, t, erf_part_1_c3); // -0x1.66b85b7fbd01ap-46, 0x1.22907eebc22e0p-41
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c4, t, erf_part_1_c5); // -0x1.4e0591fd97592p-37, 0x1.21e5e2d8544d1p-33
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c6, t, erf_part_1_c7); // -0x1.8a8f81b7e0e84p-30, 0x1.af85793b93d2fp-27
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c8, t, erf_part_1_c9); // -0x1.8157db0edbfa8p-24, 0x1.1ba3c453738fdp-21
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c10, t, erf_part_1_c11); // -0x1.595999b7e922dp-19, 0x1.5a59c27b3b856p-17
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c12, t, erf_part_1_c13); // -0x1.197b61ee37123p-15, 0x1.631f0597f62b8p-14
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c14, t, erf_part_1_c15); // -0x1.319310dfb8583p-13, 0x1.2d798353da894p-14
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c16, t, erf_part_1_c17); //  0x1.037445e25d3e5p-11,-0x1.a939f51db8c06p-10
//       r = _mm512_fmadd_pd (r, s, u);
//       u = _mm512_fmadd_pd (erf_part_1_c18, t, erf_part_1_c19); //  0x1.5878d80188695p-13, 0x1.39bc5e0e9e09ap-6
//       r = _mm512_fmadd_pd (r, s, u);
//       r = _mm512_fmadd_pd (r, t, erf_part_1_c20); // -0x1.a4fbc8f8ff7dap-4
//       r = _mm512_fmadd_pd (r, t, erf_part_1_c21); // -0x1.45f2da3ae06f8p-1
//       r = _mm512_fmadd_pd (r, t, erf_part_1_c22); // -0x1.06ebb92d9ffa8p-3
//       r = _mm512_fmadd_pd (r, t, -t);
// 
//       r = _mm512_sub_pd(one, fast_exp_1_AVX512(r)); // r = 1.0 - fast_exp_1_wo_checks_AVX512(r);
// 
//       return CopySign(a, r);    //return copysign(r, a);
// 
// }
// 
// 
// static const __m512d erf_part_2_c0 =      _mm512_set1_pd(-7.7794684889591997e-10);
// static const __m512d erf_part_2_c1 =      _mm512_set1_pd(1.3710980398024347e-8);
// static const __m512d erf_part_2_c2 =      _mm512_set1_pd(-1.6206313758492398e-7);
// static const __m512d erf_part_2_c3 =      _mm512_set1_pd( 1.6447131571278227e-6);
// static const __m512d erf_part_2_c4 =      _mm512_set1_pd(-1.4924712302009488e-5);
// static const __m512d erf_part_2_c5 =      _mm512_set1_pd(1.2055293576900605e-4);
// static const __m512d erf_part_2_c6 =      _mm512_set1_pd(-8.5483259293144627e-4);
// static const __m512d erf_part_2_c7 =      _mm512_set1_pd(5.2239776061185055e-3);
// static const __m512d erf_part_2_c8 =      _mm512_set1_pd(-2.6866170643111514e-2);
// static const __m512d erf_part_2_c9 =      _mm512_set1_pd(1.1283791670944182e-1);
// static const __m512d erf_part_2_c10 =     _mm512_set1_pd(-3.7612638903183515e-1);
// static const __m512d erf_part_2_c11 =     _mm512_set1_pd(1.2837916709551256e-1);
// 
// 
// 
// // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function //   // max ulp error = 1.01912
// inline    __m512d fast_erf_wo_checks_part_2_lower_AVX512(const __m512d a,
//                                                          const __m512d t,
//                                                          const __m512d s) {
// 
//       __m512d r, u;
// 
// 
//       r =           erf_part_2_c0;                                   // -0x1.abae491c44131p-31
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c1); //  0x1.d71b0f1b10071p-27
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c2); // -0x1.5c0726f04dbc7p-23
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c3); //  0x1.b97fd3d9927cap-20
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c4); // -0x1.f4ca4d6f3e232p-17
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c5); //  0x1.f9a2baa8fedc2p-14
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c6); // -0x1.c02db03dd71bbp-11
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c7); //  0x1.565bccf92b31ep-8
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c8); // -0x1.b82ce311fa94bp-6
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c9); //  0x1.ce2f21a040d14p-4
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c10); // -0x1.812746b0379bcp-2
//       r = _mm512_fmadd_pd (r, s, erf_part_2_c11); //  0x1.06eba8214db68p-3
// 
//       return _mm512_fmadd_pd(r, a, a);
// 
// }
// 
// 
// 
// //// // Combined fast erf function
// ///// see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function
// inline __m512d fast_erf_wo_checks_AVX512(const __m512d a) {
// 
//   const __m512d t = _mm512_abs_pd(a);
//   const __m512d s = _mm512_mul_pd(a, a);
// 
//   return _mm512_mask_blend_pd(
//     _mm512_cmp_pd_mask(t, one, _CMP_GT_OQ),
//     fast_erf_wo_checks_part_2_lower_AVX512(a, t, s),
//     fast_erf_wo_checks_part_1_upper_AVX512(a, t, s)
//   );
// 
// }
// 
// 
// 
// 
// 
// 
// 
// 






///////////////////  fns - Phi functions   ----------------------------------------------------------------------------------------------------------------------------------------
 


  


  /// use AVX-512

// static const __m512d one_half =     _mm512_set1_pd(0.5);
// static const __m512d sqrt_2_recip = _mm512_set1_pd(0.707106781186547461715);
// 
// inline    __m512d fast_Phi_wo_checks_AVX512(const __m512d x) {
// 
//   return _mm512_mul_pd(one_half, _mm512_add_pd(one, fast_erf_wo_checks_AVX512(_mm512_mul_pd(x, sqrt_2_recip)) ) ) ;
// 
// }




//// ---- fast_Phi: relative-minimax rational approximation of the Mills ratio (replaces Abramowitz & Stegun 26.2.17):
////
////   Phi(-z) = phi(z) * R(z),   R(z) = Phi(-z) / phi(z) = sqrt(pi/2) * erfcx(z / sqrt(2)),   z = |x|,
////   R(z) ~= P6(z) / Q7(z)  on z in [0, 37.5],   max relative error 7.8e-12 (measured on a dense grid in mpmath, double coefficients),
////   so  Phi(-z) = exp(-z^2/2) * N6(z) / Q7(z)  with  N6 = P6 / sqrt(2 pi)  (the 1/sqrt(2 pi) of phi is folded into N6),
////   and Phi(x) = 1 - Phi(-x) for x > 0 (the same reflection as before).
////
//// Why: A&S 26.2.17 (Zelen & Severo 1964 / Hastings 1955 form, published) has absolute error <= 7.5e-8 but RELATIVE error 5e-5 at
//// x = -3, 8e-3 at -7.5 and 0.16 at -37.5, while every caller registers the exact density phi(x) as the gradient, so value and
//// gradient disagreed in the lower tail (and log(fast_Phi) was off by up to 0.17 nats). The rational form is relative-accurate for
//// every x, branch-free, and costs the same as A&S: one exp, one divide and 13 FMAs (A&S: 1 FMA + 9 multiplies + 4 adds).
////
//// Provenance: the Mills-ratio/rational form is standard (e.g. Cody 1969, Math. Comp. 23:631-637, uses rationals for erfc);
//// THESE coefficients were fitted with a relative-error Sanathanan-Koerner / Lawson
//// near-minimax fit in 40-digit mpmath, constrained to N6(0) = 0.5 exactly so that fast_Phi(0) = 0.5 with no jump at x = 0.
//// FAST_PHI_MILLS_Z_CAP caps only the rational's argument (the exp still sees the true z): it keeps N and Q finite for
//// |x| = inf / huge |x| (exp(-z^2/2) = 0 there, so the result is 0 / 1 exactly as before) and does not change any x in [-37.5, 8.25].
#ifndef BAYESMVP_FAST_PHI_MILLS_COEFFICIENTS_DEFINED
#define BAYESMVP_FAST_PHI_MILLS_COEFFICIENTS_DEFINED
static constexpr double FAST_PHI_MILLS_Z_CAP = 38.5;
static constexpr double FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[7] = {   //// N6(z) = sum_k c[k] z^k   (includes 1/sqrt(2 pi))
        0.5,
        0.5968126365007661,
        0.34755100079419643,
        0.12199983678527791,
        0.027010741025906854,
        0.0035954377698120227,
        0.0002302422023077616 };
static constexpr double FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[8] = {   //// Q7(z) = sum_k d[k] z^k
        1.0,
        1.9915098343653488,
        1.7840969388752417,
        0.9377097714656843,
        0.314821789667328,
        0.06828298351478591,
        0.009012426603260822,
        0.0005771316095331759 };
static_assert(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0] == 0.5, "fast_Phi Mills rational: N(0) must be exactly 0.5 so that fast_Phi(0) = 0.5");
static_assert(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0] == 1.0, "fast_Phi Mills rational: Q(0) must be exactly 1");
static_assert(FAST_PHI_MILLS_Z_CAP >= 37.5, "fast_Phi Mills rational: the cap must not bind inside fast_Phi's range [-37.5, 8.25]");
#endif




//// Phi(x) = exp(-z^2/2) * N(z) / Q(z) for x <= 0, 1 minus that for x > 0 (z = |x|): see the FAST_PHI_MILLS_* block above.
ALWAYS_INLINE __m512d fast_Phi_wo_checks_AVX512(__m512d x) {

        const __m512d z = _mm512_abs_pd(x); /////  std::fabs(x);
        //// NaN x: min returns its 2nd operand (the cap), and the NaN still propagates through the exp below.
        const __m512d z_rational = _mm512_min_pd(z, _mm512_set1_pd(FAST_PHI_MILLS_Z_CAP));

        __m512d numerator = _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[6]);
        numerator = _mm512_fmadd_pd(numerator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[5]));
        numerator = _mm512_fmadd_pd(numerator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[4]));
        numerator = _mm512_fmadd_pd(numerator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[3]));
        numerator = _mm512_fmadd_pd(numerator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[2]));
        numerator = _mm512_fmadd_pd(numerator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[1]));
        numerator = _mm512_fmadd_pd(numerator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0]));

        __m512d denominator = _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[7]);
        denominator = _mm512_fmadd_pd(denominator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[6]));
        denominator = _mm512_fmadd_pd(denominator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[5]));
        denominator = _mm512_fmadd_pd(denominator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[4]));
        denominator = _mm512_fmadd_pd(denominator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[3]));
        denominator = _mm512_fmadd_pd(denominator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[2]));
        denominator = _mm512_fmadd_pd(denominator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[1]));
        denominator = _mm512_fmadd_pd(denominator, z_rational, _mm512_set1_pd(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0]));

        const __mmask8 is_x_gr_0 = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_GT_OQ);

        const __m512d exp_stuff = fast_exp_1_AVX512(_mm512_mul_pd(_mm512_set1_pd(-0.50), _mm512_mul_pd(z, z)) );
        //// N / Q is formed BEFORE multiplying by the exp so the divide runs in parallel with fast_exp_1's polynomial instead of
        //// after it (measured: (exp * N) / Q put the divide latency on the critical path and cost ~6% on AVX2).
        const __m512d res = _mm512_mul_pd(exp_stuff, _mm512_div_pd(numerator, denominator));   //// Phi(-z) = exp(-z^2/2) * N(z) / Q(z)
        const __m512d one_m_res = _mm512_sub_pd(_mm512_set1_pd(1.0), res);

        return _mm512_mask_blend_pd(is_x_gr_0,
                                    res,  //// if x < 0
                                    one_m_res); //// if x > 0

}





// inline __m512d fast_Phi_AVX512(const __m512d x) {
//   
//         const __mmask8 is_x_gr_lower = _mm512_cmp_pd_mask(x, Phi_lower_bound, _CMP_GT_OQ); // true where x > -37.5 (in range)
//         const __mmask8 is_x_lt_upper = _mm512_cmp_pd_mask(x, Phi_upper_bound, _CMP_LT_OQ); // true where x < 8.25 (in range)
//         const __mmask8 is_x_in_range = is_x_gr_lower & is_x_lt_upper ; //// _mm512_kand(is_x_gr_lower, is_x_lt_upper);
//         
//         return _mm512_mask_blend_pd(
//                                       is_x_in_range,
//                                                       _mm512_mask_blend_pd( _mm512_cmp_pd_mask(x, Phi_lower_bound, _CMP_LT_OQ),  // if x NOT in range and x < -37.5
//                                                                             one,    // if x NOT in range and x NOT < -37.5 i.e. x must be > 8.25  
//                                                                             zero),    // if x NOT in range and x < -37.5
//                                       fast_Phi_wo_checks_AVX512(x)); /// if x is in range
//   
// }


ALWAYS_INLINE __m512d fast_Phi_AVX512(const __m512d x) {
      
      
      const __m512d Phi_upper_bound =  _mm512_set1_pd(8.25);
      const __m512d Phi_lower_bound =  _mm512_set1_pd(-37.5);
  
      const __mmask8 is_x_gr_lower = _mm512_cmp_pd_mask(x, Phi_lower_bound, _CMP_GE_OQ); // equality must not fall through to 1
      const __mmask8 is_x_lt_upper = _mm512_cmp_pd_mask(x, Phi_upper_bound, _CMP_LT_OQ); // true where x < 8.25 (in range)
      const __mmask8 is_x_in_range = is_x_gr_lower & is_x_lt_upper;
      
      // Calculate the main Phi function for in-range values
      __m512d result = _mm512_mask_blend_pd(
        is_x_in_range,
        _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x, Phi_lower_bound, _CMP_LT_OQ),
                             _mm512_set1_pd(1.0),    // if x NOT in range and x NOT < -37.5 i.e. x must be > 8.25
                             _mm512_setzero_pd()),  // if x NOT in range and x < -37.5
                             fast_Phi_wo_checks_AVX512(x) // if x is in range
      );
      
      // Clamp results between 0 and 1
      result = _mm512_min_pd(_mm512_set1_pd(1.0), _mm512_max_pd(_mm512_setzero_pd(), result));
      
      // Preserve NaN input rather than turning it into a valid probability through min/max.
      return _mm512_mask_mov_pd(result, _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q), x);
  
}












///////////////////  fns - inverse-error functions (for inv_Phi)  ----------------------------------------------------------------------------------------------------------------------------------------
//  compute inverse error functions with maximum error of 2.35793 ulp  // see: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
 

// 
// 
// static const __m512d inv_erf_part_1_c0 =     _mm512_set1_pd(3.03697567e-10);
// static const __m512d inv_erf_part_1_c1 =     _mm512_set1_pd(2.93243101e-8);
// static const __m512d inv_erf_part_1_c2 =     _mm512_set1_pd(1.22150334e-6);
// static const __m512d inv_erf_part_1_c3 =     _mm512_set1_pd(2.84108955e-5);
// static const __m512d inv_erf_part_1_c4 =     _mm512_set1_pd(3.93552968e-4);
// static const __m512d inv_erf_part_1_c5 =     _mm512_set1_pd(3.02698812e-3);
// static const __m512d inv_erf_part_1_c6 =     _mm512_set1_pd(4.83185798e-3);
// static const __m512d inv_erf_part_1_c7 =     _mm512_set1_pd(-2.64646143e-1);
// static const __m512d inv_erf_part_1_c8 =     _mm512_set1_pd(8.40016484e-1);
//   
//   
//   
//   
// // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function   // max ulp error = 0.97749 (USE_EXPM1 = 1); 1.05364 (USE_EXPM1 = 0)
// inline    __m512d fast_inv_erf_wo_checks_part_1_upper_AVX512(const __m512d a, 
//                                                              const __m512d t) {
//     
//     __m512d p =            inv_erf_part_1_c0; //  0x1.4deb44p-32
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c1); //  0x1.f7c9aep-26
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c2); //  0x1.47e512p-20
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c3); //  0x1.dca7dep-16
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c4); //  0x1.9cab92p-12
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c5); //  0x1.8cc0dep-9
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c6); //  0x1.3ca920p-8
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c7); // -0x1.0eff66p-2
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_1_c8); //  0x1.ae16a4p-1
//     
//     return  _mm512_mul_pd(a, p); // a * p;
//     
//   }
//   
//   static const __m512d inv_erf_part_2_c0 =     _mm512_set1_pd(5.43877832e-9);
//   static const __m512d inv_erf_part_2_c1 =     _mm512_set1_pd(1.43285448e-7);
//   static const __m512d inv_erf_part_2_c2 =     _mm512_set1_pd(1.22774793e-6);
//   static const __m512d inv_erf_part_2_c3 =     _mm512_set1_pd(1.12963626e-7);
//   static const __m512d inv_erf_part_2_c4 =     _mm512_set1_pd(-5.61530760e-5);
//   static const __m512d inv_erf_part_2_c5 =     _mm512_set1_pd(-1.47697632e-4);
//   static const __m512d inv_erf_part_2_c6 =     _mm512_set1_pd(2.31468678e-3);
//   static const __m512d inv_erf_part_2_c7 =     _mm512_set1_pd(1.15392581e-2);
//   static const __m512d inv_erf_part_2_c8 =     _mm512_set1_pd(-2.32015476e-1);
//   static const __m512d inv_erf_part_2_c9 =     _mm512_set1_pd(8.86226892e-1);
//   
//   // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function //   // max ulp error = 1.01912
//   inline    __m512d fast_inv_erf_wo_checks_part_2_lower_AVX512(const __m512d a,
//                                                                const __m512d t) {
//     
//     __m512d p =            inv_erf_part_2_c0; //  0x1.4deb44p-32
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c1); //  0x1.f7c9aep-26
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c2); //  0x1.47e512p-20
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c3); //  0x1.dca7dep-16
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c4); //  0x1.9cab92p-12
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c5); //  0x1.8cc0dep-9
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c6); //  0x1.3ca920p-8
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c7); // -0x1.0eff66p-2
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c8); //  0x1.ae16a4p-1
//     p = _mm512_fmadd_pd (p, t,  inv_erf_part_2_c9); //  0x1.ae16a4p-1
//     
//     return  _mm512_mul_pd(a, p); // a * p;
//   
// }
// 
// 
// 
// static const __m512d inv_erf_c0 =      _mm512_set1_pd(6.125);
// 
// // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function
// inline __m512d fast_inv_erf_wo_checks_AVX512(const __m512d a) {
//   
//       __m512d t = _mm512_fmadd_pd(a, _mm512_sub_pd(zero, a), one);
//       t = fast_log_1_AVX512(t);
//       
//       return _mm512_mask_blend_pd(
//         _mm512_cmp_pd_mask(_mm512_abs_pd(t), inv_erf_c0, _CMP_GT_OQ), 
//         fast_inv_erf_wo_checks_part_2_lower_AVX512(a, t),
//         fast_inv_erf_wo_checks_part_1_upper_AVX512(a, t));
//   
// }
//  
// 
// 


///////////////////  fns - inv_Phi functions   ----------------------------------------------------------------------------------------------------------------------------------------
 





// static const __m512d sqrt_2 =  _mm512_set1_pd(1.41421356237309514547);
// 
// inline    __m512d fast_inv_Phi_wo_checks_AVX512(const __m512d x) {
//   
//     return _mm512_mul_pd(sqrt_2, fast_inv_erf_wo_checks_AVX512( _mm512_sub_pd(_mm512_mul_pd(two, x), one )))  ;
//   
// }



/// vectorised, AVX-512 version of Stan fn provided by Sean Pinkney:  https://github.com/stan-dev/math/issues/2555 


ALWAYS_INLINE __m512d fast_inv_Phi_wo_checks_case_2a_AVX512(const __m512d p,
                                                     __m512d r) { ///  CASE 2(a): if abs(q) > 0.425  AND   if r <= 5.0 
  
 // std::cout << "calling  fast_Phi_wo_checks_case_2a_AVX512"   << "\n"; 
  //std::cout << "r = " <<  r  << "\n";
  
  r = _mm512_add_pd(r, _mm512_set1_pd(-1.60));
  
  __m512d numerator = _mm512_set1_pd(0.00077454501427834140764);
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(0.0227238449892691845833));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(0.24178072517745061177));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(1.27045825245236838258));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(3.64784832476320460504));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(5.7694972214606914055));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(4.6303378461565452959));
  numerator = _mm512_fmadd_pd(r, numerator, _mm512_set1_pd(1.42343711074968357734)); 
  
  __m512d  denominator = _mm512_set1_pd(0.00000000105075007164441684324);
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(0.0005475938084995344946));
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(0.0151986665636164571966));
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(0.14810397642748007459));
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(0.68976733498510000455));
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(1.6763848301838038494));
  denominator = _mm512_fmadd_pd(r,  denominator, _mm512_set1_pd(2.05319162663775882187));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(1.0));
  
  const __m512d val = _mm512_div_pd(numerator, denominator);
  
  return val;
  
}  


ALWAYS_INLINE __m512d fast_inv_Phi_wo_checks_case_2b_AVX512(const __m512d p,
                                                     __m512d r) { ///  CASE 2(a): if abs(q) > 0.425  AND   if r > 5.0 
  
 // std::cout << "calling  fast_Phi_wo_checks_case_2b_AVX512"   << "\n";
  //std::cout << "r = " <<  r  << "\n";
  
  r = _mm512_add_pd(r, _mm512_set1_pd(-5.0));
  
  __m512d numerator =  _mm512_set1_pd(0.000000201033439929228813265);
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(0.0000271155556874348757815));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(0.0012426609473880784386));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(0.026532189526576123093));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(0.29656057182850489123));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(1.7848265399172913358));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(5.4637849111641143699));
  numerator = _mm512_fmadd_pd(r, numerator,  _mm512_set1_pd(6.6579046435011037772));
  
  __m512d denominator =  _mm512_set1_pd(0.00000000000000204426310338993978564);
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.00000014215117583164458887));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.000018463183175100546818));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.0007868691311456132591));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.0148753612908506148525));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.13692988092273580531)); 
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(0.59983220655588793769));
  denominator = _mm512_fmadd_pd(r, denominator,  _mm512_set1_pd(1.0));
  
  const __m512d val = _mm512_div_pd(numerator, denominator);
  
  return val;
  
  
} 
 


// inline __m512d fast_inv_Phi_wo_checks_case_2_AVX512(const __m512d p,
//                                                     const __m512d q) {  /// CASE 2 (i.e. one of case 2(a) or 2(b) depending on value of r)
//   
//         const __mmask8 is_q_gr_0 = _mm512_cmp_pd_mask(q, _mm512_setzero_pd(), _CMP_GT_OQ);
//          
//         ////   compute r 
//         __m512d r_if_q_gr_0 =  _mm512_sub_pd(_mm512_set1_pd(1.0), p);
//         __m512d r_if_q_lt_0 = p;
//         __m512d r = _mm512_mask_blend_pd(is_q_gr_0, r_if_q_gr_0, r_if_q_lt_0);
//         r = _mm512_sqrt_pd(_mm512_sub_pd(_mm512_setzero_pd(), fast_log_1_AVX512(r)));
//         
//         /// then call either case 2(a) fn (if r <= 5.0) or case 2(b) fn (if r > 5.0)  
//         const __mmask8 is_r_gr_5 = _mm512_cmp_pd_mask(r, _mm512_set1_pd(5.0), _CMP_GT_OQ);
//         __m512d val =  _mm512_mask_blend_pd( is_r_gr_5, 
//                                              fast_inv_Phi_wo_checks_case_2b_AVX512(p, r),  /// if r > 5.0
//                                              fast_inv_Phi_wo_checks_case_2a_AVX512(p, r));
//         
//         // Flip the sign if q is negative
//         const __mmask8 is_q_lt_zero = _mm512_cmp_pd_mask(q, _mm512_setzero_pd(), _CMP_LT_OQ); // _mm512_cmplt_pd_mask(q, _mm512_set1_pd(0.0));
//         
//         val = _mm512_mask_blend_pd(is_q_lt_zero, 
//                                    _mm512_sub_pd(_mm512_setzero_pd(), val),   //// if q < 0
//                                    val); //// if   q => 0 
//         
//         return val;
//   
//   
// }



ALWAYS_INLINE __m512d fast_inv_Phi_wo_checks_case_2_AVX512( const __m512d p, 
                                                            const __m512d q) {
  
        const __m512d zero = _mm512_setzero_pd(), one = _mm512_set1_pd(1.0);
        const __mmask8 q_pos = _mm512_cmp_pd_mask(q, zero, _CMP_GT_OQ);
        __m512d r = _mm512_mask_blend_pd(q_pos, p, _mm512_sub_pd(one, p));        // q>0 -> 1-p, else p
        r = _mm512_sqrt_pd(_mm512_sub_pd(zero, fast_log_1_AVX512(r)));
        const __mmask8 r_gr_5 = _mm512_cmp_pd_mask(r, _mm512_set1_pd(5.0), _CMP_GT_OQ);
        __m512d val = fast_inv_Phi_wo_checks_case_2a_AVX512(p, r);                // r <= 5
        // No need to evaluate the extreme-tail polynomial unless a lane will actually use it.
        if (r_gr_5 != 0) val = _mm512_mask_blend_pd(r_gr_5, val, fast_inv_Phi_wo_checks_case_2b_AVX512(p, r));
        const __mmask8 q_neg = _mm512_cmp_pd_mask(q, zero, _CMP_LT_OQ);
        return _mm512_mask_blend_pd(q_neg, val, _mm512_sub_pd(zero, val));          // q<0 -> -val
  
}



ALWAYS_INLINE __m512d fast_inv_Phi_wo_checks_case_1_AVX512( const __m512d p,
                                                            const __m512d q) { ///  CASE 1: if abs(q) <= 0.425

        const __m512d q_sq = _mm512_mul_pd(q, q);
        const __m512d r = _mm512_sub_pd(_mm512_set1_pd(0.180625), q_sq);

        __m512d numerator = _mm512_fmadd_pd(r, _mm512_set1_pd(2509.0809287301226727), _mm512_set1_pd(33430.575583588128105));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(67265.770927008700853));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(45921.953931549871457));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(13731.693765509461125));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(1971.5909503065514427));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(133.14166789178437745));
        numerator = _mm512_fmadd_pd(numerator, r, _mm512_set1_pd(3.387132872796366608));

        __m512d denominator = _mm512_fmadd_pd(r, _mm512_set1_pd(5226.495278852854561), _mm512_set1_pd(28729.085735721942674));
        denominator = _mm512_fmadd_pd(denominator, r, _mm512_set1_pd(39307.89580009271061));
        denominator = _mm512_fmadd_pd(denominator, r, _mm512_set1_pd(21213.794301586595867));
        denominator = _mm512_fmadd_pd(denominator, r, _mm512_set1_pd(5394.1960214247511077));
        denominator = _mm512_fmadd_pd(denominator, r, _mm512_set1_pd(687.1870074920579083));
        denominator = _mm512_fmadd_pd(denominator, r, _mm512_set1_pd(42.313330701600911252));
        denominator = _mm512_fmadd_pd(denominator, r,  _mm512_set1_pd(1.0));

        __m512d val = _mm512_div_pd(numerator, denominator);
        val = _mm512_mul_pd(q, val);

        return val;

}


ALWAYS_INLINE __m512d fast_inv_Phi_wo_checks_AVX512( const __m512d p) {
  
        const __m512d q = _mm512_sub_pd(p, _mm512_set1_pd(0.5));
        const __mmask8 central = _mm512_cmp_pd_mask(_mm512_abs_pd(q), _mm512_set1_pd(0.425), _CMP_LE_OQ);
        const __m512d v1 = fast_inv_Phi_wo_checks_case_1_AVX512(p, q);
        // The tail result would be discarded in every lane. Avoid its log and two rational polynomials.
        if (central == 0xff) return v1;
        const __m512d v2 = fast_inv_Phi_wo_checks_case_2_AVX512(p, q);
        return _mm512_mask_blend_pd(central, v2, v1); // central lanes -> v1
  
}

 

// inline __m512d fast_inv_Phi_wo_checks_AVX512(const __m512d p) {
//       
//       const __m512d q = _mm512_sub_pd(p, _mm512_set1_pd(0.50));
//       const __mmask8 is_q_le_threshold = _mm512_cmp_pd_mask(_mm512_abs_pd(q), _mm512_set1_pd(0.425), _CMP_LE_OQ);
//        
//       if (_mm512_kortestc(is_q_le_threshold, is_q_le_threshold)) {    // All elements in q are outside the |q| <= 0.425 range, so call case 2
//         
//         return fast_inv_Phi_wo_checks_case_2_AVX512(p, q); 
//         
//       } else { // At least one element is within the |q| <= 0.425 range, so call case 1
//          
//         return fast_inv_Phi_wo_checks_case_1_AVX512(p, q);
//       }
//   
// } 




ALWAYS_INLINE __m512d fast_mills_ratio_tail_AVX512(const __m512d x) {              // x >= NORMAL_TAIL_THRESH
  
      __m512d f = x;
      for (int k = MILLS_CF_DEPTH; k >= 1; --k) {
          f = _mm512_add_pd(x, _mm512_div_pd(_mm512_set1_pd(static_cast<double>(k)), f));
      }
      return _mm512_div_pd(_mm512_set1_pd(1.0), f);
      
}




// ---------------------------------------------------------------------------------------------
//  Relative-accurate normal tail kernels for the GHK log-scale branches.
//  Include AFTER the fast_and_approx_*_fns.hpp headers (uses fast_exp_1_*, fast_log_1_*, fast_Phi_*,
//  fast_inv_Phi_wo_checks_case_{1,2a,2b}_* and _mm256_abs_pd / _mm512_abs_pd).
//
//  Everything is built on ONE function, the Mills ratio
//        R(x) = (1 - Phi(x)) / phi(x)  =  sqrt(pi/2) * erfcx(x / sqrt(2)),      x >= 0,
//  which is smooth, O(1/x) and has no cancellation. Then:
//        Phi(x)          = phi(x) * R(-x)                               (x < 0)
//        log Phi(x)      = -x^2/2 - 0.5*log(2*pi) + log R(-x)           (x < 0)
//        log(1 - Phi(x)) = log Phi(-x)
//        d/dx log Phi(x) = phi(x)/Phi(x) = 1 / R(-x)                     (x < 0)
//        inv_Phi from log p : AS241 tail case needs only r = sqrt(-log(min(p, 1-p)))
//        dZ/dlog_p    =  Phi(Z)/phi(Z)     = exp(log_p    + Z^2/2 + 0.5 log 2pi)  (=  R(-Z) for Z<0, stable side)
//        dZ/dlog_1m_p = -(1-Phi(Z))/phi(Z) = -exp(log_1m_p + Z^2/2 + 0.5 log 2pi)  (= -R( Z) for Z>0, stable side)
//
//  R(x) is evaluated with the Laplace continued fraction  R(x) = 1/(x + 1/(x + 2/(x + 3/(x + ...))))
//  by backward recurrence of depth MILLS_CF_DEPTH. Depth 40 is full double precision for x >= 2.5
//  (run normal_tail_self_test() once; lower the depth after benchmarking if you like — the tail
//  path is only taken for |x| > NORMAL_TAIL_THRESH, and a rational minimax in 1/x can replace the
//  fraction later; the self-test validates any replacement).
//
//  Body (|x| <= NORMAL_TAIL_THRESH) keeps your existing fast_Phi (Abramowitz-Stegun). Its absolute
//  error 7.5e-8 is a relative error of 1.2e-5 at the threshold and smaller inside; the tail side is
//  exact, so the join is continuous to ~1e-5 in log Phi. The remaining tail error is your fast_log's
//  (~1e-8); swap in the double-precision log constant if you want machine precision.
// ---------------------------------------------------------------------------------------------




// log Phi(x), all x; relative-accurate in the lower tail.
ALWAYS_INLINE __m512d fast_log_Phi_AVX512(const __m512d x) {
  
      const __m512d  thresh   = _mm512_set1_pd(NORMAL_TAIL_THRESH);
      const __m512d  ax       = _mm512_abs_pd(x);
      const __m512d  R        = fast_mills_ratio_tail_AVX512(_mm512_max_pd(ax, thresh));   // body lanes compute R(2.5), then discarded
      const __m512d  log_tail = _mm512_add_pd(_mm512_fnmadd_pd(_mm512_set1_pd(0.5), _mm512_mul_pd(x, x), _mm512_set1_pd(NEG_HALF_LOG_2PI)),
                                              fast_log_1_AVX512(R));
      const __m512d  log_body = fast_log_1_AVX512(fast_Phi_AVX512(x));
      const __mmask8 is_tail  = _mm512_cmp_pd_mask(x, _mm512_sub_pd(_mm512_setzero_pd(), thresh), _CMP_LT_OQ);   // x < -2.5
      return _mm512_mask_blend_pd(is_tail, log_body, log_tail);
      
}




ALWAYS_INLINE __m512d fast_log_1m_Phi_AVX512(const __m512d x) {                     // log(1 - Phi(x)) = log Phi(-x)
  
      return fast_log_Phi_AVX512(_mm512_sub_pd(_mm512_setzero_pd(), x));
  
}




// d/dx log Phi(x) = phi(x)/Phi(x).  Tail: 1/R(-x).  Body: phi / (the same fast_Phi the lp used).
ALWAYS_INLINE __m512d fast_dlog_Phi_dx_AVX512(const __m512d x) {
  
      const __m512d  thresh  = _mm512_set1_pd(NORMAL_TAIL_THRESH);
      const __m512d  ax      = _mm512_abs_pd(x);
      const __m512d  R       = fast_mills_ratio_tail_AVX512(_mm512_max_pd(ax, thresh));
      const __m512d  d_tail  = _mm512_div_pd(_mm512_set1_pd(1.0), R);
      const __m512d  phi     = _mm512_mul_pd(_mm512_set1_pd(INV_SQRT_2PI),
                                             fast_exp_1_AVX512(_mm512_mul_pd(_mm512_set1_pd(-0.5), _mm512_mul_pd(x, x))));
      const __m512d  d_body  = _mm512_div_pd(phi, fast_Phi_AVX512(x));
      const __mmask8 is_tail = _mm512_cmp_pd_mask(x, _mm512_sub_pd(_mm512_setzero_pd(), thresh), _CMP_LT_OQ);
      return _mm512_mask_blend_pd(is_tail, d_body, d_tail);
      
}
// d/dx log(1 - Phi(x)) = -fast_dlog_Phi_dx_AVX512(-x)




// inv_Phi given log(p) and log(1-p). AS241; tail case fed r = sqrt(-min(log_p, log_1m_p)) directly.
ALWAYS_INLINE __m512d fast_inv_Phi_from_log_p_wo_checks_AVX512(const __m512d log_p,
                                                               const __m512d log_1m_p) {
  
      const __m512d  zero    = _mm512_setzero_pd();
      const __m512d  p       = fast_exp_1_AVX512(log_p);                              // only consumed by the central case
      const __m512d  q       = _mm512_sub_pd(p, _mm512_set1_pd(0.5));
      const __mmask8 central = _mm512_cmp_pd_mask(_mm512_abs_pd(q), _mm512_set1_pd(0.425), _CMP_LE_OQ);
      //// tail
      const __m512d  log_min = _mm512_min_pd(log_p, log_1m_p);                        // log(min(p, 1-p))
      const __m512d  r       = _mm512_sqrt_pd(_mm512_sub_pd(zero, log_min));
      const __mmask8 r_gr_5  = _mm512_cmp_pd_mask(r, _mm512_set1_pd(5.0), _CMP_GT_OQ);
      __m512d v_tail = _mm512_mask_blend_pd(r_gr_5,
                                            fast_inv_Phi_wo_checks_case_2a_AVX512(p, r),   // r <= 5
                                            fast_inv_Phi_wo_checks_case_2b_AVX512(p, r));  // r >  5
      const __mmask8 lower   = _mm512_cmp_pd_mask(log_p, log_1m_p, _CMP_LT_OQ);       // p < 0.5 -> Z < 0
      v_tail = _mm512_mask_blend_pd(lower, v_tail, _mm512_sub_pd(zero, v_tail));
      //// central
      const __m512d  v_central = fast_inv_Phi_wo_checks_case_1_AVX512(p, q);
      return _mm512_mask_blend_pd(central, v_tail, v_central);
      
}




// dZ/dv through the stable side (see scalar version for the identities).
ALWAYS_INLINE __m512d fast_dZ_dv_from_log_p_AVX512( const __m512d Z,
                                                    const __m512d log_p, 
                                                    const __m512d log_1m_p,
                                                    const __m512d dlog_p_dv, 
                                                    const __m512d dlog_1m_p_dv) {
  
      const __m512d  log_inv_phi = _mm512_fmadd_pd(_mm512_set1_pd(0.5), _mm512_mul_pd(Z, Z), _mm512_set1_pd(-NEG_HALF_LOG_2PI));
      const __m512d  cap    = _mm512_set1_pd(700.0);                                    // keeps the UNUSED side finite
      const __m512d  e_lo   = _mm512_min_pd(_mm512_add_pd(log_p,    log_inv_phi), cap);
      const __m512d  e_hi   = _mm512_min_pd(_mm512_add_pd(log_1m_p, log_inv_phi), cap);
      const __m512d  via_lo = _mm512_mul_pd(fast_exp_1_AVX512(e_lo), dlog_p_dv);
      const __m512d  via_hi = _mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(), fast_exp_1_AVX512(e_hi)), dlog_1m_p_dv);
      const __mmask8 Z_neg  = _mm512_cmp_pd_mask(Z, _mm512_setzero_pd(), _CMP_LT_OQ);
      return _mm512_mask_blend_pd(Z_neg, via_hi, via_lo);                               // Z<0 -> via_lo, else via_hi
      
}


#endif
 
 
 
#endif
