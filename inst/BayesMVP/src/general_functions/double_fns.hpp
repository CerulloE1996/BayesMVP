
#pragma once

#ifndef STAN_MATH_PRIM_META_MVP_DOUBLE_FNS_HPP
#define STAN_MATH_PRIM_META_MVP_DOUBLE_FNS_HPP



 

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/err/check_pos_definite.hpp>
#include <stan/math/prim/err/check_square.hpp>
#include <stan/math/prim/err/check_symmetric.hpp>


#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/transpose.hpp>
#include <stan/math/prim/fun/dot_product.hpp>
#include <stan/math/prim/fun/norm2.hpp>
#include <stan/math/prim/fun/diagonal.hpp>
#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/eigenvalues_sym.hpp>
#include <stan/math/prim/fun/diag_post_multiply.hpp>

#include <stan/math/prim/fun/log_inv_logit.hpp>
#include <stan/math/prim/fun/fma.hpp>

#include <NicoSIMD/math/nico_double_exp_log_reference.hpp>
#include <NicoSIMD/math/nico_double_probability_reference.hpp>

//// ---- NicoSIMD (14 Nov) moved these scalar functions into nico::math::reference; its BayesMVP kernel headers
////      (kernel_dispatch_and_vector_loops_of_the_BayesMVP_manual_gradient_models), included after this file, call
////      them unqualified from nico::math, as before the move, so the same functions are declared there again:
namespace nico::math {
using nico::math::reference::nico_log_Phi;
using nico::math::reference::nico_log_Phi_approx;
using nico::math::reference::nico_log_Phi_approx_wo_checks;
using nico::math::reference::nico_inv_Phi_approx;
using nico::math::reference::nico_inv_Phi_approx_wo_checks;
using nico::math::reference::nico_inv_Phi_approx_from_logit_prob;
using nico::math::reference::nico_inv_Phi_approx_from_logit_prob_wo_checks;
using nico::math::reference::nico_inv_Phi_from_log_p;
}  // namespace nico::math


#include <stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/lkj_corr_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/weibull_lpdf.hpp>
#include <stan/math/prim/prob/gamma_lpdf.hpp>
#include <stan/math/prim/prob/beta_lpdf.hpp>



 

#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>
 

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)   //// (x86-64 only: ARM CPUs have no AVX)
#include <immintrin.h>
#endif
 

 
 
#define EIGEN_DONT_PARALLELIZE
 
 


 
 
 

template <typename T, int N>
inline std::array<Eigen::Matrix<T, -1, -1>, N> array_of_mats( int rows, 
                                                              int cols) {
  
       std::array<Eigen::Matrix<T, -1, -1>, N> arr;
       for (int c = 0; c < N; ++c) {
         arr[c] = Eigen::Matrix<T, -1, -1>::Zero(rows, cols);
       }
       
       return arr;
 
}


// ---------------------------------------------------------------------------------------------
//  Relative-accurate normal tail kernels for the GHK log-scale branches.
//  Include AFTER the nico_and_approx_*_fns.hpp headers (uses nico_exp_1_*, nico_log_1_*, nico_Phi_*,
//  nico_inv_Phi_wo_checks_case_{1,2a,2b}_* and _mm256_abs_pd / _mm512_abs_pd).
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
//  (run normal_tail_self_test() once; lower the depth after benchmarking if desired - the tail
//  path is only taken for |x| > NORMAL_TAIL_THRESH, and a rational minimax in 1/x can replace the
//  fraction later; the self-test validates any replacement).
//
//  Body (|x| <= NORMAL_TAIL_THRESH) keeps the existing nico_Phi (Abramowitz-Stegun). Its absolute
//  error 7.5e-8 is a relative error of 1.2e-5 at the threshold and smaller inside; the tail side is
//  exact, so the join is continuous to ~1e-5 in log Phi. The remaining tail error is the nico_log's
//  (~1e-8); swap in the double-precision log constant if machine precision is required.
// ---------------------------------------------------------------------------------------------


// =============================================================================================
//  SCALAR
// =============================================================================================








// scalar body uses std::erfc directly (already relative-accurate); vector bodies use nico_Phi








// d/dx log(1 - Phi(x)) = -nico_dlog_Phi_dx(-x)




// AS241 rationals (identical constants to the AVX code)












// inv_Phi given log(p) and log(1-p). No exp/log of an extreme probability anywhere.




// Total derivative dZ/dv through the STABLE side: log_p when Z<0, log_1m_p when Z>=0.
// Both are exact identities (p = exp(log_p) = 1 - exp(log_1m_p)); the blend only picks the
// evaluation that cannot overflow.




////
//// ---- Exact normal tails for the autodiff (stan::math::var) reference paths:
////
//// Used by the Phi_type = "Phi" branches of MVP_lp_grad_AD_fns.hpp, std_MVP_lp_grad_AD_fns.hpp,
//// MVOP_lp_grad_AD_fns.hpp and LC_LT_lp_grad_AD_fns.hpp, so that they keep the exact normal tails beyond
//// overflow_threshold / underflow_threshold instead of the Phi_approx (cubic-logistic) tails. The values come
//// from the SAME double kernels as the manual-gradient paths (nico_log_Phi, nico_inv_Phi_from_log_p above), and
//// the derivatives are supplied analytically through precomputed_gradients:
////
////   log Phi(x):                 d/dx log Phi(x) = phi(x) / Phi(x)
////                               (= nico_dlog_Phi_dx(x) = 1 / R(-x) for x < -2.5)
////   Z = Phi^{-1}(p), p = exp(log_p):
////                               dZ/dlog_p    =  p / phi(Z)       =  exp(log_p    + Z^2/2 + 0.5 log(2 pi))
////                               dZ/dlog_1m_p = -(1 - p) / phi(Z)  = -exp(log_1m_p + Z^2/2 + 0.5 log(2 pi))
////
//// log_p and log_1m_p are two representations of the same probability, so the total derivative of Z may be
//// routed through EITHER of them; it is routed through the side that cannot overflow (log_p when Z < 0, log_1m_p
//// when Z >= 0), exactly as nico_dZ_dv_from_log_p does. This is only valid when the caller computes log_p and
//// log_1m_p as consistent functions of the same upstream vars (every call site in the AD files does: one is
//// log1m_exp of the other, or both are exact expressions in the same (Bound_Z, u)).
////
//// Requires stan/math/rev (var_fns.hpp, which includes it, is always included before this file).
////
inline stan::math::var log_Phi_exact_var(const stan::math::var &x_var) {

      const double x_value = x_var.val();
      const double log_Phi_value = nico::math::reference::nico_log_Phi(x_value);
      const double derivative_of_log_Phi_wrt_x = nico::math::reference::nico_dlog_Phi_dx(x_value);
      return stan::math::precomputed_gradients(log_Phi_value,
                                               std::vector<stan::math::var>{x_var},
                                               std::vector<double>{derivative_of_log_Phi_wrt_x});

}




inline stan::math::var inv_Phi_from_log_p_exact_var(const stan::math::var &log_p_var,
                                                    const stan::math::var &log_1m_p_var) {

      const double log_p_value    = log_p_var.val();
      const double log_1m_p_value = log_1m_p_var.val();
      const double Z_value        = nico::math::reference::nico_inv_Phi_from_log_p(log_p_value, log_1m_p_value);
      const double log_inv_phi_Z  = 0.5 * Z_value * Z_value
                                     - nico::math::reference::NEG_HALF_LOG_2PI;    // log(1/phi(Z))
      if (Z_value < 0.0) {
            const double derivative_of_Z_wrt_log_p = std::exp(log_p_value + log_inv_phi_Z);
            return stan::math::precomputed_gradients(Z_value,
                                                     std::vector<stan::math::var>{log_p_var},
                                                     std::vector<double>{derivative_of_Z_wrt_log_p});
      } else {
            const double derivative_of_Z_wrt_log_1m_p = -std::exp(log_1m_p_value + log_inv_phi_Z);
            return stan::math::precomputed_gradients(Z_value,
                                                     std::vector<stan::math::var>{log_1m_p_var},
                                                     std::vector<double>{derivative_of_Z_wrt_log_1m_p});
      }

}


 
 
//// -------------------------------------------------------------------------------------------------------------------------------------------------------------
 
 
// The scalar version for comparison
inline double test_simple_double(const double x) {
 
   const double res = 2.0*x;
   return res;
 
} 
  
 
 
 // Other Misc. functions -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
 
inline bool Eigen_any_NaNs_process_return(Eigen::Ref<Eigen::Matrix<double, -1, -1>>  x_Ref) {

   return !((x_Ref.array() == x_Ref.array()).all());  // NaN  =/=  itself, so this works to check for NaNs

 }
 
 
 
inline bool Eigen_any_NaNs(Eigen::Matrix<double, -1, -1>  &x_R_val) {

   Eigen::Ref<Eigen::Matrix<double, -1, -1>> x_Ref(x_R_val);  // Directly create Eigen::Ref from R-value
   return  Eigen_any_NaNs_process_return(x_Ref);

 }


 inline bool Eigen_any_NaNs(Eigen::Matrix<double, -1, -1> x_L_val) {

   Eigen::Ref<Eigen::Matrix<double, -1, -1>> x_Ref(x_L_val);  // Directly create Eigen::Ref from L-value
   return  Eigen_any_NaNs_process_return(x_Ref);

 }


template <typename MatrixType>
inline bool Eigen_any_NaNs(Eigen::Ref<Eigen::Block<MatrixType, -1, -1>> x_Ref) {

  return  Eigen_any_NaNs_process_return(x_Ref);

}


inline bool Eigen_any_NaNs(Eigen::Array<double, -1, -1> x) {

  Eigen::Matrix<double, -1, -1>  x_matrix = x.matrix();
  return  Eigen_any_NaNs_process_return(x_matrix);

}



// 
// inline Eigen::Matrix<double, -1, 1> fn_colvec_double(Eigen::Array<double, -1, 1> x,
//                                                      const std::string &fn,
//                                                      const std::string &vect_type = "Eigen",
//                                                      const bool &skip_checks = false) {
//   
//   Eigen::Matrix<double, -1, 1>  x_matrix = x.matrix();
//   return fn_return_colvec_Ref_double_all(x_matrix, fn, vect_type, skip_checks);
//   
//   
// }



// 
// inline bool Eigen_any_NaNs(Eigen::Array<double, -1, -1> x_array) {
//  return !((x_array == x_array).all());
// }
// 
// 
// 
// template <typename BlockType>
// inline bool Eigen_any_NaNs(Eigen::Ref<Eigen::Block<BlockType>> &block) {
//   return !((block.array() == block.array()).all());
// }
// 




// template <typename Derived>
// inline bool Eigen_any_NaNs(const Eigen::DenseBase<Derived>& x) {
//   return !((x.array() == x.array()).all());  // NaN check
// }
// 
// 
// inline bool Eigen_any_NaNs(Eigen::Block<Eigen::Matrix<double, -1, -1>>& x) {
//   return !((x.array() == x.array()).all());  // NaN check for non-const block expressions
// }



 ////////////// douuble fn's -------------------------------------------------------------------------------------------------

 

  
 
 
//// 
//// Function to manually construct a cutpoint vector from raw unconstrained parameters - using exp() / log-differences.
////
inline Eigen::Matrix<double, -1, 1>	  construct_C( const Eigen::Matrix<double, -1, 1> &C_raw_vec,
                                                   // parameter - log_diffs
                                                   bool softplus) {
  
       const int n_total_cutpoints = C_raw_vec.size();
       Eigen::Matrix<double, -1, 1> C_vec(n_total_cutpoints);
       C_vec(0) = C_raw_vec(0); // first cutpoint is same as first raw_C/log_diff
      
       if (softplus == true) { 
           Eigen::Matrix<double, -1, 1> softplus_C_vec = stan::math::log1p_exp(C_raw_vec.segment(1, n_total_cutpoints - 1));
           for (int k = 2; k <= n_total_cutpoints; ++k) {
               C_vec(k - 1) = C_vec(k - 2) + softplus_C_vec(k - 2);
           }
       } else { 
           Eigen::Matrix<double, -1, 1> exp_C_vec = stan::math::exp(C_raw_vec.segment(1, n_total_cutpoints - 1));
           for (int k = 2; k <= n_total_cutpoints; ++k) {
               C_vec(k - 1) = C_vec(k - 2) + exp_C_vec(k - 2);
           }
       }
      
       return C_vec;
   
}


inline double raw_C_to_C_log_det_J_lp( const Eigen::Matrix<double, -1, 1> &raw_C, // parameter - log_diffs
                                       bool softplus) {
  
       const int n_cutpoints = raw_C.size();
       double log_det_J = 0.0;
       if (softplus == true) log_det_J += (stan::math::log_inv_logit(raw_C.segment(1, n_cutpoints - 1))).sum();
       else                  log_det_J += raw_C.segment(1, n_cutpoints - 1).sum();  
       return log_det_J;
       
}

//// ---------------------------------------------------------------------------------------
//// Induced-Dirichlet ("ind_dir") log-density function:
//// NOTE: This can be used for both ind_dir PRIORS and ind_dir MODELS:
//// NOTE: adapted from: Betancourt et al (see: https://betanalpha.github.io/assets/case_studies/ordinal_regression.html),
//// HOWEVER my version has a (much) more computationally efficient (lower-trianglar) Jacobian computation, which is 
//// mathematically still valid. 
////
inline double induced_dirichlet_given_C_lpdf( const Eigen::Matrix<double, -1, 1> &p_ord,
                                              const Eigen::Matrix<double, -1, 1> &C,
                                              const Eigen::Matrix<double, -1, 1> &alpha,
                                              bool use_probit_link) {
  
       const int n_cat = p_ord.size();
       const int n_thr = n_cat - 1;
        
       double log_prob = 0.0;
        
       for (int k = 1; k <= n_thr; ++k) {
          if (use_probit_link == true)  log_prob += stan::math::normal_lpdf(C(k - 1), 0.0, 1.0);
          else                          log_prob += stan::math::log_inv_logit(C(k - 1)) + stan::math::log1m_inv_logit(C(k - 1));
       }
       log_prob += stan::math::dirichlet_lpdf(p_ord, alpha);
        
       return log_prob;
  
}


////
//// Convert from cumul_probs -> ord_probs:
////
inline Eigen::Matrix<double, -1, 1> cumul_probs_to_ord_probs( const Eigen::Matrix<double, -1, 1> &cumul_probs) {
  
       const int n_thr = cumul_probs.size();
       const int n_cat = n_thr + 1;
       Eigen::Matrix<double, -1, 1> ord_probs(n_cat);
        
       ord_probs(0) = cumul_probs(0) - 0.0;
       for (int k = 2; k <= n_thr; ++k) {
          ord_probs(k - 1) = cumul_probs(k - 1) - cumul_probs(k - 2); // since probs are INCREASING with k
       }
       ord_probs(n_cat - 1) =  1.0 - cumul_probs(n_cat - 2);
        
       return ord_probs;
  
}







////
//// Convert from ord_probs -> cumul_probs:
////
inline Eigen::Matrix<double, -1, 1> ord_probs_to_cumul_probs( const Eigen::Matrix<double, -1, 1> &ord_probs) {
  
       const int n_cat = ord_probs.size();
       const int n_thr = n_cat - 1;
       Eigen::Matrix<double, -1, 1> cumul_probs(n_thr);
      
       cumul_probs(0) = ord_probs(0);
       for (int k = 2; k <= n_thr; ++k) {
         cumul_probs(k - 1) = cumul_probs(k - 2) + ord_probs(k - 1); // since probs are INCREASING with k
       }
      
       return cumul_probs;
  
}


////
//// Convert from cumul_probs -> C (w/ induced dirichlet):
////
// vector ID_cumul_probs_to_C( const Eigen::Matrix<double, -1, 1> &cumul_probs, 
//                             int use_probit_link) {
// 
//       int n_thr = stan::math::num_elements(cumul_probs);
//       int n_cat = n_thr + 1;
//       Eigen::Matrix<double, -1, 1> C(n_thr);
//       
//       for (k in 1:n_thr) {
//         if (cumul_probs[k] < 1e-300) {
//           C[k] = -37.5; ////  prob = 1e-38;
//         } else if (cumul_probs[k] > 0.99999999999999999) {
//           C[k] = +8.20; ////  prob = 0.9999999999999;
//         } else {
//           if (use_probit_link == true) C[k] = stan::math::inv_Phi(cumul_probs[k]); 
//           else                         C[k] = stan::math::logit(cumul_probs[k]); 
//         }
//       }
//       
//       return C;
//     
// }

inline Eigen::Matrix<double, -1, 1> ID_cumul_probs_to_C( const Eigen::Matrix<double, -1, 1> &cumul_probs, 
                                                         bool use_probit_link) {
  
       const int n_thr = cumul_probs.size();
       const int n_cat = n_thr + 1;
       Eigen::Matrix<double, -1, 1> C(n_thr);
      
      for (int k = 0; k < n_thr; ++k) {
         if (cumul_probs(k) < 1e-300) {
           C(k) = -37.5; ////  prob = 1e-38;
         } else if (cumul_probs(k) > 0.99999999999999999) {
           C(k) = +8.20; ////  prob = 0.9999999999999;
         } else {
           if (use_probit_link == true) C(k) = stan::math::inv_Phi(cumul_probs(k)); 
           else                         C(k) = stan::math::logit(cumul_probs(k)); 
         }
       }
      
       return C;
  
}
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

inline std::array<Eigen::Matrix<double, -1, -1>, 1>      array_of_mats_1d(  int n_rows,
                                                                            int n_cols) {
   
   
       std::array<Eigen::Matrix<double, -1, -1 >, 1> my_1d_array;
       Eigen::Matrix<double, -1, -1> my_mat = Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
       my_1d_array[0] = my_mat;
     
       return my_1d_array;
   
}
 
 
inline std::array<Eigen::Matrix<double, -1, -1>, 2> array_of_mats_2d( int n_rows,
                                                                      int n_cols) {
   
       std::array<Eigen::Matrix<double, -1, -1>, 2> my_2d_array = {
                                 Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols),
                                 Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols)
                               };
     
       return  my_2d_array;
   
}
 
  
// convert std vec to eigen vec - double
inline Eigen::Matrix<double, -1, 1> std_vec_to_Eigen_vec( const std::vector<double> &std_vec) {
   
       Eigen::Matrix<double, -1, 1>  Eigen_vec(std_vec.size());
       
       for (int i = 0; i < std_vec.size(); ++i) {
         Eigen_vec(i) = std_vec[i];
       }
       
       return(Eigen_vec);
   
}
 
 
 inline std::vector<double> Eigen_vec_to_std_vec( const Eigen::Matrix<double, -1, 1> &Eigen_vec) {
   
       std::vector<double>  std_vec(Eigen_vec.rows());
       
       for (int i = 0; i < Eigen_vec.rows(); ++i) {
         std_vec[i] = Eigen_vec(i);
       }
       
       return(std_vec);
   
}
 
 
inline Eigen::Matrix<double, 1, -1>     fn_first_element_neg_rest_pos( Eigen::Matrix<double, 1, -1>  &row_vec) {
   
       row_vec(0) = - row_vec(0);
       
       return(row_vec);
       
}
 
 
 
 
 
 
 // 
 // 
 // inline  std::unique_ptr<size_t[]> get_commutation_unequal_vec
 // (unsigned const n, unsigned const m, bool const transpose){
 //   
 //   unsigned const nm = n * m,
 //     nnm_p1 = n * nm + 1L,
 //     nm_pm = nm + m;
 //   std::unique_ptr<size_t[]> out(new size_t[nm]);
 //   size_t * const o_begin = out.get();
 //   size_t idx = 0L;
 //   for(unsigned i = 0; i < n; ++i, idx += nm_pm){
 //     size_t idx1 = idx;
 //     for(unsigned j = 0; j < m; ++j, idx1 += nnm_p1)
 //       if(transpose)
 //         *(o_begin + idx1 / nm) = (idx1 % nm);
 //       else
 //         *(o_begin + idx1 % nm) = (idx1 / nm);
 //   }
 //   
 //   return out;
 //   
 // }
 // 
 // 
 // 
 // inline  Rcpp::NumericVector commutation_dot
 // (unsigned const n, unsigned const m, Rcpp::NumericVector x,
 //  bool const transpose){
 //   size_t const nm = n * m;
 //   Rcpp::NumericVector out(nm);
 //   auto const indices = get_commutation_unequal_vec(n, m, transpose);
 //   
 //   for(size_t i = 0; i < nm; ++i)
 //     out[i] = x[*(indices.get() +i )];
 //   
 //   return out;
 // }
 // 
 // inline  Rcpp::NumericMatrix get_commutation_unequal
 // (unsigned const n, unsigned const m){
 //   
 //   unsigned const nm = n * m,
 //     nnm_p1 = n * nm + 1L,
 //     nm_pm = nm + m;
 //   Rcpp::NumericMatrix out(nm, nm);
 //   double * o = &out[0];
 //   for(unsigned i = 0; i < n; ++i, o += nm_pm){
 //     double *o1 = o;
 //     for(unsigned j = 0; j < m; ++j, o1 += nnm_p1)
 //       *o1 = 1.;
 //   }
 //   
 //   return out;
 // }
 // 
 // inline  Rcpp::NumericMatrix get_commutation_equal(unsigned const m){
 //   unsigned const mm = m * m,
 //     mmm = mm * m,
 //     mmm_p1 = mmm + 1L,
 //     mm_pm = mm + m;
 //   Rcpp::NumericMatrix out(mm, mm);
 //   double * const o = &out[0];
 //   unsigned inc_i(0L);
 //   for(unsigned i = 0; i < m; ++i, inc_i += m){
 //     double *o1 = o + inc_i + i * mm,
 //       *o2 = o + i     + inc_i * mm;
 //     for(unsigned j = 0; j < i; ++j, o1 += mmm_p1, o2 += mm_pm){
 //       *o1 = 1.;
 //       *o2 = 1.;
 //     }
 //     *o1 += 1.;
 //   }
 //   return out;
 // }
 // 
 // 
 
 // 
 // 
 // inline  Eigen::Matrix<double, -1, -1  >  get_commutation(unsigned const n, unsigned const m) {
 //   
 //   if (n == m)  {
 //     
 //     Rcpp::NumericMatrix commutation_mtx_Nuemric_Matrix =  get_commutation_equal(n);
 //     
 //     double n_rows = commutation_mtx_Nuemric_Matrix.nrow();
 //     double n_cols = commutation_mtx_Nuemric_Matrix.ncol();
 //     
 //     Eigen::Matrix<double, -1, -1>  commutation_mtx_Eigen   =  Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
 //     
 //     
 //     for (int i = 0; i < n_rows; ++i) {
 //       for (int j = 0; j < n_cols; ++j) {
 //         commutation_mtx_Eigen(i, j) = commutation_mtx_Nuemric_Matrix(i, j) ;
 //       }
 //     }
 //     
 //     return commutation_mtx_Eigen;
 //     
 //     
 //   } else {
 //     
 //     Rcpp::NumericMatrix commutation_mtx_Nuemric_Matrix =  get_commutation_unequal(n, m);
 //     
 //     double n_rows = commutation_mtx_Nuemric_Matrix.nrow();
 //     double n_cols = commutation_mtx_Nuemric_Matrix.ncol();
 //     
 //     Eigen::Matrix<double, -1, -1>  commutation_mtx_Eigen   =  Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
 //     
 //     
 //     for (int i = 0; i < n_rows; ++i) {
 //       for (int j = 0; j < n_cols; ++j) {
 //         commutation_mtx_Eigen(i, j) = commutation_mtx_Nuemric_Matrix(i, j) ;
 //       }
 //     }
 //     
 //     return commutation_mtx_Eigen;
 //     
 //     
 //   }
 //   
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
 // inline Eigen::Matrix<double, -1, -1  > elimination_matrix(const int &n) {
 //   
 //   Eigen::Matrix<double, -1, -1> out   =  Eigen::Matrix<double, -1, -1>::Zero((n*(n+1))/2,  n*n);
 //   
 //   for (int j = 0; j < n; ++j) {
 //     Eigen::Matrix<double, 1, -1> e_j   =  Eigen::Matrix<double, 1, -1>::Zero(n);
 //     
 //     e_j(j) = 1.0;
 //     
 //     for (int i = j; i < n; ++i) {
 //       Eigen::Matrix<double, -1, 1> u   =  Eigen::Matrix<double, -1, 1>::Zero((n*(n+1))/2);
 //       u(j*n+i-((j+1)*j)/2) = 1.0;
 //       Eigen::Matrix<double, 1, -1> e_i   =  Eigen::Matrix<double, 1, -1>::Zero(n);
 //       e_i(i) = 1.0;
 //       
 //       out += Eigen::kroneckerProduct(u, Eigen::kroneckerProduct(e_j, e_i));
 //     }
 //   }
 //   
 //   return out;
 // }
 // 
 // 
 // 
 // 
 // 
 // 
 // 
 // 
 // 
 // inline  Eigen::Matrix<double, -1, -1  > duplication_matrix(const int &n) {
 //   
 //   //arma::mat out((n*(n+1))/2, n*n, arma::fill::zeros);
 //   Eigen::Matrix<double, -1, -1> out   =  Eigen::Matrix<double, -1, -1>::Zero((n*(n+1))/2,  n*n);
 //   
 //   for (int j = 0; j < n; ++j) {
 //     for (int i = j; i < n; ++i) {
 //       // arma::vec u((n*(n+1))/2, arma::fill::zeros);
 //       Eigen::Matrix<double, -1, 1> u   =  Eigen::Matrix<double, -1, 1>::Zero((n*(n+1))/2);
 //       u(j*n+i-((j+1)*j)/2) = 1.0;
 //       
 //       //       arma::mat T(n,n, arma::fill::zeros);
 //       Eigen::Matrix<double, -1, -1> T   =  Eigen::Matrix<double, -1, -1>::Zero(n, n);
 //       T(i,j) = 1.0;
 //       T(j,i) = 1.0;
 //       
 //       Eigen::Map<Eigen::Matrix<double, -1, 1> > T_vec(T.data(), n*n);
 //       
 //       out += u * T_vec.transpose();
 //     }
 //   }
 //   
 //   return out.transpose();
 //   
 // }
 // 
 // 
 // 
 // 
 // 
 // 
 // 
 
 
 
 
 
 
 
 
 inline  Eigen::Matrix<double, -1, -1  > inv_Phi_approx_from_logit_prob_Stan(  Eigen::Matrix<double, -1, -1  > logit_p)   {

   using namespace stan::math;
   Eigen::Array<double, -1, -1  > x_i = 0.34176618822627863*logit_p.array();
   Eigen::Array<double, -1, -1  > asinh_stuff_div_3 =  0.33333333333333331483 *   log(  ( x_i.array()  +  sqrt(fma(x_i.matrix(), x_i.matrix(), 1.0)).array()  ).matrix() ).array() ;          // now do arc_sinh part
   Eigen::Array<double, -1, -1  > exp_x_i = exp(asinh_stuff_div_3.matrix() ).array();
   return (  2.7472242570765295 * (  fma(exp_x_i.matrix(), exp_x_i.matrix(), -1.0).array()   / exp_x_i ).array()   ).matrix()    ;  //   now do sinh parth part
   
 }
 

 
 
  
  inline  Eigen::Matrix<double, -1, -1  > inv_Phi_approx_from_logit_prob_Eigen_double(  Eigen::Matrix<double, -1, -1  > logit_p)   {

   Eigen::Array<double, -1, -1  > x_i = 0.34176618822627863*logit_p.array();
   Eigen::Array<double, -1, -1  > asinh_stuff_div_3 =  0.33333333333333331483 *  ( x_i  + (x_i*x_i + 1.0 ).sqrt() ).log() ;   // now do arc_sinh part
   Eigen::Array<double, -1, -1  > exp_x_i =  (asinh_stuff_div_3).exp();
   return  (  2.7472242570765295 * ( (exp_x_i*exp_x_i  - 1.0) / exp_x_i )  ).matrix()   ;  //   now do sinh parth part

 }
 
 
 
 
 
 
 
 
 
 // // NOTE: Compiler needs  to auto-vectorise for the following to be fast!!
 inline  Eigen::Matrix<double, -1, -1  > fn_mat_loop_dbl_Eigen(  Eigen::Matrix<double, -1, -1  > x, 
                                                                 std::function<double(double)> fn_double) {

   int  N = x.rows();     ///// Eigen::Matrix<double, -1, 1  >  out(N); /// NOTABLY slower if store in new array!!!!!! (making arrays is taxing !!!!)
   int  M = x.cols();
   for (int j = 0; j < M; ++j) {
     for (int i = 0; i < N; ++i) {
       x(i, j) = fn_double(x(i, j));
     }
   }
   return  x;  

 }


 // // NOTE: Compiler needs  to auto-vectorise for the following to be fast!!
 inline  Eigen::Matrix<double, -1, 1  > fn_colvec_loop_dbl_Eigen(  Eigen::Matrix<double, -1, 1  > x, 
                                                                   std::function<double(double)> fn_double) {

   for (int i = 0; i < x.rows(); ++i) {
     x(i) = fn_double(x(i));
   }  
   return  x;

 }
  
 
 
 
 
 



 


 
 
 
 
// 
// // -------------------------------------------------------------------------------------------------------------------------------------------------------------











 

// checks specifically for COL vectors  -  i.e., true if = Eigen::Matrix<-1, 1, double> [COMPILE-time]
template<typename T>
struct is_eigen_col_vector {
  static constexpr bool value = std::is_base_of_v<Eigen::MatrixBase<typename std::decay<T>::type>, typename std::decay<T>::type> && 
    T::ColsAtCompileTime == 1;
}; 

// checks specifically for ROW vectors  -  i.e., true if = Eigen::Matrix<1, -1, double> [COMPILE-time]
template<typename T>
struct is_eigen_row_vector {
  static constexpr bool value = std::is_base_of_v<Eigen::MatrixBase<typename std::decay<T>::type>, typename std::decay<T>::type> && 
    T::RowsAtCompileTime == 1; 
}; 

// checks specifically for [DYNAMIC] MATRICES (more than one row AND more than one column) - i.e., true if = Eigen::Matrix<-1, -1, double> 
template<typename T>
struct is_dynamic_matrix {
  static constexpr bool value = 
    std::is_base_of_v<Eigen::MatrixBase<typename std::decay<T>::type>, 
                      typename std::decay<T>::type> && 
                        T::RowsAtCompileTime == -1 && 
                        T::ColsAtCompileTime == -1 &&
                        std::is_same_v<typename T::Scalar, double>;
}; 
 

///////////////////  fns - exp   -----------------------------------------------------------------------------------------------------------------------------





// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
/* For a in [0.5, 4), compute a * 2**i, -250 < i < 250 */
// Note: this function is  the same as the one in the link above, but for * double * instead of * float *.
// not handling edge cases  - should work for a in [0.5, 2.0). 

/// this is the same as nico_ldexp but handles edge-cases (will have wider input range )
/// if i is too large, splits the calc. into two steps






// Adapted from: https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
// added (optional) extra degree(s) for poly approx (oroginal float fn had 4 degrees) - using "minimaxApprox" R package to find coefficient terms
// R code:   minimaxApprox::minimaxApprox(fn = exp, lower = -0.346573590279972643113, upper = 0.346573590279972643113, degree = 5, basis ="Chebyshev")


 
 
 
 
 



// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// Note: this function is  the same as the one in the link above, but for * double * instead of * float *.
 






// 
///////////////////  fns - log   -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//https://stackoverflow.com/a/65537754/9007125

//  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html

//  compute log(1-x) without losing precision for small values of x.

//  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
 
 
 

//https://stackoverflow.com/a/65537754/9007125 /* compute natural logarithm, maximum error 0.85089 ulps */

//  compute log(1+x) without losing precision for small values of x.
//  see: https://www.johndcook.com/cpp_log_one_plus_x.html

//  compute log(1-x) without losing precision for small values of x.

//  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
 












///////////////////  fns - inv_logit   ----------------------------------------------------------------------------------------------------------------




 












 
















///////////////////  fns - log_inv_logit   ----------------------------------------------------------------------------------------------------------------




 


















 




 
 
//  ///////////////////  fns - Phi_approx   ----------------------------------------------------------------------------------------------------------------




 


 


 

 




///////////////////  fns - inv_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






inline  Eigen::Matrix<double, -1, -1  > inv_Phi_approx_Eigen_double(  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> x)   {
  
  Eigen::Matrix<double, -1, -1  > x_i = -0.34176618822627863*( 1.0/x.array()  - 1.0 ).log();
  Eigen::Matrix<double, -1, -1  > asinh_stuff_div_3 = ( 0.33333333333333331483 *  ( x_i.array()   + (x_i.array() *x_i.array()  + 1.0 ).sqrt() ).log()  ).matrix() ;          // now do arc_sinh part
  Eigen::Matrix<double, -1, -1  > exp_x_i =  (asinh_stuff_div_3).exp();
  return  2.7472242570765295 * ( ( exp_x_i.array() *exp_x_i.array()   - 1.0).array()  / exp_x_i.array()  ).matrix() ;  //   now do sinh parth part
  
} 












inline Eigen::Matrix<double, -1, -1> inv_Phi_approx_Stan(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> x)   {
  
            Eigen::Matrix<double, -1, -1  > x_i = ( -0.34176618822627863*stan::math::log(  ( 1.0/x.array() - 1.0 ).matrix()  ).array()  ).matrix() ;
            Eigen::Matrix<double, -1, -1  > asinh_stuff_div_3 = (  0.33333333333333331483 *  stan::math::log(  ( x_i.array()  + stan::math::sqrt( (x_i.array()*x_i.array() + 1.0 ).matrix() ).array()  ).matrix()  ).array() ).matrix() ;          // now do arc_sinh part
            Eigen::Matrix<double, -1, -1  > exp_x_i =  stan::math::exp(asinh_stuff_div_3.matrix() ).matrix();
            
            return  ( 2.7472242570765295 * ( ( exp_x_i.array()*exp_x_i.array()  - 1.0) / exp_x_i.array() ).array() ).matrix() ;  //   now do sinh parth part

}




 
inline Eigen::Matrix<double, -1, -1  > inv_Phi_approx_Eigen(  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  x)   {
  
            Eigen::Array<double, -1, -1  > x_i = -0.34176618822627863*( 1.0/x.array() - 1.0 ).log();
            Eigen::Array<double, -1, -1  > asinh_stuff_div_3 =  0.33333333333333331483 *  ( x_i  + (x_i*x_i + 1.0 ).sqrt() ).log() ;          // now do arc_sinh part
            Eigen::Array<double, -1, -1  > exp_x_i =  (asinh_stuff_div_3).exp();
            
            return   (  2.7472242570765295 * ( ( exp_x_i*exp_x_i  - 1.0) / exp_x_i )  ).matrix() ;  //   now do sinh parth part
            
}






 
 


 






// 




// need to add citation to this (slight modification from a forum post)


 



// need to add citation to this (slight modification from a forum post)
 

// need to add citation to this (slight modification from a forum post)
 
// 
///////////////////  fns - log_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------


inline Eigen::Matrix<double, -1, 1>  log_Phi_approx_Eigen_double( const  Eigen::Matrix<double, -1, 1> &q )  {

  return  ( - ((-(0.07056*q.array()*q.array()*q.array() + 1.5976*q.array())).exp()).log1p() ).matrix() ;

}


inline Eigen::Matrix<double, -1, 1>  log_Phi_approx_Stan_double( const  Eigen::Matrix<double, -1, 1> &q )  {

  return stan::math::log_inv_logit( (q.array()*(stan::math::fma(0.07056*q.matrix(), q.matrix(), 1.5976)).array()  ).matrix() ) ;

}


 
 
 

 







/// this still has no bounds checking as it uses log_inv_logit 
 






//////////////////// ------------- tanh  --------------------------------------------------------------------------------------------------------------------------------------

 



 



 
 
 
 
 
 
 
 
 
 
//////////////////////  fns - error functions   ----------------------------------------------------------------------------------------------------------------------------------------
// 
// 
//  // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function
//  inline  double nico_erf_wo_checks_part_1_upper(const double a,
//                                                 const double t,
//                                                 const double s) {
//      double r, u;
// 
//      // max ulp error = 0.97749 (USE_EXPM1 = 1); 1.05364 (USE_EXPM1 = 0)
//      r = fma (-5.6271698391213282e-18, t, 4.8565951797366214e-16); // -0x1.9f363ba3b515dp-58, 0x1.17f6b1d68f44bp-51
//      u = fma (-1.9912968283386570e-14, t, 5.1614612434698227e-13); // -0x1.66b85b7fbd01ap-46, 0x1.22907eebc22e0p-41
//      r = fma (r, s, u);
//      u = fma (-9.4934693745934645e-12, t, 1.3183034417605052e-10); // -0x1.4e0591fd97592p-37, 0x1.21e5e2d8544d1p-33
//      r = fma (r, s, u);
//      u = fma (-1.4354030030292210e-09, t, 1.2558925114413972e-08); // -0x1.8a8f81b7e0e84p-30, 0x1.af85793b93d2fp-27
//      r = fma (r, s, u);
//      u = fma (-8.9719702096303798e-08, t, 5.2832013824348913e-07); // -0x1.8157db0edbfa8p-24, 0x1.1ba3c453738fdp-21
//      r = fma (r, s, u);
//      u = fma (-2.5730580226082933e-06, t, 1.0322052949676148e-05); // -0x1.595999b7e922dp-19, 0x1.5a59c27b3b856p-17
//      r = fma (r, s, u);
//      u = fma (-3.3555264836700767e-05, t, 8.4667486930266041e-05); // -0x1.197b61ee37123p-15, 0x1.631f0597f62b8p-14
//      r = fma (r, s, u);
//      u = fma (-1.4570926486271945e-04, t, 7.1877160107954648e-05); // -0x1.319310dfb8583p-13, 0x1.2d798353da894p-14
//      r = fma (r, s, u);
//      u = fma ( 4.9486959714661590e-04, t,-1.6221099717135270e-03); //  0x1.037445e25d3e5p-11,-0x1.a939f51db8c06p-10
//      r = fma (r, s, u);
//      u = fma ( 1.6425707149019379e-04, t, 1.9148914196620660e-02); //  0x1.5878d80188695p-13, 0x1.39bc5e0e9e09ap-6
//      r = fma (r, s, u);
//      r = fma (r, t, -1.0277918343487560e-1); // -0x1.a4fbc8f8ff7dap-4
//      r = fma (r, t, -6.3661844223699315e-1); // -0x1.45f2da3ae06f8p-1
//      r = fma (r, t, -1.2837929411398119e-1); // -0x1.06ebb92d9ffa8p-3
//      r = fma (r, t, -t);
//      r = 1.0 - nico_exp_1(r);
// 
//      return copysign (r, a);
// 
//  }
// 
// // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function
// inline double nico_erf_wo_checks_part_2_lower(const double a,
//                                               const double t,
//                                               const double s) {
//     double r, u;
// 
//     // max ulp error = 1.01912
//     r =            -7.7794684889591997e-10; // -0x1.abae491c44131p-31
//     r = fma (r, s,  1.3710980398024347e-8); //  0x1.d71b0f1b10071p-27
//     r = fma (r, s, -1.6206313758492398e-7); // -0x1.5c0726f04dbc7p-23
//     r = fma (r, s,  1.6447131571278227e-6); //  0x1.b97fd3d9927cap-20
//     r = fma (r, s, -1.4924712302009488e-5); // -0x1.f4ca4d6f3e232p-17
//     r = fma (r, s,  1.2055293576900605e-4); //  0x1.f9a2baa8fedc2p-14
//     r = fma (r, s, -8.5483259293144627e-4); // -0x1.c02db03dd71bbp-11
//     r = fma (r, s,  5.2239776061185055e-3); //  0x1.565bccf92b31ep-8
//     r = fma (r, s, -2.6866170643111514e-2); // -0x1.b82ce311fa94bp-6
//     r = fma (r, s,  1.1283791670944182e-1); //  0x1.ce2f21a040d14p-4
//     r = fma (r, s, -3.7612638903183515e-1); // -0x1.812746b0379bcp-2
//     r = fma (r, s,  1.2837916709551256e-1); //  0x1.06eba8214db68p-3
// 
//     return fma (r, a, a);
// 
// }
// 
// // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function
// inline double nico_erf_wo_checks (const double a) {
// 
//   double r, s, t, u;
//   t = fabs (a);
//   s = a * a;
//   if (t >= 1.0)    return nico_erf_wo_checks_part_1_upper(a, t, s);
//   else             return nico_erf_wo_checks_part_2_lower(a, t, s);
// 
// }
// 
// 




// ///////////////////  fns - Phi functions   ----------------------------------------------------------------------------------------------------------------------------------------
 
 
// inline double nico_Phi_wo_checks(const double x) {
// 
//    const double sqrt_2_recip = 0.707106781186547461715;
//    return 0.5 * (1.0 + nico_erf_wo_checks(x * sqrt_2_recip));
// 
// }
//  
 


//// ---- nico_Phi: relative-minimax rational approximation of the Mills ratio (instead of Abramowitz & Stegun
////      26.2.17):
////
////   Phi(-z) = phi(z) * R(z),   R(z) = Phi(-z) / phi(z) = sqrt(pi/2) * erfcx(z / sqrt(2)),   z = |x|,
////   R(z) ~= P6(z) / Q7(z)  on z in [0, 37.5],   max relative error 7.8e-12 (measured on a dense grid in mpmath,
////                                                double coefficients),
////   so  Phi(-z) = exp(-z^2/2) * N6(z) / Q7(z)  with  N6 = P6 / sqrt(2 pi)  (the 1/sqrt(2 pi) of phi is folded
////                                                into N6),
////   and Phi(x) = 1 - Phi(-x) for x > 0 (the same reflection).
////
//// Why: A&S 26.2.17 (Zelen & Severo 1964 / Hastings 1955 form, published) has absolute error <= 7.5e-8 but
//// RELATIVE error 5e-5 at x = -3, 8e-3 at -7.5 and 0.16 at -37.5, whilst every caller registers the exact
//// density phi(x) as the gradient, so value and gradient would disagree in the lower tail (and log(nico_Phi)
//// would be off by up to 0.17 nats). The rational form is relative-accurate for every x, branch-free, and costs
//// the same as A&S: one exp, one divide and 13 FMAs (A&S: 1 FMA + 9 multiplies + 4 adds).
////
//// Provenance: the Mills-ratio/rational form is standard (e.g. Cody 1969, Math. Comp. 23:631-637, uses rationals
//// for erfc); THESE coefficients were fitted with a relative-error Sanathanan-Koerner / Lawson near-minimax fit
//// in 40-digit mpmath, constrained to N6(0) = 0.5 exactly so that nico_Phi(0) = 0.5 with no jump at x = 0.
//// FAST_PHI_MILLS_Z_CAP caps only the rational's argument (the exp still sees the true z): it keeps N and Q
//// finite for |x| = inf / huge |x| (exp(-z^2/2) = 0 there, so the result is exactly 0 / 1) and does not change
//// any x in [-37.5, 8.25].




//// Phi(x) = exp(-z^2/2) * N(z) / Q(z) for x < 0, 1 minus that for x >= 0 (z = |x|): see the FAST_PHI_MILLS_*
//// block above. Same rational as nico_Phi_wo_checks_AVX2 / _AVX512. It uses std::exp, not the scalar nico_exp_1:
//// nico_exp_1_wo_checks's -log(2) split (l2h + l2l) is off by 4.7e-11, which gives exp(-z^2/2) a relative error
//// of ~|i| * 4.7e-11 (i = the binary exponent), i.e. 2e-9 at z = 7.5 and 5e-8 at z = 37.5; std::exp keeps the
//// scalar kernel within ~1e-14 of the AVX kernels (nico_exp_1_AVX2 / _AVX512 use the exact split
//// -0.693145751953125 + -1.42860682030941723212e-6).













// ///////////////////  fns - inverse-error functions   ----------------------------------------------------------------------------------------------------------------------------------------
//  //  compute inverse error functions with maximum error of 2.35793 ulp  // see: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
// 
//  inline double nico_inv_erf_wo_checks_part_1_upper(const double a,
//                                                    const double t) {
// 
//      double p =            3.03697567e-10; //  0x1.4deb44p-32
// 
//      p = fma (p, t,  2.93243101e-8); //  0x1.f7c9aep-26
//      p = fma (p, t,  1.22150334e-6); //  0x1.47e512p-20
//      p = fma (p, t,  2.84108955e-5); //  0x1.dca7dep-16
//      p = fma (p, t,  3.93552968e-4); //  0x1.9cab92p-12
//      p = fma (p, t,  3.02698812e-3); //  0x1.8cc0dep-9
//      p = fma (p, t,  4.83185798e-3); //  0x1.3ca920p-8
//      p = fma (p, t, -2.64646143e-1); // -0x1.0eff66p-2
//      p = fma (p, t,  8.40016484e-1); //  0x1.ae16a4p-1
// 
//      return a * p;
// 
//  }
// 
// //  compute inverse error functions with maximum error of 2.35793 ulp  // see: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
// 
//  inline double nico_inv_erf_wo_checks_part_2_lower(const double a,
//                                                    const double t) {
// 
//      double p =             5.43877832e-9;  //  0x1.75c000p-28
// 
//      p = fma (p, t,  1.43285448e-7); //  0x1.33b402p-23
//      p = fma (p, t,  1.22774793e-6); //  0x1.499232p-20
//      p = fma (p, t,  1.12963626e-7); //  0x1.e52cd2p-24
//      p = fma (p, t, -5.61530760e-5); // -0x1.d70bd0p-15
//      p = fma (p, t, -1.47697632e-4); // -0x1.35be90p-13
//      p = fma (p, t,  2.31468678e-3); //  0x1.2f6400p-9
//      p = fma (p, t,  1.15392581e-2); //  0x1.7a1e50p-7
//      p = fma (p, t, -2.32015476e-1); // -0x1.db2aeep-3
//      p = fma (p, t,  8.86226892e-1); //  0x1.c5bf88p-1
// 
//      return a * p;
// 
//  }
// 
// //  compute inverse error functions with maximum error of 2.35793 ulp  // see: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
// 
//  inline double nico_inv_erf_wo_checks(const double a) {
// 
//   double t = fma(a, 0.0 - a, 1.0);
//   t = nico_log_1(t);
// 
//   if (fabs(t) > 6.125) return nico_inv_erf_wo_checks_part_1_upper(a, t);
//   else                 return nico_inv_erf_wo_checks_part_2_lower(a, t);
// 
// }
// 
// 
// 
// 
// 
// ///////////////////  fns - inv_Phi functions   ----------------------------------------------------------------------------------------------------------------------------------------
// 
// inline double nico_inv_Phi_wo_checks(const double x) {
// 
//    const double sqrt_2 = 1.41421356237309514547;
//    return sqrt_2 * nico_inv_erf_wo_checks(2.0*x - 1.0);
// 
// }


/// FMA version of the  Stan fn provided by Sean Pinkney here:  https://github.com/stan-dev/math/issues/2555 


 
 
 
 /////////////////// other  ----------------------------------------------------------------------------------------------------------------------------------------
  
 
inline Eigen::Matrix<double, 1, -1>     fn_first_element_neg_rest_pos( Eigen::Ref<Eigen::Matrix<double, 1, -1>>  row_vec    ) {

   row_vec(0) = - row_vec(0);

   return(row_vec);

 }

 

 
 
 
#endif
 
 
 
 
 
 
 
 
 
 
 
 
 









