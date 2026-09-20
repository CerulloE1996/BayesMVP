
#pragma once
 
 

#include <stan/math/rev.hpp>
 
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
 
 
#include <stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/lkj_corr_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/weibull_lpdf.hpp>
#include <stan/math/prim/prob/gamma_lpdf.hpp>
#include <stan/math/prim/prob/beta_lpdf.hpp>





#include <Eigen/Dense>


 
 
#if defined(__AVX2__) || defined(__AVX512F__) 
#include <immintrin.h>
#endif
 
 
 

// [[Rcpp::plugins(cpp17)]]

 

using namespace Eigen;

 
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
 
 
 
 
////////////// var fn's -------------------------------------------------------------------------------------------------

////
//// Function to manually construct a cutpoint vector from raw unconstrained parameters - using exp() / log-differences.
////
inline  Eigen::Matrix<stan::math::var, -1, 1> construct_C_var( Eigen::Matrix<stan::math::var, -1, 1> C_raw_vec, // parameter - log_diffs
                                                              bool softplus) {
       
       const int n_total_cutpoints = C_raw_vec.size();
       Eigen::Matrix<stan::math::var, -1, 1> C_vec(n_total_cutpoints);
       C_vec(0) = C_raw_vec(0); // first cutpoint is same as first raw_C/log_diff
       
       if (softplus == true) { 
         Eigen::Matrix<stan::math::var, -1, 1> softplus_C_vec = stan::math::log1p_exp(C_raw_vec.segment(1, n_total_cutpoints - 1));
         for (int k = 2; k <= n_total_cutpoints; ++k) {
           C_vec(k - 1) = C_vec(k - 2) + softplus_C_vec(k - 2);
         }
       } else { 
         Eigen::Matrix<stan::math::var, -1, 1> exp_C_vec = stan::math::exp(C_raw_vec.segment(1, n_total_cutpoints - 1));
         for (int k = 2; k <= n_total_cutpoints; ++k) {
           C_vec(k - 1) = C_vec(k - 2) + exp_C_vec(k - 2);
         }
       }
       
       return C_vec; 
   
}


inline  stan::math::var raw_C_to_C_log_det_J_lp_var( Eigen::Matrix<stan::math::var, -1, 1> raw_C, // parameter - log_diffs
                                                    bool softplus) {
         
       const int n_cutpoints = raw_C.size();
       stan::math::var log_det_J = 0.0;
       if (softplus == true) log_det_J += (stan::math::log_inv_logit(raw_C.segment(1, n_cutpoints - 1))).sum(); 
       else                  log_det_J += raw_C.segment(1, n_cutpoints - 1).sum();
       return log_det_J;
  
}


//// ---------------------------------------------------------------------------------------
//// Induced-Dirichlet ("ind_dir") log-density function:
//// NOTE: You can use this for both ind_dir PRIORS and ind_dir MODELS:
//// NOTE: adapted from: Betancourt et al (see: https://betanalpha.github.io/assets/case_studies/ordinal_regression.html),
//// HOWEVER my version has a (much) more computationally efficient (lower-trianglar) Jacobian computation, which is 
//// mathematically still valid. 
////
inline  stan::math::var induced_dirichlet_given_C_lpdf_var( Eigen::Matrix<stan::math::var, -1, 1> p_ord,
                                                           Eigen::Matrix<stan::math::var, -1, 1> C,
                                                           Eigen::Matrix<stan::math::var, -1, 1> alpha,
                                                           bool use_probit_link
) {
 
       const int n_cat = p_ord.size();
       const int n_thr = n_cat - 1;
       
       stan::math::var log_prob = 0.0;
       
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
inline  Eigen::Matrix<stan::math::var, -1, 1> cumul_probs_to_ord_probs_var( Eigen::Matrix<stan::math::var, -1, 1> cumul_probs) {
      
       const int n_thr = cumul_probs.size();
       const int n_cat = n_thr + 1;
       Eigen::Matrix<stan::math::var, -1, 1> ord_probs(n_cat);
      
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
inline  Eigen::Matrix<stan::math::var, -1, 1> ord_probs_to_cumul_probs_var( Eigen::Matrix<stan::math::var, -1, 1> ord_probs) {
  
       const int n_cat = ord_probs.size();
       const int n_thr = n_cat - 1;
       Eigen::Matrix<stan::math::var, -1, 1> cumul_probs(n_thr);
      
       cumul_probs(0) = ord_probs(0);
       for (int k = 2; k <= n_thr; ++k) {
         cumul_probs(k - 1) = cumul_probs(k - 2) + ord_probs(k - 1); // since probs are INCREASING with k 
       }
      
       return cumul_probs;
   
}


////
//// Convert from cumul_probs -> C (w/ induced dirichlet):
////
inline  Eigen::Matrix<stan::math::var, -1, 1> ID_cumul_probs_to_C( Eigen::Matrix<stan::math::var, -1, 1> cumul_probs, 
                                                                  bool use_probit_link
) {
  
       const int n_thr = cumul_probs.size();
       const int n_cat = n_thr + 1;
       Eigen::Matrix<stan::math::var, -1, 1> C(n_thr);
      
       for (int k = 1; k <= n_thr; ++k) {
         if (cumul_probs(k - 1) < 1e-300) {
           C(k - 1) = -37.5; ////  prob = 1e-38; 
         } else if (cumul_probs(k - 1) > 0.99999999999999999) {
           C(k - 1) = +8.20; ////  prob = 0.9999999999999;
         } else {
           if (use_probit_link == true) C(k - 1) = stan::math::inv_Phi(cumul_probs(k - 1)); 
           else                         C(k - 1) = stan::math::logit(cumul_probs(k - 1)); 
         }
       }
      
       return C;
  
} 


































inline  Eigen::Matrix<stan::math::var, -1, 1> lb_ub_lp( stan::math::var  y,
                                                               stan::math::var lb,
                                                               stan::math::var ub
) {
   
       stan::math::var target = 0.0;
       
       // stan::math::var val   = (lb  + (ub  - lb) * stan::math::inv_logit(y)) ;
       stan::math::var val   =  lb +  (ub - lb) *  0.5 * (1.0 +  stan::math::tanh(y));
       
       // target += stan::math::log(ub - lb) + stan::math::log_inv_logit(y) + stan::math::log1m_inv_logit(y);
       target +=  stan::math::log(ub - lb) - log(2)  + stan::math::log1m(stan::math::square(stan::math::tanh(y)));
       // target += stan::math::log(ub - lb) - log(2.0) + 2.0 * (log(2.0) - y - stan::math::log1p_exp(-2.0 * y));
       
       Eigen::Matrix<stan::math::var, -1, 1 > out_mat  = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(2);
       out_mat(0) = target;
       out_mat(1) = val;
       
       return(out_mat) ;
   
} 
 
 
 
inline  Eigen::Matrix<double, -1, 1> lb_ub_lp_dbl( double y,
                                                          double lb,
                                                          double ub
) {
       
       double target = 0.0;
       
       // double val   = (lb  + (ub  - lb) * stan::math::inv_logit(y)) ;
       double val   =  lb +  (ub - lb) *  0.5 * (1.0 +  stan::math::tanh(y));
       
       // target += stan::math::log(ub - lb) + stan::math::log_inv_logit(y) + stan::math::log1m_inv_logit(y);
       target +=  stan::math::log(ub - lb) - log(2)  + stan::math::log1m(stan::math::square(stan::math::tanh(y)));
       // target += stan::math::log(ub - lb) - log(2.0) + 2.0 * (log(2.0) - y - stan::math::log1p_exp(-2.0 * y));
       
       Eigen::Matrix<double, -1, 1 > out_mat  = Eigen::Matrix<double, -1, 1 >::Zero(2);
       out_mat(0) = target;
       out_mat(1) = val;
       
       return(out_mat) ;
       
} 
 
 
inline  Eigen::Matrix<stan::math::var, -1, 1> lb_ub_lp_vec_y( Eigen::Matrix<stan::math::var, -1, 1 > y,
                                                                     Eigen::Matrix<stan::math::var, -1, 1 > lb,
                                                                     Eigen::Matrix<stan::math::var, -1, 1 > ub
) {
     
       stan::math::var target = 0.0;
    
       //   stan::math::var val   =  lb +  (ub - lb) *  0.5 * (1 +  stan::math::tanh(y));
       Eigen::Matrix<stan::math::var, -1, 1 >  vec =   (lb.array() +  (ub.array()  - lb.array() ) *  0.5 * (1.0 +  stan::math::tanh(y).array() )).matrix();
       
       //  target += (stan::math::log( (ub.array() - lb.array()).matrix()).array() + stan::math::log_inv_logit(y).array() + stan::math::log1m_inv_logit(y).array()).matrix().sum() ;
       target +=  (stan::math::log((ub.array() - lb.array()).matrix()).array() - log(2)  +  stan::math::log1m(stan::math::square(stan::math::tanh(y))).array()).matrix().sum();
       // target += ( stan::math::log((ub.array() - lb.array()).matrix()).array() - log(2.0)
       //              + 2.0 * ( log(2.0) - y.array() - stan::math::log1p_exp(-2.0 * y.array()) ) ).matrix().sum();
                                 
       Eigen::Matrix<stan::math::var, -1, 1 > out_mat  = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(vec.rows() + 1);
       out_mat(0) = target;
       out_mat.segment(1, vec.rows()) = vec;
       
       return(out_mat);
   
}


inline  Eigen::Matrix<double, -1, 1> lb_ub_lp_vec_y_dbl( Eigen::Matrix<double, -1, 1 > y,
                                                                Eigen::Matrix<double, -1, 1 > lb,
                                                                Eigen::Matrix<double, -1, 1 > ub
) {
   
       double target = 0.0  ;
       //   double val   =  lb +  (ub - lb) *  0.5 * (1 +  stan::math::tanh(y));
       Eigen::Matrix<double, -1, 1 >  vec =   (lb.array() +  (ub.array()  - lb.array() ) *  0.5 * (1.0 +  stan::math::tanh(y).array() )).matrix();
       
       //  target += (stan::math::log( (ub.array() - lb.array()).matrix()).array() + stan::math::log_inv_logit(y).array() + stan::math::log1m_inv_logit(y).array()).matrix().sum() ;
       target +=  (stan::math::log((ub.array() - lb.array()).matrix()).array() - log(2)  +  stan::math::log1m(stan::math::square(stan::math::tanh(y))).array()).matrix().sum();
       // target += ( stan::math::log((ub.array() - lb.array()).matrix()).array() - log(2.0)
       //               + 2.0 * ( log(2.0) - y.array() - stan::math::log1p_exp(-2.0 * y.array()) ) ).matrix().sum();                
       
       Eigen::Matrix<double, -1, 1 > out_mat  = Eigen::Matrix<double, -1, 1 >::Zero(vec.rows() + 1);
       out_mat(0) = target;
       out_mat.segment(1, vec.rows()) = vec;
       
       return(out_mat);
   
}




inline  Eigen::Matrix<double, -1, -1> Pinkney_LDL_bounds_opt_dbl( int K,
                                                                         Eigen::Matrix<double, -1, -1> lb,
                                                                         Eigen::Matrix<double, -1, -1> ub,
                                                                         Eigen::Matrix<double, -1, -1> Omega_theta_unconstrained_array,
                                                                         Eigen::Matrix<int, -1, -1> known_values_indicator,
                                                                         Eigen::Matrix<double, -1, -1> known_values
) {
   
       double target = 0.0;
       
       Eigen::Matrix<double, -1, -1> L = Eigen::Matrix<double, -1, -1>::Zero(K, K);
       
       for (int i = 0; i < K; ++i) {
         L(i, i) = 1.0;
       } 
       
       Eigen::Matrix<double, -1, 1> D = Eigen::Matrix<double, -1, 1>::Zero(K);
       
       D(0) = 1.0;
       for (int i = 2; i < K + 1; ++i) {
         if (known_values_indicator(i-1, 0) == 1) {
           L(i-1, 0) = known_values(i-1, 0);
         } else {
           stan::math::check_greater("Pinkney_LDL_bounds_opt_dbl", "first-column upper bound", ub(i-1, 0), lb(i-1, 0));
           Eigen::VectorXd lb_ub_lp_outs = lb_ub_lp_dbl(Omega_theta_unconstrained_array(i-1, 0), lb(i-1, 0), ub(i-1, 0));
           target += lb_ub_lp_outs(0); // Only free entries contribute a transform Jacobian.
           L(i-1, 0) = lb_ub_lp_outs(1);
         }
         D(i-1) = 1.0 - stan::math::square(L(i-1, 0)); // Compute after fixing the entry, including D(1).
         stan::math::check_positive("Pinkney_LDL_bounds_opt_dbl", "first-column LDL remainder", D(i-1));
         stan::math::check_finite("Pinkney_LDL_bounds_opt_dbl", "first-column LDL remainder", D(i-1));
       }
       
       for (int i = 3; i < K + 1; ++i) {
         
             D(i-1) = 1.0 - stan::math::square(L(i-1, 0)); // checked
             Eigen::Matrix<double, 1, -1> row_vec_rep = stan::math::rep_row_vector(1.0 - stan::math::square(L(i-1, 0)), i - 2); // checked
             L.row(i - 1).segment(1, i - 2) = row_vec_rep; // checked
             double l_ij_old = L(i-1, 1); // checked
             
             for (int j = 2; j < i; ++j) {
               
                     double b1 = stan::math::dot_product(L.row(j - 1).head(j - 1), (D.head(j - 1).transpose().array() * L.row(i - 1).head(j - 1).array()).matrix()); // checked
                     
                     Eigen::Matrix<double, -1, 1> low_vec_to_max(2);
                     Eigen::Matrix<double, -1, 1> up_vec_to_min(2);
                     //// ---- POSITIVE-DEFINITENESS BOUND on x := L(i,j) * D(j).
                     //// R(i,j) = b1 + x, and the running Schur complement is updated by
                     ////     l_ij_old -= D(j) * L(i,j)^2 = x^2 / D(j),
                     //// so keeping it positive requires x^2 < l_ij_old * D(j), i.e. |x| < sqrt(l_ij_old * D(j)).
                     //// This previously read sqrt(l_ij_old) * D(j), which is the correct bound multiplied by
                     //// sqrt(D(j)) and therefore TOO TIGHT whenever D(j) < 1: valid correlation matrices were
                     //// unreachable. E.g. for R = [1,.8,.2; .8,1,.6; .2,.6,1] (det 0.152 > 0) the (3,2) entry
                     //// needs x = 0.44, the old bound capped |x| at 0.353 and the correct bound is 0.588.
                     low_vec_to_max(0) = - stan::math::sqrt(l_ij_old * D(j-1));
                     low_vec_to_max(1) = (lb(i-1, j-1) - b1);
                     up_vec_to_min(0) =   stan::math::sqrt(l_ij_old * D(j-1));
                     up_vec_to_min(1) = (ub(i-1, j-1) - b1);

                     double low = stan::math::max(low_vec_to_max);   // new
                     double up = stan::math::min(up_vec_to_min);   // new

                     if (known_values_indicator(i-1, j-1) == 1) {
                       //// R(i,j) = b1 + L(i,j)*D(j), so pinning R(i,j) to a known value must subtract the
                       //// b1 already accumulated from columns < j. Without it the realised entry is
                       //// b1 + known_value, not known_value; the M3 / M4 zero patterns happen to have
                       //// b1 = 0 so they were unaffected, but any other fixed pattern is wrong.
                       L(i-1, j-1) = (known_values(i-1, j-1) - b1) / D(j-1);
                     } else {
                       stan::math::check_greater("Pinkney_LDL_bounds_opt_dbl", "upper residual bound", up, low);
                       Eigen::Matrix<double, -1, 1> lb_ub_lp_outs = lb_ub_lp_dbl(Omega_theta_unconstrained_array(i-1, j-1), low, up);
                       target += lb_ub_lp_outs.eval()(0);
                       double x = lb_ub_lp_outs.eval()(1);    // logit bounds
                       L(i-1, j-1) = x / D(j-1);
                       target += -0.5 * stan::math::log(D(j-1));
                       // target += - stan::math::log(D(j-1));
                     } 
                     
                     // l_ij_old *= 1.0 - (D(j-1) * stan::math::square(L(i-1, j-1))) / l_ij_old;
                     l_ij_old -= D(j-1) * stan::math::square(L(i-1, j-1));
                     stan::math::check_positive("Pinkney_LDL_bounds_opt_dbl", "LDL remainder", l_ij_old);
                     stan::math::check_finite("Pinkney_LDL_bounds_opt_dbl", "LDL remainder", l_ij_old);
               
             }
              
             D(i-1) = l_ij_old;
         
       }
       //L(0, 0) = 1;
       
       //////////// output
       Eigen::Matrix<double, -1, -1> out_mat = Eigen::Matrix<double, -1, -1>::Zero(1 + K, K);
       
       out_mat(0, 0) = target;
       // out_mat.block(1, 0, n, n) = L;
       out_mat.block(1, 0, K, K) = stan::math::diag_post_multiply(L, stan::math::sqrt(D));
       
       return(out_mat);
   
}




inline  Eigen::Matrix<stan::math::var, -1, -1> Pinkney_LDL_bounds_opt(  int K,
                                                                               Eigen::Matrix<stan::math::var, -1, -1 >  lb,
                                                                               Eigen::Matrix<stan::math::var, -1, -1 >  ub,
                                                                               Eigen::Matrix<stan::math::var, -1, -1 >  Omega_theta_unconstrained_array,
                                                                               Eigen::Matrix<int, -1, -1 >  known_values_indicator,
                                                                               Eigen::Matrix<double, -1, -1 >  known_values
) { 
     
       stan::math::var target = 0.0 ;
       
       Eigen::Matrix<stan::math::var, -1, -1 > L = Eigen::Matrix<stan::math::var, -1, -1 >::Zero(K, K);
       
       for (int i = 0; i < K; ++i) {
         L(i, i) = 1.0;
       }
       
       Eigen::Matrix<stan::math::var, -1, 1 >  D = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(K);
       
       D(0) = 1.0;
       for (int i = 2; i < K + 1; ++i) {
         if (known_values_indicator(i-1, 0) == 1) {
           L(i-1, 0) = stan::math::to_var(known_values(i-1, 0));
         } else {
           stan::math::check_greater("Pinkney_LDL_bounds_opt", "first-column upper bound", ub(i-1, 0), lb(i-1, 0));
           Eigen::Matrix<stan::math::var, -1, 1> lb_ub_lp_outs = lb_ub_lp(Omega_theta_unconstrained_array(i-1, 0), lb(i-1, 0), ub(i-1, 0));
           target += lb_ub_lp_outs(0); // Only free entries contribute a transform Jacobian.
           L(i-1, 0) = lb_ub_lp_outs(1);
         }
         D(i-1) = 1.0 - stan::math::square(L(i-1, 0)); // Compute after fixing the entry, including D(1).
         stan::math::check_positive("Pinkney_LDL_bounds_opt", "first-column LDL remainder", D(i-1));
         stan::math::check_finite("Pinkney_LDL_bounds_opt", "first-column LDL remainder", D(i-1));
       }
       
       for (int i = 3; i < K + 1; ++i) {
             
             D(i-1) = 1.0 - stan::math::square(L(i-1, 0)) ; // checked
             Eigen::Matrix<stan::math::var, 1, -1 >  row_vec_rep = stan::math::rep_row_vector(1.0 - stan::math::square(L(i-1, 0)), i - 2) ; // checked
             L.row(i - 1).segment(1, i - 2) = row_vec_rep; // checked
             stan::math::var   l_ij_old = L(i-1, 1); // checked
             
             for (int j = 2; j < i; ++j) {
               
                   stan::math::var b1 = stan::math::dot_product(L.row(j - 1).head(j - 1), (D.head(j - 1).transpose().array() * L.row(i - 1).head(j - 1).array() ).matrix()  ) ; // checked
                   
                   Eigen::Matrix<stan::math::var, -1, 1 > low_vec_to_max(2);
                   Eigen::Matrix<stan::math::var, -1, 1 > up_vec_to_min(2);
                   //// ---- POSITIVE-DEFINITENESS BOUND on x := L(i,j) * D(j). See the dbl version above:
                   //// l_ij_old -= x^2 / D(j) must stay positive, so |x| < sqrt(l_ij_old * D(j)).
                   //// This previously read sqrt(l_ij_old) * D(j) - too tight by a factor sqrt(D(j)) - and
                   //// excluded valid correlation matrices.
                   low_vec_to_max(0) = - stan::math::sqrt(l_ij_old * D(j-1)) ;
                   low_vec_to_max(1) =   (lb(i-1, j-1) - b1) ;
                   up_vec_to_min(0) =    stan::math::sqrt(l_ij_old * D(j-1)) ;
                   up_vec_to_min(1) =    (ub(i-1, j-1) - b1)  ;

                   stan::math::var  low =    stan::math::max( low_vec_to_max   );   // new
                   stan::math::var  up  =    stan::math::min( up_vec_to_min    );   // new

                   if (known_values_indicator(i-1, j-1) == 1) {
                     //// R(i,j) = b1 + L(i,j)*D(j): subtract the b1 accumulated from columns < j, or the
                     //// realised entry is b1 + known_value rather than known_value.
                     L(i-1, j-1) =  (stan::math::to_var(known_values(i-1, j-1)) - b1) /  D(j-1)  ;
                   } else {
                     stan::math::check_greater("Pinkney_LDL_bounds_opt", "upper residual bound", up, low);
                     Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_outs = lb_ub_lp(Omega_theta_unconstrained_array(i-1, j-1), low,  up) ;
                     target += lb_ub_lp_outs.eval()(0);
                     stan::math::var x = lb_ub_lp_outs.eval()(1);    // logit bounds
                     L(i-1, j-1)  = x / D(j-1) ;
                     target += -0.5 * stan::math::log(D(j-1)) ;
                     // target += -  stan::math::log(D(j-1)) ;
                   }
                   
                   // l_ij_old *= 1.0 - (D(j-1) *  stan::math::square(L(i-1, j-1) )) / l_ij_old;
                   l_ij_old -= D(j-1) * stan::math::square(L(i-1, j-1));
                     stan::math::check_positive("Pinkney_LDL_bounds_opt", "LDL remainder", l_ij_old);
                     stan::math::check_finite("Pinkney_LDL_bounds_opt", "LDL remainder", l_ij_old);
               
             }
             
             D(i-1) = l_ij_old;
         
       }
       //L(0, 0) = 1;
       
       //////////// output
       Eigen::Matrix<stan::math::var, -1, -1 > out_mat = Eigen::Matrix<stan::math::var, -1, -1 >::Zero(1 + K , K);
       
       out_mat(0, 0) = target;
       // out_mat.block(1, 0, n, n) = L;
       out_mat.block(1, 0, K, K) = stan::math::diag_post_multiply(L, stan::math::sqrt(D));
       
       return(out_mat);
   
}
 
 
 
 
 


// =============================================================================
// Pinkney C-vine correlation parameterization with per-element bounds
// =============================================================================
//
// Based on Sean Pinkney's corr_cvine2.stan (private repo), adapted for
// BayesMVP with:
//   - Per-element bounds (lb[i,j], ub[i,j] matrices on correlations)
//   - Known values support (known_values_indicator, known_values)
//   - Both stan::math::var (AD) and double versions
//   - Eigen C++ matching existing BayesMVP interface
//
// KEY DIFFERENCE FROM LDL:
//   LDL parameterizes Cholesky entries L[i,j] with a shrinking remainder
//   l_ij_old that couples all parameters within each row through a ratio
//   (x/D[j]) and multiplicative update. The C-vine parameterizes *partial
//   correlations* P[i,j] directly. Each partial correlation controls one
//   degree of freedom via an affine map C[i,j] = c + a*P[i,j], where c and
//   a depend on previous P values but the coupling is through products of
//   sqrt(1 - P^2) rather than through a collapsing remainder.
//
// RETURN FORMAT:
//   (K+1) x K matrix:
//     out(0, 0) = target (log Jacobian of unconstrained y -> C)
//     out(0, 1) = log_det_C = sum log(1 - P[i,j]^2), for LKJ prior
//     out.block(1, 0, K, K) = C (the correlation matrix, NOT Cholesky factor)
//
// LKJ PRIOR APPLICATION (user applies separately):
//   target += (eta - 1.0) * log_det_C;
//   where log_det_C = out(0, 1)
//
//   This is algebraically equivalent to lkj_corr_lpdf(C | eta) up to const.
//   det(C) = prod(1 - P[i,j]^2) is a known identity for the C-vine.
//
// CHOLESKY FACTOR (if needed for GHK simulator etc):
//   L = chol(C)  -- user computes this separately
//   For K=6 this is ~36 flops, negligible vs likelihood.
//
// UNCONSTRAINED PARAMETER LAYOUT in Omega_theta_unconstrained_array (K x K):
//   Column 0, rows 1..K-1 : first-column correlations C(j, 0) -> P(0, j)
//   Lower triangle (j, i) for 1 <= i < j <= K-1 : partial correlation P(i, j)
//   Same total count K(K-1)/2 as LDL.
//   NOTE: Different positions than LDL for inner entries. Starting values
//   from an LDL run cannot be reused directly.
//
// =============================================================================


// =============================================================================
// Scalar lb_ub_lp: unconstrained y -> bounded [lb, ub] via tanh
// Returns 2-vector: (0) = log Jacobian contribution, (1) = bounded value
// These should match your existing lb_ub_lp_dbl / lb_ub_lp functions.
// Included here for self-containedness; remove if already defined.
// =============================================================================
inline  Eigen::Matrix<double, -1, 1> lb_ub_lp_cvine_dbl( double y, 
                                                        double lb,
                                                        double ub
) {
  
       Eigen::Matrix<double, -1, 1> out(2);
       double tanh_y = std::tanh(y);
       out(1) = lb + (ub - lb) * 0.5 * (1.0 + tanh_y);
       
       // Jacobian: log(ub - lb) - log(2) + log(1 - tanh^2(y))
       // Stable form: log(ub - lb) - log(2) + 2*(log(2) - y - log1p(exp(-2y)))
       out(0) = std::log(ub - lb) - std::log(2.0) + 2.0 * (std::log(2.0) - std::abs(y) - stan::math::log1p_exp(-2.0 * std::abs(y)));
       
       return out;
   
}


inline  Eigen::Matrix<stan::math::var, -1, 1> lb_ub_lp_cvine( stan::math::var y, stan::math::var lb, stan::math::var ub) {
   
       Eigen::Matrix<stan::math::var, -1, 1> out(2);
       stan::math::var tanh_y = stan::math::tanh(y);
       
       out(1) = lb + (ub - lb) * 0.5 * (1.0 + tanh_y);
       out(0) = stan::math::log(ub - lb) - std::log(2.0) + 2.0 * (std::log(2.0) - stan::math::abs(y) - stan::math::log1p_exp(-2.0 * stan::math::abs(y)));
       
       return out;
   
}


// =============================================================================
// Vectorised lb_ub for column 0 (all entries share the same transform type)
// Returns (K-1+1)-vector: (0) = total log Jacobian, (1..K-1) = bounded values
// =============================================================================
inline  Eigen::Matrix<double, -1, 1> lb_ub_lp_vec_cvine_dbl(  const Eigen::Matrix<double, -1, 1> &y,
                                                             const Eigen::Matrix<double, -1, 1> &lb_vec,
                                                             const Eigen::Matrix<double, -1, 1> &ub_vec
) {
      
       int N = y.size();
       Eigen::Matrix<double, -1, 1> out(1 + N);
       double target = 0.0;
       
       for (int n = 0; n < N; ++n) {
         double tanh_y = std::tanh(y(n));
         out(1 + n) = lb_vec(n) + (ub_vec(n) - lb_vec(n)) * 0.5 * (1.0 + tanh_y);
         target += std::log(ub_vec(n) - lb_vec(n)) - std::log(2.0)
           + 2.0 * (std::log(2.0) - std::abs(y(n)) - stan::math::log1p_exp(-2.0 * std::abs(y(n))));
       }
       
       out(0) = target;
       return out;
   
}


inline  Eigen::Matrix<stan::math::var, -1, 1> lb_ub_lp_vec_cvine( const Eigen::Matrix<stan::math::var, -1, 1> &y,
                                                                 const Eigen::Matrix<stan::math::var, -1, 1> &lb_vec,
                                                                 const Eigen::Matrix<stan::math::var, -1, 1> &ub_vec
) {
  
       int N = y.size();
       Eigen::Matrix<stan::math::var, -1, 1> out(1 + N);
       stan::math::var target = 0.0;
       
       for (int n = 0; n < N; ++n) {
         stan::math::var tanh_y = stan::math::tanh(y(n));
         out(1 + n) = lb_vec(n) + (ub_vec(n) - lb_vec(n)) * 0.5 * (1.0 + tanh_y);
         target += stan::math::log(ub_vec(n) - lb_vec(n)) - std::log(2.0)
           + 2.0 * (std::log(2.0) - stan::math::abs(y(n)) - stan::math::log1p_exp(-2.0 * stan::math::abs(y(n))));
       }
       
       out(0) = target;
       return out;
   
}
 
 
// =============================================================================
// C-VINE: DOUBLE VERSION
// =============================================================================
inline  Eigen::Matrix<double, -1, -1> Pinkney_cvine_bounds_opt_dbl( int K,
                                                                   Eigen::Matrix<double, -1, -1> lb,
                                                                   Eigen::Matrix<double, -1, -1> ub,
                                                                   Eigen::Matrix<double, -1, -1> Omega_theta_unconstrained_array,
                                                                   Eigen::Matrix<int, -1, -1> known_values_indicator,
                                                                   Eigen::Matrix<double, -1, -1> known_values
) {
   
       double target = 0.0;
       double log_det_C = 0.0;
       
       // C = correlation matrix (output), P = partial correlations (vine structure)
       Eigen::Matrix<double, -1, -1> C = Eigen::Matrix<double, -1, -1>::Identity(K, K);
       Eigen::Matrix<double, -1, -1> P = Eigen::Matrix<double, -1, -1>::Zero(K, K);
       
       // =========================================================================
       // Column 0: raw correlations C(j, 0) for j = 1, ..., K-1
       // These are also the vine level-0 "partial correlations" (no conditioning)
       // P(0, j) = C(0, j) = C(j, 0)
       // Unconstrained params: Omega_theta_unconstrained_array(j, 0) for j=1..K-1
       // Bounds: lb(j, 0) <= C(j, 0) <= ub(j, 0)
       // =========================================================================
       Eigen::Matrix<double, -1, 1> first_col_y = Omega_theta_unconstrained_array.col(0).segment(1, K - 1);
       Eigen::Matrix<double, -1, 1> first_col_lb = lb.col(0).segment(1, K - 1);
       Eigen::Matrix<double, -1, 1> first_col_ub = ub.col(0).segment(1, K - 1);
       
       // Process column 0 with known-value support
       for (int j = 1; j < K; ++j) {
         
           if (known_values_indicator(j, 0) == 1) {
             P(0, j) = known_values(j, 0);
           } else {
             Eigen::Matrix<double, -1, 1> out = lb_ub_lp_cvine_dbl(
               Omega_theta_unconstrained_array(j, 0), lb(j, 0), ub(j, 0));
             target += out(0);
             P(0, j) = out(1);
           }
           C(j, 0) = P(0, j);
           C(0, j) = P(0, j);
           // Accumulate log_det_C: each partial corr contributes log(1 - P^2)
           log_det_C += std::log(1.0 - P(0, j) * P(0, j));
         
       }
       
       // =========================================================================
       // Inner entries: vine levels 1, ..., K-2  (0-indexed)
       //
       // At vine level i (0-indexed), for each pair (i, j) with j > i:
       //   P(i, j) = partial correlation of tests i and j given tests 0,...,i-1
       //   C(i, j) = full correlation = c + a * P(i, j)
       //   where c = reconstruction(P=0) and a = prod(b2) > 0
       //
       // Unconstrained param: Omega_theta_unconstrained_array(j, i) [lower tri]
       // Correlation bounds: lb(j, i) <= C(i, j) <= ub(j, i)
       //   -> translated to partial correlation bounds:
       //      P_low = max(-1+eps, (lb_C - c) / a)
       //      P_up  = min( 1-eps, (ub_C - c) / a)
       // =========================================================================
       for (int i = 1; i <= K - 2; ++i) {
         
           for (int j = i + 1; j <= K - 1; ++j) {
               
               // --- Compute b1, b2 vectors (size i) ---
               // b1[k] = P(m, i) * P(m, j)     for m = i-1, ..., 0
               // b2[k] = sqrt((1 - P(m,i)^2)(1 - P(m,j)^2))
               Eigen::Matrix<double, -1, 1> b1_vec(i);
               Eigen::Matrix<double, -1, 1> b2_vec(i);
               int m = i;
               for (int k = 0; k < i; ++k) {
                 m -= 1;   // m goes from i-1 down to 0
                 b1_vec(k) = P(m, i) * P(m, j);
                 b2_vec(k) = std::sqrt((1.0 - P(m, i) * P(m, i)) * (1.0 - P(m, j) * P(m, j)));
               }
               
               // --- Compute c = reconstruction(P[i,j] = 0) and a = prod(b2) ---
               // Since C[i,j] = c + a * P[i,j], this gives us the affine map.
               double c_val = 0.0;
               for (int k = 0; k < i; ++k) {
                 c_val = b1_vec(k) + c_val * b2_vec(k);
               }
               double a_val = 1.0;
               for (int k = 0; k < i; ++k) {
                 a_val *= b2_vec(k);
               }
               
               if (known_values_indicator(j, i) == 1) {
                 // Known correlation C(i, j) -> compute forced partial correlation
                 P(i, j) = (known_values(j, i) - c_val) / a_val;
               } else {
                 // Translate correlation bounds to partial correlation bounds
                 // C = c + a*P,  a > 0  =>  P = (C - c) / a
                 double lb_C = lb(j, i);
                 double ub_C = ub(j, i);
                 double P_low = std::max(-1.0 + 1e-10, (lb_C - c_val) / a_val);
                 double P_up  = std::min( 1.0 - 1e-10, (ub_C - c_val) / a_val);
                 
                 // Map unconstrained -> bounded partial correlation
                 Eigen::Matrix<double, -1, 1> out = lb_ub_lp_cvine_dbl( Omega_theta_unconstrained_array(j, i), P_low, P_up);
                 target += out(0);
                 P(i, j) = out(1);
                 
                 // Vine Jacobian: log |dC/dP| = log(a) for this entry
                 target += std::log(a_val);
               }
               
               // Accumulate log_det_C
               log_det_C += std::log(1.0 - P(i, j) * P(i, j));
               
               // --- Reconstruct C(i, j) = c + a * P(i, j) ---
               C(i, j) = c_val + a_val * P(i, j);
               C(j, i) = C(i, j);
               
           }
         
       }
       
       // =========================================================================
       // Pack output
       // =========================================================================
       Eigen::Matrix<double, -1, -1> out_mat = Eigen::Matrix<double, -1, -1>::Zero(1 + K, K);
       out_mat(0, 0) = target;
       out_mat(0, 1) = log_det_C;
       out_mat.block(1, 0, K, K) = stan::math::cholesky_decompose(C);
       
       return out_mat;
       
}
 

// =============================================================================
// C-VINE: stan::math::var (AUTODIFF) VERSION
// =============================================================================
inline Eigen::Matrix<stan::math::var, -1, -1> Pinkney_cvine_bounds_opt(  int K,
                                                                         Eigen::Matrix<stan::math::var, -1, -1> lb,
                                                                         Eigen::Matrix<stan::math::var, -1, -1> ub,
                                                                         Eigen::Matrix<stan::math::var, -1, -1> Omega_theta_unconstrained_array,
                                                                         Eigen::Matrix<int, -1, -1> known_values_indicator,
                                                                         Eigen::Matrix<double, -1, -1> known_values
) {
  
       stan::math::var target = 0.0;
       stan::math::var log_det_C = 0.0;
       
       Eigen::Matrix<stan::math::var, -1, -1> C = Eigen::Matrix<double, -1, -1>::Identity(K, K).cast<stan::math::var>();
       Eigen::Matrix<stan::math::var, -1, -1> P = Eigen::Matrix<double, -1, -1>::Zero(K, K).cast<stan::math::var>();
       
       // =========================================================================
       // Column 0
       // =========================================================================
       for (int j = 1; j < K; ++j) {
         
           if (known_values_indicator(j, 0) == 1) {
             P(0, j) = known_values(j, 0);
           } else {
             Eigen::Matrix<stan::math::var, -1, 1> out = lb_ub_lp_cvine(
               Omega_theta_unconstrained_array(j, 0), lb(j, 0), ub(j, 0));
             target += out(0);
             P(0, j) = out(1);
           }
           
           C(j, 0) = P(0, j);
           C(0, j) = P(0, j);
           log_det_C += stan::math::log1m(stan::math::square(P(0, j)));
         
       }
       
       // =========================================================================
       // Inner entries: vine levels 1, ..., K-2
       // =========================================================================
       for (int i = 1; i <= K - 2; ++i) {
         
           for (int j = i + 1; j <= K - 1; ++j) {
                 
               Eigen::Matrix<stan::math::var, -1, 1> b1_vec(i);
               Eigen::Matrix<stan::math::var, -1, 1> b2_vec(i);
               int m = i;
               for (int k = 0; k < i; ++k) {
                 m -= 1;
                 b1_vec(k) = P(m, i) * P(m, j);
                 b2_vec(k) = stan::math::sqrt(
                   (1.0 - stan::math::square(P(m, i)))
                   * (1.0 - stan::math::square(P(m, j))));
               }
               
               // c = reconstruction with P[i,j] = 0
               stan::math::var c_val = 0.0;
               for (int k = 0; k < i; ++k) {
                 c_val = b1_vec(k) + c_val * b2_vec(k);
               }
               // a = prod(b2)
               stan::math::var a_val = 1.0;
               for (int k = 0; k < i; ++k) {
                 a_val *= b2_vec(k);
               }
               
               if (known_values_indicator(j, i) == 1) {
                 P(i, j) = (known_values(j, i) - c_val) / a_val;
               } else {
                 // Translate correlation bounds to partial correlation bounds
                 stan::math::var lb_C = lb(j, i);
                 stan::math::var ub_C = ub(j, i);
                 stan::math::var P_low = stan::math::fmax(stan::math::var(-1.0 + 1e-10), (lb_C - c_val) / a_val);
                 stan::math::var P_up  = stan::math::fmin(stan::math::var( 1.0 - 1e-10), (ub_C - c_val) / a_val);
                 
                 Eigen::Matrix<stan::math::var, -1, 1> out = lb_ub_lp_cvine( Omega_theta_unconstrained_array(j, i), P_low, P_up);
                 target += out(0);
                 P(i, j) = out(1);
                 
                 // Vine Jacobian
                 target += stan::math::log(a_val);
               }
               
               log_det_C += stan::math::log1m(stan::math::square(P(i, j)));
               
               C(i, j) = c_val + a_val * P(i, j);
               C(j, i) = C(i, j);
             
           }
           
       }
       
       // =========================================================================
       // Pack output
       // =========================================================================
       Eigen::Matrix<stan::math::var, -1, -1> out_mat = Eigen::Matrix<double, -1, -1>::Zero(1 + K, K).cast<stan::math::var>();
       out_mat(0, 0) = target;
       out_mat(0, 1) = log_det_C;
       out_mat.block(1, 0, K, K) = stan::math::cholesky_decompose(C);
       
       return out_mat;
   
 }
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
// =============================================================================
// Pinkney Cholesky-corr "proposal" parameterization  (NO bounds, NO known values), from:
// https://discourse.mc-stan.org/t/updated-cholesky-corr-parameterization-testing/38827
// =============================================================================
//
// Direct translation of Sean Pinkney's cholesky_corr_constrain_proposal_lp (Stan).
// Maps unconstrained y (length K*(K-1)/2) DIRECTLY to a Cholesky factor L of a
// correlation matrix + log-Jacobian. Row i (1-based) construction:
//
//   y_star  = the (i-1) raw params for row i
//   dsy     = dot_self(y_star)
//   alpha_r = 1 / (dsy + 1)                      <- diagonal entry, always > 0
//   gamma   = sqrt(dsy + 2) * alpha_r
//   L[i, 1:i-1] = gamma * y_star ;  L[i, i] = alpha_r
//   target += 0.5*(i-2)*log(dsy + 2) - i*log1p(dsy)
//
// Unit row norms by construction: gamma^2*dsy + alpha_r^2
//   = ((dsy+2)*dsy + 1)/(dsy+1)^2 = 1.
//
// PARAMETER LAYOUT: y is stacked ROW-WISE over the strict lower triangle
//   (row 2: 1 element, row 3: 2 elements, ...) -- i.e. the SAME flat order as
//   the existing theta packing loop (for i, for j < i), so per class c just pass
//   Omega_raw_vec.segment(c * dim_choose_2, dim_choose_2) directly; no
//   fn_convert_std_vec_of_corrs_to_3d_array needed. NB: VALUES are not
//   transferable from LDL / C-vine runs (different transform), layout only.
//
// RETURN FORMAT (matches Pinkney_LDL_bounds_opt):
//   (K+1) x K matrix:
//     out(0, 0)             = target (log Jacobian of unconstrained y -> L)
//     out.block(1, 0, K, K) = L  (Cholesky factor of the correlation matrix)
//
// LKJ PRIOR (user applies separately, same as existing AD-block usage):
//   target += stan::math::lkj_corr_cholesky_lpdf(L, eta);
//
// KEY DIFFERENCES FROM LDL / C-VINE VERSIONS:
//   - NO per-element bounds and NO known-value pinning: lb_corr / ub_corr /
//     known_values_indicator / corr_force_positive CANNOT be honoured here.
//     Do not route models that need those through this transform.
//   - Row i of L depends ONLY on that row's own y_star block => dL/dy is
//     BLOCK-DIAGONAL by row. For the num_diff deriv_L_wrt_unc_full loop:
//     perturbing one y changes ONE row of L only (all other rows' derivs are
//     exactly 0), so the FD fill can skip every other row -- and
//     deriv_L_wrt_unc_full is mostly structural zeros.
// =============================================================================
inline  Eigen::Matrix<double, -1, -1> Pinkney_chol_corr_proposal_lp_dbl( int K,
                                                                                const Eigen::Matrix<double, -1, 1> y  // length K*(K-1)/2
) {
  
       double target = 0.0;
       
       Eigen::Matrix<double, -1, -1> L = Eigen::Matrix<double, -1, -1>::Identity(K, K);
       
       int counter = 0;   // 0-based (Stan version starts at 1)
       
       for (int i = 2; i < K + 1; ++i) {   // i = 1-based row index (matches Stan / LDL loop style)
         
             const int n_row = i - 1;   // number of raw params in this row
             
             Eigen::Matrix<double, -1, 1> y_star = y.segment(counter, n_row);
             
             double dsy     = stan::math::dot_self(y_star); // y_star.squaredNorm();   // dot_self(y_star)
             double alpha_r = 1.0 / (dsy + 1.0);
             double gamma   = stan::math::sqrt(dsy + 2.0) * alpha_r;
             
             L.row(i - 1).head(n_row) = gamma * y_star.transpose();
             L(i - 1, i - 1) = alpha_r;
             
             target += 0.5 * (i - 2) * stan::math::log(dsy + 2.0) - i * stan::math::log1p(dsy);
             // target += 0.6931471805599453 + 0.5 * (i - 3) * std::log(dsy + 2.0) - i * std::log1p(dsy);
             
             counter += n_row;
         
       }
       
       //////////// output
       Eigen::Matrix<double, -1, -1> out_mat = Eigen::Matrix<double, -1, -1>::Zero(1 + K, K);
       
       out_mat(0, 0) = target;
       out_mat.block(1, 0, K, K) = L;
       
       return(out_mat);
   
}



inline  Eigen::Matrix<stan::math::var, -1, -1> Pinkney_chol_corr_proposal_lp( int K,
                                                                                     const Eigen::Matrix<stan::math::var, -1, 1> y  // length K*(K-1)/2
) {
  
       stan::math::var target = 0.0;
       
       Eigen::Matrix<stan::math::var, -1, -1> L = Eigen::Matrix<double, -1, -1>::Identity(K, K).cast<stan::math::var>();
       
       int counter = 0;   // 0-based (Stan version starts at 1)
       
       for (int i = 2; i < K + 1; ++i) {   // i = 1-based row index (matches Stan / LDL loop style)
         
             const int n_row = i - 1;   // number of raw params in this row
             
             Eigen::Matrix<stan::math::var, -1, 1> y_star = y.segment(counter, n_row);
             
             stan::math::var dsy     = stan::math::dot_self(y_star);
             stan::math::var alpha_r = 1.0 / (dsy + 1.0);
             stan::math::var gamma   = stan::math::sqrt(dsy + 2.0) * alpha_r;
             
             L.row(i - 1).head(n_row) = gamma * y_star.transpose();
             L(i - 1, i - 1) = alpha_r;
             
             target += 0.5 * (i - 2) * stan::math::log(dsy + 2.0) - i * stan::math::log1p(dsy);
             // target += 0.6931471805599453 + 0.5 * (i - 3) * stan::math::log(dsy + 2.0) - i * stan::math::log1p(dsy);
             
             counter += n_row;
         
       }
       
       //////////// output
       Eigen::Matrix<stan::math::var, -1, -1> out_mat = Eigen::Matrix<double, -1, -1>::Zero(1 + K, K).cast<stan::math::var>();
       
       out_mat(0, 0) = target;
       out_mat.block(1, 0, K, K) = L;
       
       return(out_mat);
   
}
 
 
 
 
 
 
 
 
 
// =============================================================================
// MASTER correlation-transform dispatch
// =============================================================================
//
// Inspects the constraint pattern (lb / ub / known_values) ONCE and routes to
// the recommended Pinkney transform (per S.P. corr-bounds blog + Discourse):
//
//   code 0: fully UNRESTRICTED (all lb == -1, ub == +1, no known values)
//             -> Pinkney_chol_corr_proposal_lp   ("proposal", Discourse 38827)
//   code 1: every free cell SIGN-restricted ([0,1] or [-1,0]), no known values
//             -> tri-sign transform  ** NOT YET PORTED -> falls back to LDL **
//   code 2: known values ALL ZERO, free cells unrestricted
//             -> "tri" subspace transform ** NOT YET PORTED -> falls back to LDL **
//   code 3: anything else (finite boxes, nonzero fixed values, mixtures)
//             -> Pinkney_LDL_bounds_opt  ("entry-wise")
//
// RETURN FORMAT: identical across all branches, (K+1) x K:
//     out(0, 0)             = target (log Jacobian, unconstrained -> L)
//     out.block(1, 0, K, K) = L (Cholesky factor of correlation matrix)
//   so existing callers (Chol_Schur_outs usage) work unchanged.
//   All branches pair with lkj_corr_cholesky_lpdf(L, eta) applied downstream.
//
// NB: the DISPATCH DECISION is computed from the DOUBLE bound matrices in BOTH
//     the dbl and var versions (var version takes lb/ub as doubles and converts
//     internally) => the two versions CANNOT dispatch differently.
//
// NB: raw-parameter VALUES are NOT transferable across transforms (same layout,
//     different maps). Saved warmup states / inits from an LDL run are garbage
//     under code 0 and vice versa.
// =============================================================================
inline  int fn_detect_corr_constraint_type( const int K,
                                                   const Eigen::Matrix<double, -1, -1> lb,
                                                   const Eigen::Matrix<double, -1, -1> ub,
                                                   const Eigen::Matrix<int, -1, -1> known_values_indicator,
                                                   const Eigen::Matrix<double, -1, -1> known_values
) {
  
       bool any_known = false;
       bool all_known_zero = true;
       bool all_free_unrestricted = true;
       bool all_free_sign = true;   //// every free cell exactly [0,1] or [-1,0]
       
       for (int i = 1; i < K; ++i) {
         for (int j = 0; j < i; ++j) {
           
               if (known_values_indicator(i, j) == 1) {
                 
                     any_known = true;
                     if (known_values(i, j) != 0.0) all_known_zero = false;
                 
               } else {
                 
                     const double l = lb(i, j);
                     const double u = ub(i, j);
                     const bool cell_unrestricted = (l == -1.0) && (u ==  1.0);
                     const bool cell_pos_sign     = (l ==  0.0) && (u ==  1.0);
                     const bool cell_neg_sign     = (l == -1.0) && (u ==  0.0);
                     ////
                     if (!cell_unrestricted)                  all_free_unrestricted = false;
                     if (!(cell_pos_sign || cell_neg_sign))   all_free_sign = false;
                 
               }
           
         }
       }
       
       if (!any_known && all_free_unrestricted)                    return 0;  //// -> proposal
       if (!any_known && all_free_sign)                            return 1;  //// -> tri-sign (fallback: LDL)
       if ( any_known && all_known_zero && all_free_unrestricted)  return 2;  //// -> "tri"    (fallback: LDL)
       return 3;                                                              //// -> LDL
   
}


//// helper: extract flat raw vector (row-wise strict lower triangle, i.e. the SAME
//// (i, j<i) order as the existing theta packing / num_diff cnt_2 loops):
inline  Eigen::Matrix<double, -1, 1> fn_corr_unc_mat_to_flat_dbl( const int K,
                                                                         const Eigen::Matrix<double, -1, -1> Omega_unc
) {
       Eigen::Matrix<double, -1, 1> y(K * (K - 1) / 2);
       int counter = 0;
       for (int i = 1; i < K; ++i) {
         for (int j = 0; j < i; ++j) {
           y(counter) = Omega_unc(i, j);
           counter += 1;
         }
       }
       
       return y;
}


inline  Eigen::Matrix<stan::math::var, -1, 1> fn_corr_unc_mat_to_flat_var( const int K,
                                                                                  const Eigen::Matrix<stan::math::var, -1, -1> Omega_unc
) {
       Eigen::Matrix<stan::math::var, -1, 1> y(K * (K - 1) / 2);
       int counter = 0;
       for (int i = 1; i < K; ++i) {
         for (int j = 0; j < i; ++j) {
           y(counter) = Omega_unc(i, j);
           counter += 1;
         }
       }
       
       return y;
}


// =============================================================================
// MASTER: DOUBLE VERSION
// =============================================================================
inline  Eigen::Matrix<double, -1, -1> Pinkney_corr_master_dbl( int K,
                                                                      const Eigen::Matrix<double, -1, -1> lb,
                                                                      const Eigen::Matrix<double, -1, -1> ub,
                                                                      const Eigen::Matrix<double, -1, -1> Omega_theta_unconstrained_array,
                                                                      const Eigen::Matrix<int, -1, -1> known_values_indicator,
                                                                      const Eigen::Matrix<double, -1, -1> known_values
) {
  
       // const int constraint_type = fn_detect_corr_constraint_type(K, lb, ub, known_values_indicator, known_values);
       // 
       // if (constraint_type == 0) {   //// fully unrestricted -> "proposal" transform
       //   
       //       Eigen::Matrix<double, -1, 1> y = fn_corr_unc_mat_to_flat_dbl(K, Omega_theta_unconstrained_array);
       //       return Pinkney_chol_corr_proposal_lp_dbl(K, y);
       //   
       // }
       
       //// constraint_type 1 (tri-sign) + 2 ("tri" zeros-only): recommended transforms
       //// NOT yet ported to C++ -> LDL fallback (handles both correctly, just less efficiently).
       //// constraint_type 3: LDL is the recommended transform.
       return Pinkney_LDL_bounds_opt_dbl( K,
                                          lb,
                                          ub,
                                          Omega_theta_unconstrained_array,
                                          known_values_indicator,
                                          known_values);
   
}


// =============================================================================
// MASTER: stan::math::var (AUTODIFF) VERSION
// NB: takes lb / ub as DOUBLES (converts internally for the LDL branch) so the
//     dispatch decision is guaranteed identical to the dbl version.
// =============================================================================
inline  Eigen::Matrix<stan::math::var, -1, -1> Pinkney_corr_master( int K,
                                                                           const Eigen::Matrix<double, -1, -1> lb,
                                                                           const Eigen::Matrix<double, -1, -1> ub,
                                                                           const Eigen::Matrix<stan::math::var, -1, -1> Omega_theta_unconstrained_array,
                                                                           const Eigen::Matrix<int, -1, -1> known_values_indicator,
                                                                           const Eigen::Matrix<double, -1, -1> known_values
) {
  
       // const int constraint_type = fn_detect_corr_constraint_type(K, lb, ub, known_values_indicator, known_values);
       // 
       // if (constraint_type == 0) {   //// fully unrestricted -> "proposal" transform
       //   
       //       Eigen::Matrix<stan::math::var, -1, 1> y = fn_corr_unc_mat_to_flat_var(K, Omega_theta_unconstrained_array);
       //       return Pinkney_chol_corr_proposal_lp(K, y);
       //   
       // }
       
       Eigen::Matrix<stan::math::var, -1, -1> lb_var = stan::math::to_var(lb);
       Eigen::Matrix<stan::math::var, -1, -1> ub_var = stan::math::to_var(ub);
       
       return Pinkney_LDL_bounds_opt( K,
                                      lb_var,
                                      ub_var,
                                      Omega_theta_unconstrained_array,
                                      known_values_indicator,
                                      known_values);
   
}



 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

 
 // 
 // 
 // 
 // // function for use in the log-posterior function (i.e. the function to calculate gradients for)
 // Eigen::Matrix<stan::math::var, -1, -1>	  fn_calculate_cutpoints_AD(
 //     Eigen::Matrix<stan::math::var, -1, 1> log_diffs, //this is aparameter (col vec)
 //     stan::math::var first_cutpoint, // this is constant
 //     int K) {
 //   
 //   Eigen::Matrix<stan::math::var, -1, -1> cutpoints_set_full(K+1, 1);
 //   
 //   cutpoints_set_full(0,0) = -1000;
 //   cutpoints_set_full(1,0) = first_cutpoint;
 //   cutpoints_set_full(K,0) = +1000;
 //   
 //   for (int k=2; k < K; ++k)
 //     cutpoints_set_full(k,0) =     cutpoints_set_full(k-1,0)  + (exp(log_diffs(k-2))) ;
 //   
 //   return cutpoints_set_full; // output is a parameter to use in the log-posterior function to be differentiated
 // }
 // 
 // 
 // 
 // 
 // 
  
 // inline std::array<Eigen::Matrix<stan::math::var, -1, -1>, 2>      array_of_mats_test_2d_var( int n_rows,
 //                                                                                              int n_cols) {
 //   
 //   
 //   std::array<Eigen::Matrix<stan::math::var, -1, -1 >, 2> my_2d_array;
 //   Eigen::Matrix<stan::math::var, -1, -1> my_mat = Eigen::Matrix<stan::math::var, -1, -1>::Zero(n_rows, n_cols);
 //   
 //   for (int c = 0; c < 2; ++c) {
 //     my_2d_array[c] = my_mat;
 //   }
 //   
 //   return my_2d_array;
 //   
 // }
 // 
 // 
 
 
 
 
 
 
 
 
 
inline std::vector<Eigen::Matrix<stan::math::var, -1, -1>> vec_of_mats_var(int n_rows, int n_cols, int n_mats) {
       
       std::vector<Eigen::Matrix<stan::math::var, -1, -1>> my_vec;
       
       my_vec.reserve(n_mats);  
        
       for (int c = 0; c < n_mats; ++c) {
         my_vec.emplace_back(Eigen::Matrix<stan::math::var, -1, -1>::Zero(n_rows, n_cols));
       }
        
       return my_vec;
   
}
 
 


 // 
 // 
 // 
 // 
 // 
 // 
 // // input vector, outputs upper-triangular 3d array of corrs- double
 // std::vector<Eigen::Matrix<stan::math::var, -1, -1> >  fn_convert_Eigen_vec_of_corrs_to_3d_array_var(
 //     Eigen::Matrix<stan::math::var, -1, -1  >  input_vec,
 //     int n_rows,
 //     int n_arrays) {
 //   
 //   std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >   output_array = vec_of_mats_test_var(n_rows, n_rows, n_arrays); // 1d vector to output
 //   
 //   int k = 0;
 //   for (int c = 0; c < n_arrays; ++c) {
 //     for (int i = 1; i < n_rows; ++i)  {
 //       for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
 //         output_array[c](i,j) =  input_vec(i);
 //         k += 1;
 //       }
 //     }
 //   }
 //   
 //   return output_array; // output is a parameter to use in the log-posterior function to be differentiated
 // }
 // 
 // 
 // 
 // 
 // 
 // 
 
 
 
 
 
 
 
 
 // // convert std vec to eigen vec - var
 // Eigen::Matrix<stan::math::var, -1, 1> std_vec_to_Eigen_vec_var(std::vector<stan::math::var> std_vec) {
 //   
 //   Eigen::Matrix<stan::math::var, -1, 1>  Eigen_vec(std_vec.size());
 //   
 //   for (int i = 0; i < std_vec.size(); ++i) {
 //     Eigen_vec(i) = std_vec[i];
 //   }
 //   
 //   return(Eigen_vec);
 // }
 // 
 // 
 
 
 
 
 
inline std::vector<stan::math::var> Eigen_vec_to_std_vec_var(Eigen::Matrix<stan::math::var, -1, 1> Eigen_vec) {

       std::vector<stan::math::var>  std_vec(Eigen_vec.rows(), 0.0);
    
       for (int i = 0; i < Eigen_vec.rows(); ++i) {
         std_vec[i] = Eigen_vec(i);
       }
    
       return(std_vec);
   
}



 
 
 
 
 
 
 
 // 
 // std::vector<std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > > vec_of_vec_of_mats_test_var(int n_rows,
 //                                                                                                 int n_cols,
 //                                                                                                 int n_mats_inner,
 //                                                                                                 int n_mats_outer) {
 //   
 //   /// need to figure out more efficient way to do this + make work for all types easily (not just double)
 //   std::vector<std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > > my_vec_of_vecs(n_mats_outer);
 //   Eigen::Matrix<stan::math::var, -1, -1 > mat_sizes(n_rows, n_cols);
 //   
 //   
 //   
 //   for (int c1 = 0; c1 < n_mats_outer; ++c1) {
 //     std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > my_vec(n_mats_inner);
 //     my_vec_of_vecs[c1] = my_vec;
 //     for (int c2 = 0; c2 < n_mats_inner; ++c2) {
 //       my_vec_of_vecs[c1][c2] = mat_sizes;
 //       for (int i = 0; i < n_rows; ++i) {
 //         for (int j = 0; j < n_cols; ++j) {
 //           my_vec_of_vecs[c1][c2](i,j) = 0;
 //         }
 //       }
 //     }
 //   }
 //   
 //   
 //   return(my_vec_of_vecs);
 //   
 // }
 // 
 // 
 // 
 
 
 
 
 
// input vector, outputs upper-triangular 3d array of corrs- double
inline std::vector<Eigen::Matrix<stan::math::var, -1, -1>> fn_convert_std_vec_of_corrs_to_3d_array_var( std::vector<stan::math::var> input_vec,
                                                                                                        int n_rows,
                                                                                                        int n_arrays) {

       std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >   output_array = vec_of_mats_var(n_rows, n_rows, n_arrays); // 1d vector to output
    
       int k = 0;
       for (int c = 0; c < n_arrays; ++c) {
         for (int i = 1; i < n_rows; ++i)  {
           for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
             output_array[c](i,j) =  input_vec[k];
             k = k + 1;
           }
         }
       }
    
       return output_array; // output is a parameter to use in the log-posterior function to be differentiated
   
}




 
 
// input vector, outputs upper-triangular 3d array of corrs- double
inline std::vector<Eigen::Matrix<stan::math::var, -1, -1>> fn_convert_Eigen_vec_of_corrs_to_3d_array_var( Eigen::Matrix<stan::math::var, -1, 1> input_vec,
                                                                                                          int n_rows,
                                                                                                          int n_arrays) {
    
       std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >   output_array = vec_of_mats_var(n_rows, n_rows, n_arrays); // 1d vector to output
        
       int k = 0;
       for (int c = 0; c < n_arrays; ++c) {
         for (int i = 1; i < n_rows; ++i)  {
           for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
             output_array[c](i, j) =  input_vec(i);
             k = k + 1;
           }
         }
       } 
       
       return output_array; // output is a parameter to use in the log-posterior function to be differentiated
   
}
 
 
 
 
 
 
 
 
 
inline stan::math::var  inv_Phi_approx_var( stan::math::var x )  {
   stan::math::var m_logit_p =   stan::math::log( 1.0/x  - 1.0)  ;
   stan::math::var x_i = -0.3418*m_logit_p;
   stan::math::var asinh_stuff_div_3 =  0.33333333333333331483 *  stan::math::log( x_i  +   stan::math::sqrt(  stan::math::fma(x_i, x_i, 1.0) ) )  ;          // now do arc_sinh part
   stan::math::var exp_x_i =   stan::math::exp(asinh_stuff_div_3);
   return  2.74699999999999988631 * (  stan::math::fma(exp_x_i, exp_x_i , -1.0) / exp_x_i ) ;  //   now do sinh parth part
}


inline Eigen::Matrix<stan::math::var, -1, 1  >  inv_Phi_approx_var( Eigen::Matrix<stan::math::var, -1, 1  > x )  {
   Eigen::Matrix<stan::math::var, -1, 1  > x_i = -0.3418*stan::math::log( ( 1.0/x.array()  - 1.0).matrix() );
   Eigen::Matrix<stan::math::var, -1, 1  > asinh_stuff_div_3 =  0.33333333333333331483 *  stan::math::log( x_i  +   stan::math::sqrt(  stan::math::fma(x_i, x_i, 1.0) ) )  ;          // now do arc_sinh part
   Eigen::Matrix<stan::math::var, -1, 1  > exp_x_i =   stan::math::exp(asinh_stuff_div_3);
   return  2.74699999999999988631 * (  stan::math::fma(exp_x_i, exp_x_i , -1.0).array() / exp_x_i.array() ) ;  //   now do sinh parth part
}


inline stan::math::var  inv_Phi_approx_from_logit_prob_var( stan::math::var logit_p )  {
   stan::math::var x_i = 0.3418*logit_p;
   stan::math::var asinh_stuff_div_3 =  0.33333333333333331483 *  stan::math::log( x_i  +   stan::math::sqrt(  stan::math::fma(x_i, x_i, 1.0) ) )  ;          // now do arc_sinh part
   stan::math::var exp_x_i =   stan::math::exp(asinh_stuff_div_3);
   return  2.74699999999999988631 * (  stan::math::fma(exp_x_i, exp_x_i , -1.0) / exp_x_i ) ;  //   now do sinh parth part
}







inline Eigen::Matrix<stan::math::var, -1, 1  >   log_sum_exp_2d_Stan_var(   Eigen::Matrix<stan::math::var, -1, 2  >  x )  {

   int N = x.rows();
   Eigen::Matrix<stan::math::var, -1, 2  > rowwise_maxes_2d_array(N, 2);
   rowwise_maxes_2d_array.col(0) = x.array().rowwise().maxCoeff().matrix();
   rowwise_maxes_2d_array.col(1) = rowwise_maxes_2d_array.col(0);

   return      rowwise_maxes_2d_array.col(0)   +   stan::math::log(    stan::math::exp( (x  -  rowwise_maxes_2d_array).matrix() ).rowwise().sum().array().abs().matrix()   ).matrix()    ;

}


 
  
 
 
// inline Eigen::Matrix<double, -1, 1  > fn_log_sum_exp_2d_double(     Eigen::Ref<Eigen::Matrix<double, -1, 2>>  x,    // Eigen::Matrix<double, -1, 2> &x, 
//                                                                       const std::string &vect_type = "Eigen",
//                                                                       const bool &skip_checks = false) {
//   
//   
//   {
//     if (vect_type == "Eigen") {
//       return  log_sum_exp_2d_Eigen_double(x);
//     } else if (vect_type == "Stan") {
//       return  log_sum_exp_2d_Stan_double(x);
//     } else if (vect_type == "AVX2") {
//       if (skip_checks == false)   return  fast_log_sum_exp_2d_AVX2_double(x);
//       else                        return  fast_log_sum_exp_2d_AVX2_double(x);
//     } else if (vect_type == "AVX512") {
//       if (skip_checks == false)   return  fast_log_sum_exp_2d_AVX512_double(x);
//       else                        return  fast_log_sum_exp_2d_AVX512_double(x);
//     } else if (vect_type == "Loop") {
//       //if (skip_checks == false)   return  fast_log_sum_exp_2d_double(x);
//       // else                        return  fast_log_sum_exp_2d_double(x);
//     } else { 
//       std::stringstream os;
//       os << "Invalid input argument to log_sum_exp_2d_double"  ;
//       throw std::invalid_argument( os.str() );
//     }
//     
//   }
//   
//    return  x.col(0);
//   
//   
// }





    
 
// 
// 
//  // function for use in the log-posterior function (i.e. the function to calculate gradients for)
//  Eigen::Matrix<stan::math::var, -1, -1>	  fn_calculate_cutpoints_AD(
//      Eigen::Matrix<stan::math::var, -1, 1> log_diffs, //this is aparameter (col vec)
//      stan::math::var first_cutpoint, // this is constant
//      int K) {
// 
//    Eigen::Matrix<stan::math::var, -1, -1> cutpoints_set_full(K+1, 1);
// 
//    cutpoints_set_full(0,0) = -1000;
//    cutpoints_set_full(1,0) = first_cutpoint;
//    cutpoints_set_full(K,0) = +1000;
// 
//    for (int k=2; k < K; ++k)
//      cutpoints_set_full(k,0) =     cutpoints_set_full(k-1,0)  + (exp(log_diffs(k-2))) ;
// 
//    return cutpoints_set_full; // output is a parameter to use in the log-posterior function to be differentiated
//  }
// 
// 






 // // function for use in the log-posterior function (i.e. the function to calculate gradients for)
 // // [[Rcpp::export]]
 // Eigen::Matrix<double, -1, -1>	  fn_calculate_cutpoints(
 //     Eigen::Matrix<double, -1, 1> log_diffs, //this is a parameter (col vec)
 //     double first_cutpoint, // this is constant
 //     int K) {
 // 
 //   Eigen::Matrix<double, -1, -1> cutpoints_set_full(K+1, 1);
 // 
 //   cutpoints_set_full(0,0) = -1000;
 //   cutpoints_set_full(1,0) = first_cutpoint;
 //   cutpoints_set_full(K,0) = +1000;
 // 
 //   for (int k=2; k < K; ++k)
 //     cutpoints_set_full(k,0) =     cutpoints_set_full(k-1,0)  + (exp(log_diffs(k-2))) ;
 // 
 //   return cutpoints_set_full; // output is a parameter to use in the log-posterior function to be differentiated
 // }
 // 
 // 
 // 
 // 



// 
// 
// inline std::vector<Eigen::Matrix<double, -1, -1>>      vec_of_mats_test_2d( int n_rows,
//                                                                      int n_cols) {
// 
// 
//   std::vector<Eigen::Matrix<double, -1, -1>> my_2d_vec(2);
//   Eigen::Matrix<double, -1, -1> my_mat = Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
// 
//   for (int c = 0; c < 2; ++c) {
//     my_2d_vec[c] = my_mat;
//   }
// 
//   return(my_2d_vec);
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
//  
// 
// 
// 
// 
// inline std::array<Eigen::Matrix<double, -1, -1>, 1>      array_of_mats_test_1d( int n_rows,
//                                                                          int n_cols) {
// 
// 
//   std::array<Eigen::Matrix<double, -1, -1 >, 1> my_1d_array;
//   Eigen::Matrix<double, -1, -1> my_mat = Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
//   my_1d_array[0] = my_mat;
// 
//   return my_1d_array;
// 
// }
// 
// 
// 
// inline std::array<Eigen::Matrix<double, -1, -1>, 2>      array_of_mats_test_2d( int n_rows,
//                                                                          int n_cols) {
// 
// 
//    std::array<Eigen::Matrix<double, -1, -1 >, 2> my_2d_array;
//    Eigen::Matrix<double, -1, -1> my_mat = Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
// 
//    for (int c = 0; c < 2; ++c) {
//      my_2d_array[c] = my_mat;
//    }
// 
//    return my_2d_array;
// 
//  }



// 
// inline std::array<Eigen::Matrix<stan::math::var, -1, -1>, 2>      array_of_mats_test_2d_var( int n_rows,
//                                                                                              int n_cols) {
//   
//   
//   std::array<Eigen::Matrix<stan::math::var, -1, -1 >, 2> my_2d_array;
//   Eigen::Matrix<stan::math::var, -1, -1> my_mat = Eigen::Matrix<stan::math::var, -1, -1>::Zero(n_rows, n_cols);
//   
//   for (int c = 0; c < 2; ++c) {
//     my_2d_array[c] = my_mat;
//   }
//   
//   return my_2d_array;
//   
// }
// 


// 
// 
// inline std::array<Eigen::Matrix<float, -1, -1>, 2>      array_of_mats_test_2d_float( int n_rows,
//                                                                                int n_cols) {
//   
//   
//   std::array<Eigen::Matrix<float, -1, -1 >, 2> my_2d_array;
//   Eigen::Matrix<float, -1, -1> my_mat = Eigen::Matrix<float, -1, -1>::Zero(n_rows, n_cols);
//   
//   for (int c = 0; c < 2; ++c) {
//     my_2d_array[c] = my_mat;
//   }
//   
//   return my_2d_array;
//   
// }
// 
// 
// 
// 
// 
// 
// 
// inline  std::vector<Eigen::Matrix<double, -1, -1 > > vec_of_mats_test(int n_rows,
//                                                                int n_cols,
//                                                                int n_mats) {
// 
// 
//    std::vector<Eigen::Matrix<double, -1, -1 > > my_vec(n_mats);
//    Eigen::Matrix<double, -1, -1> my_mat = Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
// 
//    for (int c = 0; c < n_mats; ++c) {
//      my_vec[c] = my_mat;
//    }
// 
//    return(my_vec);
// 
//  }
// 
// 
// 
//  
// 
// 
// 
//  // [[Rcpp::export]]
//  std::vector<Eigen::Matrix<int, -1, -1 > > vec_of_mats_test_int(int n_rows,
//                                                                 int n_cols,
//                                                                 int n_mats) {
// 
// 
//    std::vector<Eigen::Matrix<int, -1, -1 > > my_vec(n_mats);
//    Eigen::Matrix<int, -1, -1 > mats  =   Eigen::Matrix<int, -1, -1>::Zero(n_rows, n_cols);
// 
//    for (int c = 0; c < n_mats; ++c) {
//      my_vec[c] = mats;
//    }
// 
//    return(my_vec);
// 
//  }
// 
// 
// 
// 
//  // [[Rcpp::export]]
//  std::vector<Eigen::Matrix<bool, -1, -1 > > vec_of_mats_test_bool(int n_rows,
//                                                                   int n_cols,
//                                                                   int n_mats) {
// 
// 
//    std::vector<Eigen::Matrix<bool, -1, -1 > > my_vec(n_mats);
//    Eigen::Matrix<bool, -1, -1 > mats(n_rows, n_cols);
// 
//    for (int c = 0; c < n_mats; ++c) {
//      my_vec[c] = mats;
//    }
// 
//    return(my_vec);
// 
//  }
// 
// 
// 
// 
// 
//  
// 
// 
//  std::vector<Eigen::Matrix<float, -1, -1 > > vec_of_mats_test_float(int n_rows,
//                                                                     int n_cols,
//                                                                     int n_mats) {
// 
//    std::vector<Eigen::Matrix<float, -1, -1 > > my_vec(n_mats);
//    Eigen::Matrix<float, -1, -1 > mats  =   Eigen::Matrix<float, -1, -1>::Zero(n_rows, n_cols);
// 
//    for (int c = 0; c < n_mats; ++c) {
//      my_vec[c] = mats;
//    }
// 
//    return(my_vec);
// 
// 
//  }
// 
// 
// 
// 





 // std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > vec_of_mats_test_var(int n_rows,
 //                                                                            int n_cols,
 //                                                                            int n_mats) {
 // 
 //   std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > my_vec(n_mats);
 //   Eigen::Matrix<stan::math::var, -1, -1 > mats  =   Eigen::Matrix<stan::math::var, -1, -1>::Zero(n_rows, n_cols);
 // 
 //   for (int c = 0; c < n_mats; ++c) {
 //     my_vec[c] = mats;
 //   }
 // 
 //   return(my_vec);
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
 // // input vector, outputs upper-triangular 3d array of corrs- double
 // std::vector<Eigen::Matrix<stan::math::var, -1, -1> >  fn_convert_Eigen_vec_of_corrs_to_3d_array_var(
 //                                                                                                                         Eigen::Matrix<stan::math::var, -1, -1  >  input_vec,
 //                                                                                                                         int n_rows,
 //                                                                                                                         int n_arrays) {
 // 
 //   std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >   output_array = vec_of_mats_test_var(n_rows, n_rows, n_arrays); // 1d vector to output
 // 
 //   int k = 0;
 //   for (int c = 0; c < n_arrays; ++c) {
 //     for (int i = 1; i < n_rows; ++i)  {
 //       for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
 //         output_array[c](i,j) =  input_vec(i);
 //         k += 1;
 //       }
 //     }
 //   }
 // 
 //   return output_array; // output is a parameter to use in the log-posterior function to be differentiated
 // }
 // 
 // 
 // 
 // 
 // 
 // 



 



// 
//  // convert std vec to eigen vec - var
//  Eigen::Matrix<stan::math::var, -1, 1> std_vec_to_Eigen_vec_var(std::vector<stan::math::var> std_vec) {
// 
//    Eigen::Matrix<stan::math::var, -1, 1>  Eigen_vec(std_vec.size());
// 
//    for (int i = 0; i < std_vec.size(); ++i) {
//      Eigen_vec(i) = std_vec[i];
//    }
// 
//    return(Eigen_vec);
//  }
// 


// 
//  // convert std vec to eigen vec - double
//  // [[Rcpp::export]]
//  Eigen::Matrix<double, -1, 1> std_vec_to_Eigen_vec(std::vector<double> std_vec) {
// 
//    Eigen::Matrix<double, -1, 1>  Eigen_vec(std_vec.size());
// 
//    for (int i = 0; i < std_vec.size(); ++i) {
//      Eigen_vec(i) = std_vec[i];
//    }
// 
//    return(Eigen_vec);
//  }
// 
//  // [[Rcpp::export]]
//  std::vector<double> Eigen_vec_to_std_vec(Eigen::Matrix<double, -1, 1> Eigen_vec) {
// 
//    std::vector<double>  std_vec(Eigen_vec.rows());
// 
//    for (int i = 0; i < Eigen_vec.rows(); ++i) {
//      std_vec[i] = Eigen_vec(i);
//    }
// 
//    return(std_vec);
//  }


 // std::vector<stan::math::var> Eigen_vec_to_std_vec_var(Eigen::Matrix<stan::math::var, -1, 1> Eigen_vec) {
 // 
 //   std::vector<stan::math::var>  std_vec(Eigen_vec.rows());
 // 
 //   for (int i = 0; i < Eigen_vec.rows(); ++i) {
 //     std_vec[i] = Eigen_vec(i);
 //   }
 // 
 //   return(std_vec);
 // }
 // 
 // 
 // 
 // 
 // 
 // 



 // std::vector<std::vector<Eigen::Matrix<double, -1, -1 > > > vec_of_vec_of_mats_test(int n_rows,
 //                                                                                    int n_cols,
 //                                                                                    int n_mats_inner,
 //                                                                                    int n_mats_outer) {
 // 
 //   /// need to figure out more efficient way to do this + make work for all types easily (not just double)
 //   std::vector<std::vector<Eigen::Matrix<double, -1, -1 > > > my_vec_of_vecs(n_mats_outer);
 //   Eigen::Matrix<double, -1, -1 > mat_sizes(n_rows, n_cols);
 // 
 // 
 // 
 //   for (int c1 = 0; c1 < n_mats_outer; ++c1) {
 //     std::vector<Eigen::Matrix<double, -1, -1 > > my_vec(n_mats_inner);
 //     my_vec_of_vecs[c1] = my_vec;
 //     for (int c2 = 0; c2 < n_mats_inner; ++c2) {
 //       my_vec_of_vecs[c1][c2] = mat_sizes;
 //       for (int i = 0; i < n_rows; ++i) {
 //         for (int j = 0; j < n_cols; ++j) {
 //           my_vec_of_vecs[c1][c2](i, j) = 0;
 //         }
 //       }
 //     }
 //   }
 // 
 // 
 //   return(my_vec_of_vecs);
 // 
 // }
 // 
 // 






// 
// 
//  std::vector<std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > > vec_of_vec_of_mats_test_var(int n_rows,
//                                                                                                  int n_cols,
//                                                                                                  int n_mats_inner,
//                                                                                                  int n_mats_outer) {
// 
//    /// need to figure out more efficient way to do this + make work for all types easily (not just double)
//    std::vector<std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > > my_vec_of_vecs(n_mats_outer);
//    Eigen::Matrix<stan::math::var, -1, -1 > mat_sizes(n_rows, n_cols);
// 
// 
// 
//    for (int c1 = 0; c1 < n_mats_outer; ++c1) {
//      std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > my_vec(n_mats_inner);
//      my_vec_of_vecs[c1] = my_vec;
//      for (int c2 = 0; c2 < n_mats_inner; ++c2) {
//        my_vec_of_vecs[c1][c2] = mat_sizes;
//        for (int i = 0; i < n_rows; ++i) {
//          for (int j = 0; j < n_cols; ++j) {
//            my_vec_of_vecs[c1][c2](i,j) = 0;
//          }
//        }
//      }
//    }
// 
// 
//    return(my_vec_of_vecs);
// 
//  }
// 
// 
// 
// 


// 
// 
//  // input vector, outputs upper-triangular 3d array of corrs- double
//  std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >   fn_convert_std_vec_of_corrs_to_3d_array_var(
//      std::vector<stan::math::var>   input_vec,
//      int n_rows,
//      int n_arrays) {
// 
//    std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >   output_array = vec_of_mats_test_var(n_rows, n_rows, n_arrays); // 1d vector to output
// 
//    int k = 0;
//    for (int c = 0; c < n_arrays; ++c) {
//      for (int i = 1; i < n_rows; ++i)  {
//        for (int j = 0; j < i; ++j) { // equiv to 1 to K - 2 in R
//          output_array[c](i,j) =  input_vec[k];
//          k = k + 1;
//        }
//      }
//    }
// 
//    return output_array; // output is a parameter to use in the log-posterior function to be differentiated
//  }
// 
// 
// 






// 
// 
//  
// 
// inline Eigen::Matrix<double, 1, -1>     fn_first_element_neg_rest_pos(      Eigen::Matrix<double, 1, -1>  row_vec    ) {
// 
//    row_vec(0) = - row_vec(0);
// 
//    return(row_vec);
// 
//  }
// 
// 

 


// 
// inline  Eigen::Matrix<stan::math::var, 1, -1>     fn_first_element_neg_rest_pos_var(
//      Eigen::Matrix<stan::math::var, 1, -1>  row_vec
//  ) {
// 
//    row_vec(0) = - row_vec(0);
// 
//    return(row_vec);
// 
//  }
// 



// 
// 
//  std::unique_ptr<size_t[]> get_commutation_unequal_vec
//   (unsigned const n, unsigned const m, bool const transpose){
//    unsigned const nm = n * m,
//      nnm_p1 = n * nm + 1L,
//      nm_pm = nm + m;
//    std::unique_ptr<size_t[]> out(new size_t[nm]);
//    size_t * const o_begin = out.get();
//    size_t idx = 0L;
//    for(unsigned i = 0; i < n; ++i, idx += nm_pm){
//      size_t idx1 = idx;
//      for(unsigned j = 0; j < m; ++j, idx1 += nnm_p1)
//        if(transpose)
//          *(o_begin + idx1 / nm) = (idx1 % nm);
//        else
//          *(o_begin + idx1 % nm) = (idx1 / nm);
//    }
// 
//    return out;
//  }
// 
// // [[Rcpp::export(rng = false)]]
// Rcpp::NumericVector commutation_dot
//   (unsigned const n, unsigned const m, Rcpp::NumericVector x,
//    bool const transpose){
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
// Rcpp::NumericMatrix get_commutation_unequal
//   (unsigned const n, unsigned const m){
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
// Rcpp::NumericMatrix get_commutation_equal(unsigned const m){
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
// // [[Rcpp::export(rng = false)]]
// Eigen::Matrix<double, -1, -1  >  get_commutation(unsigned const n, unsigned const m) {
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
// // [[Rcpp::export(rng = false)]]
// Eigen::Matrix<double, -1, -1  > elimination_matrix(const int &n) {
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
// // [[Rcpp::export(rng = false)]]
// Eigen::Matrix<double, -1, -1  > duplication_matrix(const int &n) {
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
//  Eigen::Matrix<stan::math::var, -1, 1 >                        lb_ub_lp (stan::math::var  y,
//                                                                          stan::math::var lb,
//                                                                          stan::math::var ub) {
// 
//    stan::math::var target = 0 ;
// 
//    // stan::math::var val   = (lb  + (ub  - lb) * stan::math::inv_logit(y)) ;
//    stan::math::var val   =  lb +  (ub - lb) *  0.5 * (1 +  stan::math::tanh(y));
// 
//    // target += stan::math::log(ub - lb) + stan::math::log_inv_logit(y) + stan::math::log1m_inv_logit(y);
//    target +=  stan::math::log(ub - lb) - log(2)  + stan::math::log1m(stan::math::square(stan::math::tanh(y)));
// 
//    Eigen::Matrix<stan::math::var, -1, 1 > out_mat  = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(2);
//    out_mat(0) = target;
//    out_mat(1) = val;
// 
//    return(out_mat) ;
// 
//  }
// 
// 
// 
// 
// 
// 
//  Eigen::Matrix<stan::math::var, -1, 1 >   lb_ub_lp_vec_y (Eigen::Matrix<stan::math::var, -1, 1 > y,
//                                                           Eigen::Matrix<stan::math::var, -1, 1 > lb,
//                                                           Eigen::Matrix<stan::math::var, -1, 1 > ub) {
// 
//    stan::math::var target = 0 ;
// 
// 
//    //   stan::math::var val   =  lb +  (ub - lb) *  0.5 * (1 +  stan::math::tanh(y));
//    Eigen::Matrix<stan::math::var, -1, 1 >  vec =   (lb.array() +  (ub.array()  - lb.array() ) *  0.5 * (1 +  stan::math::tanh(y).array() )).matrix();
// 
//    //  target += (stan::math::log( (ub.array() - lb.array()).matrix()).array() + stan::math::log_inv_logit(y).array() + stan::math::log1m_inv_logit(y).array()).matrix().sum() ;
//    target +=  (stan::math::log((ub.array() - lb.array()).matrix()).array() - log(2)  +  stan::math::log1m(stan::math::square(stan::math::tanh(y))).array()).matrix().sum();
// 
//    Eigen::Matrix<stan::math::var, -1, 1 > out_mat  = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(vec.rows() + 1);
//    out_mat(0) = target;
//    out_mat.segment(1, vec.rows()) = vec;
// 
//    return(out_mat);
// 
//  }
// 
// 


 // //
 // Eigen::Matrix<stan::math::var, -1, -1 >    Pinkney_cholesky_corr_transform_opt( int n,
 //                                                                                  Eigen::Matrix<stan::math::var, -1, -1 >  lb,
 //                                                                                  Eigen::Matrix<stan::math::var, -1, -1 >  ub,
 //                                                                                  Eigen::Matrix<stan::math::var, -1, -1 >  Omega_theta_unconstrained_array,
 //                                                                                  Eigen::Matrix<int, -1, -1 >  known_values_indicator,
 //                                                                                  Eigen::Matrix<double, -1, -1 >  known_values) {
 // 
 // 
 //   stan::math::var target = 0 ;
 // 
 // 
 //   Eigen::Matrix<stan::math::var, -1, -1 > L = Eigen::Matrix<stan::math::var, -1, -1 >::Zero(n, n);
 //   Eigen::Matrix<stan::math::var, -1, 1 > first_col = Omega_theta_unconstrained_array.col(0).segment(1, n - 1);
 // 
 //   Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_outs = lb_ub_lp_vec_y(first_col, lb.col(0), ub.col(0)) ;  // logit bounds
 //   target += lb_ub_lp_vec_y_outs.eval()(0);
 // 
 //   Eigen::Matrix<stan::math::var, -1, 1 >  z = lb_ub_lp_vec_y_outs.segment(1, n - 1);
 //   L.col(0).segment(1, n - 1) = z;
 // 
 //   for (int i = 2; i < n + 1; ++i) {
 //     if (known_values_indicator(i-1, 0) == 1) {
 //       L(i-1, 0) = stan::math::to_var(known_values(i-1, 0));
 //       Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_out = lb_ub_lp(first_col(i-2), lb(i-1, 0), ub(i-1, 0)) ;  // logit bounds
 //       target += - lb_ub_lp_vec_y_out.eval()(0); // undo jac adjustment
 //     }
 //   }
 //   L(1, 1) = stan::math::sqrt(1 - stan::math::square(L(1, 0))) ;
 // 
 //   for (int i = 3; i < n + 1; ++i) {
 // 
 //     Eigen::Matrix<stan::math::var, 1, -1 >  row_vec_rep = stan::math::rep_row_vector(stan::math::sqrt(1 - L(i - 1, 0)* L(i - 1, 0)), i - 1) ;
 //     L.row(i - 1).segment(1, i - 1) = row_vec_rep;
 // 
 //     for (int j = 2; j < i; ++j) {
 // 
 //       stan::math::var   l_ij_old = L(i-1, j-1);
 //       stan::math::var   l_ij_old_x_l_jj = l_ij_old * L(j-1, j-1); // new
 //       stan::math::var b1 = stan::math::dot_product(L.row(j - 1).segment(0, j - 1), L.row(i - 1).segment(0, j - 1)) ;
 //       // stan::math::var b2 = L(j - 1, j - 1) * L(i - 1, j - 1) ; // old
 // 
 //       // stan::math::var  low = std::min(   std::max( b1 - b2, lb(i-1, j-1) / stan::math::abs(L(i-1, j-1)) ), b1 + b2 ); // old
 //       // stan::math::var   up = std::max(   std::min( b1 + b2, ub(i-1, j-1) / stan::math::abs(L(i-1, j-1)) ), b1 - b2 ); // old
 // 
 //       stan::math::var  low =   std::max( -l_ij_old_x_l_jj, (lb(i-1, j-1) - b1)    );   // new
 //       stan::math::var   up =   std::min( +l_ij_old_x_l_jj, (ub(i-1, j-1) - b1)    ); // new
 // 
 //       if (known_values_indicator(i-1, j-1) == 1) {
 //         // L(i-1, j-1) *= ( stan::math::to_var(known_values(i-1, j-1))  - b1) / b2; // old
 //         L(i-1, j-1)  = stan::math::to_var(known_values(i-1, j-1)) / L(j-1, j-1);  // new
 //       } else {
 //         Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_outs = lb_ub_lp(Omega_theta_unconstrained_array(i-1, j-1), low,  up) ;
 //         target += lb_ub_lp_outs.eval()(0); // old
 // 
 //         stan::math::var x = lb_ub_lp_outs.eval()(1);    // logit bounds
 //         target +=  - stan::math::log(L(j-1, j-1)) ;  //  Jacobian for transformation  z -> L_Omega
 // 
 //         //   L(i-1, j-1) *= (x - b1) / b2; // old
 //         L(i-1, j-1)  = x / L(j-1, j-1); //  low + (up - low) * x; // new
 //       }
 // 
 //       //    target += - stan::math::log(L(j-1, j-1)); // old
 // 
 //       stan::math::var   l_ij_new = L(i-1, j-1);
 //       L.row(i - 1).segment(j, i - j).array() *= stan::math::sqrt(  1 -  ( (l_ij_new / l_ij_old) * (l_ij_new / l_ij_old)  )  );
 // 
 //     }
 // 
 //   }
 //   L(0, 0) = 1;
 // 
 //   //////////// output
 //   Eigen::Matrix<stan::math::var, -1, -1 > out_mat = Eigen::Matrix<stan::math::var, -1, -1 >::Zero(1 + n , n);
 // 
 //   out_mat(0, 0) = target;
 //   out_mat.block(1, 0, n, n) = L;
 // 
 //   return(out_mat);
 // 
 // }
 // 
 // 

 
 
 
 
 
 
 
 
 







// 
//  Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor >    Pinkney_LDL_bounds_opt_RM( int K,
//                                                                      Eigen::Matrix<stan::math::var, -1, -1 >  lb,
//                                                                      Eigen::Matrix<stan::math::var, -1, -1 >  ub,
//                                                                      Eigen::Matrix<stan::math::var, -1, -1 >  Omega_theta_unconstrained_array,
//                                                                      Eigen::Matrix<int, -1, -1 >  known_values_indicator,
//                                                                      Eigen::Matrix<double, -1, -1 >  known_values) {
// 
// 
//    stan::math::var target = 0.0;
// 
//    Eigen::Matrix<stan::math::var, -1, 1 > first_col = Omega_theta_unconstrained_array.col(0).segment(1, K - 1);
//    Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_outs = lb_ub_lp_vec_y(first_col, lb.col(0), ub.col(0)) ;  // logit bounds
//    target += lb_ub_lp_vec_y_outs.eval()(0);
//    Eigen::Matrix<stan::math::var, -1, 1 >  z = lb_ub_lp_vec_y_outs.segment(1, K - 1);
// 
//    Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor  > L = Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor  >::Zero(K, K);
// 
//    for (int i = 0; i < K; ++i) {
//      L(i, i) = 1.0;
//    }
// 
//    Eigen::Matrix<stan::math::var, -1, 1 >  D = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(K);
// 
//    D(0) = 1.0;
//    L.col(0).segment(1, K - 1) = z;
//    D(1) = 1.0 -  stan::math::square(L(1, 0)) ;
// 
//    for (int i = 2; i < K + 1; ++i) {
//      if (known_values_indicator(i-1, 0) == 1) {
//        L(i-1, 0) = stan::math::to_var(known_values(i-1, 0));
//        Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_out = lb_ub_lp(first_col(i-2), lb(i-1, 0), ub(i-1, 0)) ;  // logit bounds
//        target += - lb_ub_lp_vec_y_out.eval()(0); // undo jac adjustment
//      }
//    }
// 
//    for (int i = 3; i < K + 1; ++i) {
// 
//      D(i-1) = 1 - stan::math::square(L(i-1, 0)) ;
//      Eigen::Matrix<stan::math::var, 1, -1 >  row_vec_rep = stan::math::rep_row_vector(1 - stan::math::square(L(i-1, 0)), i - 2) ;
//      L.row(i - 1).segment(1, i - 2) = row_vec_rep;
//      stan::math::var   l_ij_old = L(i-1, 1);
// 
//      for (int j = 2; j < i; ++j) {
// 
//        stan::math::var b1 = stan::math::dot_product(L.row(j - 1).head(j - 1), (D.head(j - 1).transpose().array() *  L.row(i - 1).head(j - 1).array() ).matrix()  ) ;
// 
//        Eigen::Matrix<stan::math::var, -1, 1 > low_vec_to_max(2);
//        Eigen::Matrix<stan::math::var, -1, 1 > up_vec_to_min(2);
//        low_vec_to_max(0) = - stan::math::sqrt(l_ij_old) * D(j-1) ;
//        low_vec_to_max(1) =   (lb(i-1, j-1) - b1) ;
//        up_vec_to_min(0) =    stan::math::sqrt(l_ij_old) * D(j-1) ;
//        up_vec_to_min(1) =    (ub(i-1, j-1) - b1)  ;
// 
//        stan::math::var  low =    stan::math::max( low_vec_to_max   );   // new
//        stan::math::var  up  =    stan::math::min( up_vec_to_min    );   // new
// 
//        if (known_values_indicator(i-1, j-1) == 1) {
//          L(i-1, j-1) =  stan::math::to_var(known_values(i-1, j-1)) /  D(j-1)  ; // new
//        } else {
//          Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_outs = lb_ub_lp(Omega_theta_unconstrained_array(i-1, j-1), low,  up) ;
//          target += lb_ub_lp_outs.eval()(0);
//          stan::math::var x = lb_ub_lp_outs.eval()(1);    // logit bounds
//          L(i-1, j-1)  = x / D(j-1) ;
//          target += -0.5 * stan::math::log(D(j-1)) ;
//          // target += -  stan::math::log(D(j-1)) ;
//        }
// 
//        l_ij_old *= 1 - (D(j-1) *  stan::math::square(L(i-1, j-1) )) / l_ij_old;
//      }
//      D(i-1) = l_ij_old;
//    }
//    //L(0, 0) = 1;
// 
//    //////////// output
//    Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor  > out_mat = Eigen::Matrix<stan::math::var, -1, -1, Eigen::RowMajor  >::Zero(1 + K , K);
// 
//    out_mat(0, 0) = target;
//    // out_mat.block(1, 0, n, n) = L;
//    out_mat.block(1, 0, K, K) = stan::math::diag_post_multiply(L, stan::math::sqrt(stan::math::abs(D)));
// 
//    return(out_mat);
// 
//  }
// 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
