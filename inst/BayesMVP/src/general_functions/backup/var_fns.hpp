
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

 

 
 
 Eigen::Matrix<stan::math::var, -1, 1 >                        lb_ub_lp (stan::math::var  y,
                                                                         stan::math::var lb,
                                                                         stan::math::var ub) {
   
   stan::math::var target = 0.0 ;
   
   // stan::math::var val   = (lb  + (ub  - lb) * stan::math::inv_logit(y)) ;
   stan::math::var val   =  lb +  (ub - lb) *  0.5 * (1.0 +  stan::math::tanh(y));
   
   // target += stan::math::log(ub - lb) + stan::math::log_inv_logit(y) + stan::math::log1m_inv_logit(y);
   target +=  stan::math::log(ub - lb) - log(2)  + stan::math::log1m(stan::math::square(stan::math::tanh(y)));
   
   Eigen::Matrix<stan::math::var, -1, 1 > out_mat  = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(2);
   out_mat(0) = target;
   out_mat(1) = val;
   
   return(out_mat) ;
   
 } 
 
 
 
 
 
 
 Eigen::Matrix<stan::math::var, -1, 1 >   lb_ub_lp_vec_y (Eigen::Matrix<stan::math::var, -1, 1 > y,
                                                          Eigen::Matrix<stan::math::var, -1, 1 > lb,
                                                          Eigen::Matrix<stan::math::var, -1, 1 > ub) {
   
   stan::math::var target = 0.0  ;
   
   
   //   stan::math::var val   =  lb +  (ub - lb) *  0.5 * (1 +  stan::math::tanh(y));
   Eigen::Matrix<stan::math::var, -1, 1 >  vec =   (lb.array() +  (ub.array()  - lb.array() ) *  0.5 * (1.0 +  stan::math::tanh(y).array() )).matrix();
   
   //  target += (stan::math::log( (ub.array() - lb.array()).matrix()).array() + stan::math::log_inv_logit(y).array() + stan::math::log1m_inv_logit(y).array()).matrix().sum() ;
   target +=  (stan::math::log((ub.array() - lb.array()).matrix()).array() - log(2)  +  stan::math::log1m(stan::math::square(stan::math::tanh(y))).array()).matrix().sum();
   
   Eigen::Matrix<stan::math::var, -1, 1 > out_mat  = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(vec.rows() + 1);
   out_mat(0) = target;
   out_mat.segment(1, vec.rows()) = vec;
   
   return(out_mat);
   
 }
 
 
 
 
 
 
 
 Eigen::Matrix<stan::math::var, -1, -1 >    Pinkney_LDL_bounds_opt( int K,
                                                                    Eigen::Matrix<stan::math::var, -1, -1 >  lb,
                                                                    Eigen::Matrix<stan::math::var, -1, -1 >  ub,
                                                                    Eigen::Matrix<stan::math::var, -1, -1 >  Omega_theta_unconstrained_array,
                                                                    Eigen::Matrix<int, -1, -1 >  known_values_indicator,
                                                                    Eigen::Matrix<double, -1, -1 >  known_values) { 
   
   
   stan::math::var target = 0.0 ;
   
   Eigen::Matrix<stan::math::var, -1, 1 > first_col = Omega_theta_unconstrained_array.col(0).segment(1, K - 1); // first col except first element
   Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_outs = lb_ub_lp_vec_y(first_col, lb.col(0).segment(1, K - 1), ub.col(0).segment(1, K - 1)) ;  // logit bounds
   target += lb_ub_lp_vec_y_outs.eval()(0);
   Eigen::Matrix<stan::math::var, -1, 1 >  z = lb_ub_lp_vec_y_outs.segment(1, K - 1);
   
   Eigen::Matrix<stan::math::var, -1, -1 > L = Eigen::Matrix<stan::math::var, -1, -1 >::Zero(K, K);
   
   for (int i = 0; i < K; ++i) {
     L(i, i) = 1.0;
   }
   
   Eigen::Matrix<stan::math::var, -1, 1 >  D = Eigen::Matrix<stan::math::var, -1, 1 >::Zero(K);
   
   D(0) = 1.0;
   L.col(0).segment(1, K - 1) = z.head(K - 1);
   D(1) = 1.0 -  stan::math::square(L(1, 0)) ;
   
   for (int i = 2; i < K + 1; ++i) {
     if (known_values_indicator(i-1, 0) == 1) {
       L(i-1, 0) = stan::math::to_var(known_values(i-1, 0));
       Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_vec_y_out = lb_ub_lp(first_col(i-2), lb(i-1, 0), ub(i-1, 0)) ;  // logit bounds
       target += - lb_ub_lp_vec_y_out.eval()(0); // undo jac adjustment
     }
   }
   
   for (int i = 3; i < K + 1; ++i) {
     
     D(i-1) = 1.0 - stan::math::square(L(i-1, 0)) ; // checked
     Eigen::Matrix<stan::math::var, 1, -1 >  row_vec_rep = stan::math::rep_row_vector(1.0 - stan::math::square(L(i-1, 0)), i - 2) ; // checked
     L.row(i - 1).segment(1, i - 2) = row_vec_rep; // checked
     stan::math::var   l_ij_old = L(i-1, 1); // checked
     
     for (int j = 2; j < i; ++j) {
       
       stan::math::var b1 = stan::math::dot_product(L.row(j - 1).head(j - 1), (D.head(j - 1).transpose().array() *  L.row(i - 1).head(j - 1).array() ).matrix()  ) ; // checked
       
       Eigen::Matrix<stan::math::var, -1, 1 > low_vec_to_max(2);
       Eigen::Matrix<stan::math::var, -1, 1 > up_vec_to_min(2);
       low_vec_to_max(0) = - stan::math::sqrt(l_ij_old) * D(j-1) ;
       low_vec_to_max(1) =   (lb(i-1, j-1) - b1) ;
       up_vec_to_min(0) =    stan::math::sqrt(l_ij_old) * D(j-1) ;
       up_vec_to_min(1) =    (ub(i-1, j-1) - b1)  ;
       
       stan::math::var  low =    stan::math::max( low_vec_to_max   );   // new
       stan::math::var  up  =    stan::math::min( up_vec_to_min    );   // new
       
       if (known_values_indicator(i-1, j-1) == 1) {
         L(i-1, j-1) =  stan::math::to_var(known_values(i-1, j-1)) /  D(j-1)  ; // new
       } else {
         Eigen::Matrix<stan::math::var, -1, 1 >  lb_ub_lp_outs = lb_ub_lp(Omega_theta_unconstrained_array(i-1, j-1), low,  up) ;
         target += lb_ub_lp_outs.eval()(0);
         stan::math::var x = lb_ub_lp_outs.eval()(1);    // logit bounds
         L(i-1, j-1)  = x / D(j-1) ;
         target += -0.5 * stan::math::log(D(j-1)) ;
         // target += -  stan::math::log(D(j-1)) ;
       }
       
       l_ij_old *= 1.0 - (D(j-1) *  stan::math::square(L(i-1, j-1) )) / l_ij_old; // checked
     }
     
     D(i-1) = l_ij_old;
   }
   //L(0, 0) = 1;
   
   //////////// output
   Eigen::Matrix<stan::math::var, -1, -1 > out_mat = Eigen::Matrix<stan::math::var, -1, -1 >::Zero(1 + K , K);
   
   out_mat(0, 0) = target;
   // out_mat.block(1, 0, n, n) = L;
   out_mat.block(1, 0, K, K) = stan::math::diag_post_multiply(L, stan::math::sqrt(stan::math::abs(D)));
   
   return(out_mat);
   
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
 
 
 
 
 
 
 
 
 
 std::vector<Eigen::Matrix<stan::math::var, -1, -1>> vec_of_mats_var(int n_rows, int n_cols, int n_mats) {
   
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
 
 
 
 
 
 std::vector<stan::math::var> Eigen_vec_to_std_vec_var(Eigen::Matrix<stan::math::var, -1, 1> Eigen_vec) {

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
 
 
 
 
 // 
 // input vector, outputs upper-triangular 3d array of corrs- double
 std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >   fn_convert_std_vec_of_corrs_to_3d_array_var(
     std::vector<stan::math::var>   input_vec,
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




 
 // 
 // input vector, outputs upper-triangular 3d array of corrs- double
 std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >   fn_convert_Eigen_vec_of_corrs_to_3d_array_var(
     Eigen::Matrix<stan::math::var, -1, 1 >   input_vec,
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

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 