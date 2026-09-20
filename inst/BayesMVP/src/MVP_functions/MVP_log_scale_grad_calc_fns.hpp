
//// MVP_log_scale_grad_calc_fns.hpp

#pragma once


 

 
#include <Eigen/Dense>
 


 
 
//// using namespace Eigen;

 
 

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE

 
 
#pragma once
// ---------------------------------------------------------------------------------------------
//  Drop-in replacements for fn_MVP_compute_lp_GHK_cols_log_scale_underflow / _overflow.
//  Requires normal_tail_kernels.hpp (fast_log_Phi_*, fast_inv_Phi_from_log_p_*).
//
//  What changed vs. the old versions (same signatures, same output arrays, same semantics):
//    log_Phi_approx (logistic)              ->  exact, relative-accurate log Phi   (fast_log_Phi)
//    inv_Phi_approx_from_logit_prob (cubic) ->  exact inv_Phi fed log p, log(1-p) (fast_inv_Phi_from_log_p)
//    logistic-derivative "log phi"          ->  exact Gaussian  log phi(z) = -z^2/2 - 0.5 log 2pi
//  The gradient code already consumes log_phi_Bound_Z / log_phi_Z_recip / y1_log_prob, so it
//  needs no change: these arrays are the exact derivatives of the lp computed here, as before.
//  vect_type strings: Model_args_strings(8) for log_Phi, (9) for inv_Phi. (10) is now unused.
// ---------------------------------------------------------------------------------------------

//// ---- two small dispatch helpers (chunked SIMD over a contiguous Eigen vector, scalar remainder)
inline Eigen::Matrix<double, -1, 1> fn_EIGEN_log_Phi(const Eigen::Matrix<double, -1, 1> &x,
                                                      const std::string &vect_type) {
      const int N = x.size();
      Eigen::Matrix<double, -1, 1> out(N);
      int i = 0;
#if defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)
      if (vect_type == "AVX512") {
          for (; i + 8 <= N; i += 8) {
              _mm512_storeu_pd(out.data() + i, fast_log_Phi_AVX512(_mm512_loadu_pd(x.data() + i)));
          }
      }
#endif
#if defined(__AVX2__)
      if (vect_type == "AVX2") {
          for (; i + 4 <= N; i += 4) {
              _mm256_storeu_pd(out.data() + i, fast_log_Phi_AVX2(_mm256_loadu_pd(x.data() + i)));
          }
      }
#endif
      for (; i < N; ++i) out(i) = fast_log_Phi(x(i));
      return out;
}

inline Eigen::Matrix<double, -1, 1> fn_EIGEN_inv_Phi_from_log_p(const Eigen::Matrix<double, -1, 1> &log_p,
                                                                 const Eigen::Matrix<double, -1, 1> &log_1m_p,
                                                                 const std::string &vect_type) {
      const int N = log_p.size();
      Eigen::Matrix<double, -1, 1> out(N);
      int i = 0;
#if defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)
      if (vect_type == "AVX512") {
          for (; i + 8 <= N; i += 8) {
              _mm512_storeu_pd(out.data() + i,
                               fast_inv_Phi_from_log_p_wo_checks_AVX512(_mm512_loadu_pd(log_p.data() + i),
                                                                        _mm512_loadu_pd(log_1m_p.data() + i)));
          }
      }
#endif
#if defined(__AVX2__)
      if (vect_type == "AVX2") {
          for (; i + 4 <= N; i += 4) {
              _mm256_storeu_pd(out.data() + i,
                               fast_inv_Phi_from_log_p_wo_checks_AVX2(_mm256_loadu_pd(log_p.data() + i),
                                                                      _mm256_loadu_pd(log_1m_p.data() + i)));
          }
      }
#endif
      for (; i < N; ++i) out(i) = fast_inv_Phi_from_log_p(log_p(i), log_1m_p(i));
      return out;
}



#include <atomic>
static std::atomic<long long> g_n_tail_obs{0};




// //// ---------------------------------------------------------------------------------------------
// //// UNDERFLOW branch:  Bound_Z << 0,  y == 0.   Phi_Z = u * Phi(Bound_Z)
// //// ---------------------------------------------------------------------------------------------
// ALWAYS_INLINE void fn_MVP_compute_lp_GHK_cols_log_scale_underflow(        const int t,
//                                                                           const std::vector<int> &index,
//                                                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
//                                                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
//                                                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
//                                                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_Z_std_norm,
//                                                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
//                                                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
//                                                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z,
//                                                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
//                                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  Bound_Z,
//                                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  u_array,
//                                                                           const Model_fn_args_struct &Model_args_as_cpp_struct
// ) {
//   
//             g_n_tail_obs += static_cast<long long>(index.size());
//   
//             const double half_log_2pi = 0.91893853320467274178;
// 
//             const std::string vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8);
//             const std::string vect_type_exp     = Model_args_as_cpp_struct.Model_args_strings(3);
//             const std::string vect_type_log     = Model_args_as_cpp_struct.Model_args_strings(4);
//             const std::string vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);
// 
//             //// exact, relative-accurate log Phi(Bound_Z)
//             Eigen::Matrix<double, -1, 1> temp = Bound_Z(index, t);
//             Eigen::Matrix<double, -1, 1> log_Bound_U_Phi_Bound_Z = fn_EIGEN_log_Phi(temp, vect_type_log_Phi);
//             Bound_U_Phi_Bound_Z(index, t) = fn_EIGEN_double(log_Bound_U_Phi_Bound_Z, "exp", vect_type_exp);
// 
//             //// log Phi_Z = log u + log Phi(Bound_Z)
//             temp = u_array(index, t);
//             Eigen::Matrix<double, -1, 1> u_log     = fn_EIGEN_double(temp, "log", vect_type_log);
//             Eigen::Matrix<double, -1, 1> log_Phi_Z = u_log + log_Bound_U_Phi_Bound_Z;
//             Phi_Z(index, t) = fn_EIGEN_double(log_Phi_Z, "exp", vect_type_exp);
// 
//             //// log(1 - Phi_Z) = log1m(u * Phi(Bound_Z))   (argument is tiny here -> accurate)
//             temp.array() = Bound_U_Phi_Bound_Z(index, t).array() * u_array(index, t).array();
//             Eigen::Matrix<double, -1, 1> log_1m_Phi_Z = fn_EIGEN_double(temp, "log1m", vect_type_log);
// 
//             //// exact inv_Phi from the two logs (AS241 tail case fed sqrt(-log p) directly)
//             Z_std_norm(index, t) = fn_EIGEN_inv_Phi_from_log_p(log_Phi_Z, log_1m_Phi_Z, vect_type_inv_Phi);
//             temp = stan::math::abs(Z_std_norm(index, t));
//             log_Z_std_norm(index, t) = fn_EIGEN_double(temp, "log", vect_type_log);
// 
//             y1_log_prob(index, t) = log_Bound_U_Phi_Bound_Z;
//             prob(index, t)        = Bound_U_Phi_Bound_Z(index, t);
// 
//             //// exact Gaussian densities: log phi(Bound_Z) and -log phi(Z).
//             //// These are the derivatives of the exact Phi / inv_Phi used above (gradient code unchanged).
//             temp = Bound_Z(index, t);
//             log_phi_Bound_Z(index, t).array() = -0.5 * temp.array().square() - half_log_2pi;
//             temp = Z_std_norm(index, t);
//             log_phi_Z_recip(index, t).array() =  0.5 * temp.array().square() + half_log_2pi;
// }




// //// ---------------------------------------------------------------------------------------------
// //// OVERFLOW branch:  Bound_Z >> 0,  y == 1.   Phi_Z = Phi(Bound_Z) + u * (1 - Phi(Bound_Z))
// //// ---------------------------------------------------------------------------------------------
// ALWAYS_INLINE void fn_MVP_compute_lp_GHK_cols_log_scale_overflow(     const int t,
//                                                                       const int num_overflows,
//                                                                       const std::vector<int> &index,
//                                                                       Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
//                                                                       Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
//                                                                       Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
//                                                                       Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_Z_std_norm,
//                                                                       Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
//                                                                       Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
//                                                                       Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z,
//                                                                       Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
//                                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  Bound_Z,
//                                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  u_array,
//                                                                       const Model_fn_args_struct &Model_args_as_cpp_struct
// ) {
//   
//          g_n_tail_obs += static_cast<long long>(index.size());
//   
//          const double half_log_2pi = 0.91893853320467274178;
// 
//          const std::string vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8);
//          const std::string vect_type_exp     = Model_args_as_cpp_struct.Model_args_strings(3);
//          const std::string vect_type_log     = Model_args_as_cpp_struct.Model_args_strings(4);
//          const std::string vect_type_lse     = Model_args_as_cpp_struct.Model_args_strings(5);
//          const std::string vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);
// 
//          //// exact, relative-accurate log(1 - Phi(Bound_Z)) = log Phi(-Bound_Z)
//          Eigen::Matrix<double, -1, 1> temp = Bound_Z(index, t);
//          Eigen::Matrix<double, -1, 1> neg_Bound_Z = -temp;
//          Eigen::Matrix<double, -1, 1> log_Bound_U_Phi_Bound_Z_1m = fn_EIGEN_log_Phi(neg_Bound_Z, vect_type_log_Phi);
//          Eigen::Matrix<double, -1, 1> Bound_U_Phi_Bound_Z_1m     = fn_EIGEN_double(log_Bound_U_Phi_Bound_Z_1m, "exp", vect_type_exp);
//          Eigen::Matrix<double, -1, 1> log_Bound_U_Phi_Bound_Z    = fn_EIGEN_double(Bound_U_Phi_Bound_Z_1m, "log1m", vect_type_log);  // log(1 - tiny): accurate
//          Bound_U_Phi_Bound_Z(index, t).array() = 1.0 - Bound_U_Phi_Bound_Z_1m.array();
// 
//          //// log Phi_Z = LSE( log(1-Phi_b) + log u ,  log Phi_b )
//          Eigen::Matrix<double, -1, -1> tmp_array_2d_to_lse = Eigen::Matrix<double, -1, -1>::Zero(num_overflows, 2);
//          temp = u_array(index, t);
//          tmp_array_2d_to_lse.col(0) = log_Bound_U_Phi_Bound_Z_1m + fn_EIGEN_double(temp, "log", vect_type_log);
//          tmp_array_2d_to_lse.col(1) = log_Bound_U_Phi_Bound_Z;
//          Eigen::Matrix<double, -1, 1> log_Phi_Z = fn_log_sum_exp_2d_double(tmp_array_2d_to_lse, vect_type_lse);
//          Phi_Z(index, t) = fn_EIGEN_double(log_Phi_Z, "exp", vect_type_exp);
// 
//          //// log(1 - Phi_Z) = log(1-u) + log(1-Phi_b)
//          temp = u_array(index, t);
//          Eigen::Matrix<double, -1, 1> log_1m_Phi_Z = fn_EIGEN_double(temp, "log1m", vect_type_log);
//          log_1m_Phi_Z.array() += log_Bound_U_Phi_Bound_Z_1m.array();
// 
//          //// exact inv_Phi from the two logs
//          Z_std_norm(index, t) = fn_EIGEN_inv_Phi_from_log_p(log_Phi_Z, log_1m_Phi_Z, vect_type_inv_Phi);
//          temp = stan::math::abs(Z_std_norm(index, t));
//          log_Z_std_norm(index, t) = fn_EIGEN_double(temp, "log", vect_type_log);
// 
//          y1_log_prob(index, t) = log_Bound_U_Phi_Bound_Z_1m;
//          prob(index, t)        = Bound_U_Phi_Bound_Z_1m;
// 
//          //// exact Gaussian densities (derivatives of the exact Phi / inv_Phi used above)
//          temp = Bound_Z(index, t);
//          log_phi_Bound_Z(index, t).array() = -0.5 * temp.array().square() - half_log_2pi;
//          temp = Z_std_norm(index, t);
//          log_phi_Z_recip(index, t).array() =  0.5 * temp.array().square() + half_log_2pi;
// }




ALWAYS_INLINE void fn_MVP_compute_lp_GHK_cols_log_scale_underflow(        const int t,
                                                                          const std::vector<int> &index,
                                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_Z_std_norm,
                                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z,
                                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  Bound_Z,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  u_array,
                                                                          const Model_fn_args_struct &Model_args_as_cpp_struct
) {

            const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
            const double a = 0.07056;
            const double b = 1.5976;
            const double a_times_3 = 3.0 * a;

            const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
            const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);

            const std::string vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8);
            const std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
            const std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
            const std::string vect_type_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_strings(10);

            Eigen::Matrix<double, -1, 1> temp = Bound_Z(index, t);
            Eigen::Matrix<double, -1, 1> log_Bound_U_Phi_Bound_Z =   fn_EIGEN_double(temp, "log_Phi_approx",  vect_type_log_Phi);
            Bound_U_Phi_Bound_Z(index, t)    =    fn_EIGEN_double( log_Bound_U_Phi_Bound_Z, "exp", vect_type_exp);

            temp = u_array(index, t);
            Eigen::Matrix<double, -1, 1> u_log =  fn_EIGEN_double( temp, "log",  vect_type_log);
            Eigen::Matrix<double, -1, 1> log_Phi_Z = u_log + log_Bound_U_Phi_Bound_Z ; /// log(u * Phi_Bound_Z);
            Phi_Z(index, t) =   fn_EIGEN_double(log_Phi_Z, "exp",  vect_type_exp);  //// computed but not actually used

            //// log_1m_Phi_Z =   stan::math::log1m_exp( u_log + log_Bound_U_Phi_Bound_Z );
            temp.array() = Bound_U_Phi_Bound_Z(index, t).array() * u_array(index, t).array();
            Eigen::Matrix<double, -1, 1> log_1m_Phi_Z = fn_EIGEN_double(temp, "log1m",  vect_type_log);
            Eigen::Matrix<double, -1, 1> logit_Phi_Z =   log_Phi_Z - log_1m_Phi_Z;
            Z_std_norm(index, t)    =  fn_EIGEN_double( logit_Phi_Z, "inv_Phi_approx_from_logit_prob",  vect_type_inv_Phi_approx_from_logit_prob);

            temp = Z_std_norm(index, t);
            temp  = stan::math::abs(temp);
            log_Z_std_norm(index, t)   = fn_EIGEN_double(temp, "log", vect_type_log);

            y1_log_prob(index, t)  =    log_Bound_U_Phi_Bound_Z ;
            prob(index, t) =        Bound_U_Phi_Bound_Z(index, t) ; //// computed but not actually used

            ////  log_Bound_U_Phi_Bound_Z_1m =  stan::math::log1m_exp(log_Bound_U_Phi_Bound_Z); //// use log1m_exp for stability!
            temp =  Bound_U_Phi_Bound_Z(index, t);
            Eigen::Matrix<double, -1, 1> log_Bound_U_Phi_Bound_Z_1m = fn_EIGEN_double(temp, "log1m",  vect_type_log);

            temp = Bound_Z(index, t);
            Eigen::Matrix<double, -1, 1> temp_sq = stan::math::square(temp); // .array().square();
            temp.array() = a_times_3 * temp_sq.array() + b;
            temp = fn_EIGEN_double(temp, "log", vect_type_log);
            temp.array() += log_Bound_U_Phi_Bound_Z.array();
            temp.array() += log_Bound_U_Phi_Bound_Z_1m.array();
            log_phi_Bound_Z(index, t) = temp;
            ////log_phi_Bound_Z(index, t)   = fn_EIGEN_double(temp, "log", vect_type_log);

            temp = Z_std_norm(index, t);
            temp_sq = stan::math::square(temp); // .array().square();
            temp.array() = a_times_3 * temp_sq.array() + b;
            temp = fn_EIGEN_double(temp, "log", vect_type_log);
            temp.array() += log_Phi_Z.array();
            temp.array() += log_1m_Phi_Z.array();
            temp = -1.0*temp;
            ////  log_phi_Z_recip(index, t) = -1.0*fn_EIGEN_double(temp, "log", vect_type_log);
            log_phi_Z_recip(index, t) = temp;


}












ALWAYS_INLINE void fn_MVP_compute_lp_GHK_cols_log_scale_overflow(     const int t,
                                                                      const int num_overflows,
                                                                      const std::vector<int> &index,
                                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_Z_std_norm,
                                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z,
                                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  Bound_Z,
                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  u_array,
                                                                      const Model_fn_args_struct &Model_args_as_cpp_struct
) {

         const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
         const double a = 0.07056;
         const double b = 1.5976;
         const double a_times_3 = 3.0 * a;

         const std::string vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8);
         const std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
         const std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
         const std::string vect_type_lse = Model_args_as_cpp_struct.Model_args_strings(5);
         const std::string vect_type_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_strings(10);

         Eigen::Matrix<double, -1, 1>  temp = Bound_Z(index, t);
         temp = -1.0*temp;
         Eigen::Matrix<double, -1, 1> log_Bound_U_Phi_Bound_Z_1m =    fn_EIGEN_double( temp, "log_Phi_approx",  vect_type_log_Phi); /// TEMP
         Eigen::Matrix<double, -1, 1> Bound_U_Phi_Bound_Z_1m =        fn_EIGEN_double( log_Bound_U_Phi_Bound_Z_1m, "exp",  vect_type_exp);

         Eigen::Matrix<double, -1, 1> log_Bound_U_Phi_Bound_Z     =  fn_EIGEN_double(Bound_U_Phi_Bound_Z_1m, "log1m",  vect_type_log);
         // Eigen::Matrix<double, -1, 1>  log_Bound_U_Phi_Bound_Z =  stan::math::log1m_exp(log_Bound_U_Phi_Bound_Z_1m); //// use log1m_exp for stability!
         // log_Bound_U_Phi_Bound_Z = log_Bound_U_Phi_Bound_Z.array().min(700.0).max(-700.0);

         Bound_U_Phi_Bound_Z(index, t).array() =   1.0 - Bound_U_Phi_Bound_Z_1m.array(); //// this is computed but not actually used?

         Eigen::Matrix<double, -1, -1>  tmp_array_2d_to_lse =   Eigen::Matrix<double, -1, -1>::Zero(num_overflows, 2);
         temp = u_array(index, t);
         tmp_array_2d_to_lse.col(0)  =    log_Bound_U_Phi_Bound_Z_1m + fn_EIGEN_double(temp, "log", vect_type_log);
         tmp_array_2d_to_lse.col(1)  =    log_Bound_U_Phi_Bound_Z;
         Eigen::Matrix<double, -1, 1> log_Phi_Z = fn_log_sum_exp_2d_double(tmp_array_2d_to_lse, vect_type_lse);

         Phi_Z(index, t) = fn_EIGEN_double( log_Phi_Z, "exp",  vect_type_exp);
         temp =  u_array(index, t);
         Eigen::Matrix<double, -1, 1> log_1m_Phi_Z = fn_EIGEN_double( temp, "log1m",  vect_type_log);// + log_Bound_U_Phi_Bound_Z_1m;
         log_1m_Phi_Z.array() += log_Bound_U_Phi_Bound_Z_1m.array();

         Eigen::Matrix<double, -1, 1> logit_Phi_Z = log_Phi_Z;// - log_1m_Phi_Z;
         logit_Phi_Z.array() += -log_1m_Phi_Z.array();
         Z_std_norm(index, t)   =     fn_EIGEN_double( logit_Phi_Z, "inv_Phi_approx_from_logit_prob", vect_type_inv_Phi_approx_from_logit_prob);
         temp = stan::math::abs(Z_std_norm(index, t));
         log_Z_std_norm(index, t)   = fn_EIGEN_double( temp, "log", vect_type_log);

         y1_log_prob(index, t)  =    log_Bound_U_Phi_Bound_Z_1m;
         prob(index, t)   =              Bound_U_Phi_Bound_Z_1m;  //// this is computed but not actually used?

         temp =   Bound_Z(index, t);
         temp = stan::math::square(temp);
         temp.array() = a_times_3 * temp.array() + b;
         temp = fn_EIGEN_double(temp, "log", vect_type_log);
         temp.array() += log_Bound_U_Phi_Bound_Z.array();
         temp.array() += log_Bound_U_Phi_Bound_Z_1m.array();
         log_phi_Bound_Z(index, t) = temp;
         // log_phi_Bound_Z(index, t).array()  =          stan::math::log( a_times_3 * Bound_Z(index, t).array().square() + b  ).array()  +   log_Bound_U_Phi_Bound_Z.array()  +   log_Bound_U_Phi_Bound_Z_1m.array();


         temp =   Z_std_norm(index, t);
         temp = stan::math::square(temp);
         temp.array() = a_times_3 * temp.array() + b;
         temp = fn_EIGEN_double(temp, "log", vect_type_log);
         temp.array() += log_Phi_Z.array();
         temp.array() += log_1m_Phi_Z.array();
         temp = -1.0*temp;
         log_phi_Z_recip(index, t) = temp;
         // log_phi_Z_recip(index, t).array()  =    - (   stan::math::log(  ( a_times_3 * Z_std_norm(index, t).array().square() + b  ).array()  ).array()  +   log_Phi_Z.array()  +  log_1m_Phi_Z.array()  ).array() ;

}


// 




 
 
 
 


ALWAYS_INLINE  void fn_MVP_grad_prep_log_scale(          Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_prob_rowwise_prod_temp,
                                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_prob_recip_rowwise_prod_temp,
                                                         Eigen::Ref<Eigen::Matrix<double, -1, 1>>  log_prob_rowwise_prod_temp_all,
                                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_common_grad_term_1,
                                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob_recip,
                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  log_prob_n_recip,
                                                         ////
                                                         Eigen::Ref<Eigen::Matrix<double, -1, 1>> log_prev_per_obs_given_c, //// ----
                                                         ////
                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z,
                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_recip_double,
                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_L_Omega_recip_double,
                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_sign_chunk,
                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_m_y_sign_x_u,
                                                         const Model_fn_args_struct &Model_args_as_cpp_struct
) {
     
         const int n_tests = y1_log_prob.cols();
         const int chunk_size = y1_log_prob.rows();
         
         const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
         const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
         const std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
         const std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
         
// #ifdef _WIN32

      Eigen::Matrix<double, -1, 1>  temp_rowwise_sum =       Eigen::Matrix<double, -1, 1>::Zero(chunk_size);

       {
  
           for (int i = 0; i < n_tests; i++) {
             
                   Eigen::Matrix<double, -1, -1>  block_temp = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, i + 1);

                   int t = n_tests - (i + 1);
                   
                   block_temp = y1_log_prob.block(0, t + 0, chunk_size, i + 1);
                   temp_rowwise_sum =  block_temp.rowwise().sum();
                   log_prob_rowwise_prod_temp.col(t)  =  temp_rowwise_sum;

                   block_temp = y1_log_prob_recip.block(0, t + 0, chunk_size, i + 1);
                   temp_rowwise_sum =  block_temp.rowwise().sum();
                   log_prob_recip_rowwise_prod_temp.col(t)  =    temp_rowwise_sum;

           }

             log_prob_rowwise_prod_temp_all  =   y1_log_prob.rowwise().sum();

           if (n_class > 1) { //// i.e. if latent class
             
                 Eigen::Matrix<double, -1, 1> log_prev_p_log_prob_n_recip =            Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
                 Eigen::Matrix<double, -1, 1> res =                                    Eigen::Matrix<double, -1, 1>::Zero(chunk_size);

                 for (int i = 0; i < n_tests; i++) {

                       int t = n_tests - (i + 1);
                   
                       temp_rowwise_sum =  y1_log_prob.rowwise().sum();
                       log_prev_p_log_prob_n_recip = log_prob_n_recip;
                       log_prev_p_log_prob_n_recip.array() += log_prev_per_obs_given_c.array();
                      
                       res = log_prev_p_log_prob_n_recip; //  + temp_rowwise_sum + log_prob_recip_rowwise_prod_temp.col(t);
                       res.array() += temp_rowwise_sum.array();
                       res.array() += log_prob_recip_rowwise_prod_temp.col(t).array();
                       log_common_grad_term_1.col(t) = res;

                 }

           } else {

                 log_common_grad_term_1.setConstant(-700.0);

           }
           
           Eigen::Matrix<double, -1, 1> log_abs_res_1 =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
           Eigen::Matrix<double, -1, 1> log_abs_res_2 =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
           Eigen::Matrix<double, -1, 1> sign_res_1 =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
           Eigen::Matrix<double, -1, 1> sign_res_2 =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
           Eigen::Matrix<double, -1, 1> temp_col_t =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
           Eigen::Matrix<double, -1, 1> abs_res_1 =      Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
           Eigen::Matrix<double, -1, 1> sign_temp =      Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
           
           for (int t = 0; t < n_tests; t++) {

                 const double L_Omega_diag_log_abs = log_abs_L_Omega_recip_double(t, t);
                 const double L_Omega_diag_sign = sign_L_Omega_recip_double(t, t);
                 
                 log_abs_res_1 = log_phi_Bound_Z.col(t);
                 log_abs_res_1.array() += L_Omega_diag_log_abs;
                 log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t) = log_abs_res_1;
           
                 temp_col_t = y_m_y_sign_x_u.col(t);
                 abs_res_1 = stan::math::abs(temp_col_t);
                 log_abs_res_1 = fn_EIGEN_double( abs_res_1, "log",  vect_type_log);
                 log_abs_res_2  =   log_abs_res_1;
                 log_abs_res_2.array() +=   log_phi_Z_recip.col(t).array();
                 log_abs_res_2.array() +=   log_phi_Bound_Z.col(t).array();
                 log_abs_res_2.array() +=   L_Omega_diag_log_abs;
                 log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t) = log_abs_res_2;

                 //// note that densities and probs are always positive so signs = +1
                 temp_col_t = y_sign_chunk.col(t);
                 sign_temp = stan::math::sign(temp_col_t);
                 sign_res_1  = sign_temp *  L_Omega_diag_sign ;
                 sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t) = sign_res_1;

                 temp_col_t = y_m_y_sign_x_u.col(t);
                 sign_temp = stan::math::sign(temp_col_t);
                 sign_res_2  = (sign_temp * L_Omega_diag_sign) ;
                 sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t) = sign_res_2;

           }

       }
// #else
//        {
//            for (int i = 0; i < n_tests; i++) {
//                int t = n_tests - (i + 1);
//                log_prob_rowwise_prod_temp.col(t)     =               (y1_log_prob.block(0, t + 0, chunk_size, i + 1).rowwise().sum());
//                log_prob_recip_rowwise_prod_temp.col(t).array()  =    (y1_log_prob_recip.block(0, t + 0, chunk_size, i + 1).rowwise().sum()); //.array();
//            }
// 
//            log_prob_rowwise_prod_temp_all  =   (y1_log_prob.rowwise().sum());
// 
//            if (n_class > 1) { ///// i.e. if latent class
// 
//                  for (int i = 0; i < n_tests; i++) {
//                    int t = n_tests - (i + 1) ;
//                    log_common_grad_term_1.col(t) =    ( (  log_prev + log_prob_n_recip.array() ).matrix() + y1_log_prob.rowwise().sum()  +    log_prob_recip_rowwise_prod_temp.col(t) )  ;
//                  }
// 
//            } else {
// 
//                  log_common_grad_term_1.setConstant(-700);
// 
//            }
// 
//            for (int t = 0; t < n_tests; t++) {
// 
//                  log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   =   log_phi_Bound_Z.col(t).array() + ((log_abs_L_Omega_recip_double(t, t))) ;
// 
//                  log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array() = fn_EIGEN_double( y_m_y_sign_x_u.col(t).array().abs().matrix(), "log",  vect_type_log).array()
//                                                                                                                      + log_phi_Z_recip.col(t).array()  + log_phi_Bound_Z.col(t).array()  +  ((log_abs_L_Omega_recip_double(t, t)));
// 
//                  //// note that densities and probs are always positive so signs = +1
//                  sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()  = y_sign_chunk.col(t).array().sign() *  (sign_L_Omega_recip_double(t, t)) ;
//                  sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()  = y_m_y_sign_x_u.col(t).array().sign() *  (sign_L_Omega_recip_double(t, t)) ;
// 
//            }
// 
//        }
// 
// #endif
     
}








 
 
 
 

inline  void fn_MVP_compute_nuisance_grad_log_scale(       const std::vector<int> &n_problem_array,
                                                                  const std::vector<std::vector<int>> &problem_index_array,
                                                                  Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_abs_u_grad_array_CM_chunk,  // indexed
                                                                  Eigen::Ref<Eigen::Matrix<double, -1, -1>> u_grad_array_CM_chunk,  // indexed
                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double,
                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip, // indexed
                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob, // indexed
                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prob_recip,  // indexed
                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prob_rowwise_prod_temp,  // indexed
                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,  // indexed
                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,  // indexed
                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,  // indexed
                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,  // indexed
                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_common_grad_term_1,  // indexed
                                                                  Eigen::Matrix<double, -1, -1>  &log_abs_z_grad_term,  /// NOT Eigen::Ref as can't resize them !!!
                                                                  Eigen::Matrix<double, -1, -1>  &sign_z_grad_term,
                                                                  Eigen::Matrix<double, -1, -1>  &log_abs_grad_prob,
                                                                  Eigen::Matrix<double, -1, -1>  &sign_grad_prob,
                                                                  Eigen::Matrix<double, -1, 1>   &log_abs_prod_container_or_inc_array,
                                                                  Eigen::Matrix<double, -1, 1>   &sign_prod_container_or_inc_array,
                                                                  Eigen::Matrix<double, -1, 1>   &log_sum_result,  
                                                                  Eigen::Matrix<double, -1, 1>   &sign_sum_result, 
                                                                  Eigen::Matrix<double, -1, -1>  &log_terms,
                                                                  Eigen::Matrix<double, -1, -1>  &sign_terms,
                                                                  Eigen::Matrix<double, -1, 1> &log_abs_a,
                                                                  Eigen::Matrix<double, -1, 1> &log_abs_b,
                                                                  Eigen::Matrix<double, -1, 1> &sign_a,
                                                                  Eigen::Matrix<double, -1, 1> &sign_b,
                                                                  Eigen::Matrix<double, -1, 1> &container_max_logs,
                                                                  Eigen::Matrix<double, -1, 1> &container_sum_exp_signed,
                                                                  const Model_fn_args_struct &Model_args_as_cpp_struct
) {
  
  
  const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
  const int  n_class = Model_args_as_cpp_struct.Model_args_ints(1);
  
  const std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
  
  const int chunk_size = u_grad_array_CM_chunk.rows();
  const int n_tests = u_grad_array_CM_chunk.cols();
  
  log_abs_z_grad_term.setConstant(-700.0);
  log_abs_grad_prob.setConstant(-700.0);
  log_abs_prod_container_or_inc_array.setConstant(-700.0);
  log_sum_result.setConstant(-700.0);
  log_terms.setConstant(-700.0);
  log_abs_a.setConstant(-700.0);
  log_abs_b.setConstant(-700.0);
  container_max_logs.setConstant(-700.0);
  
  sign_z_grad_term.setOnes();
  sign_grad_prob.setOnes();
  sign_prod_container_or_inc_array.setOnes();
  sign_sum_result.setOnes();
  sign_terms.setOnes();
  sign_a.setOnes();
  sign_b.setOnes();
  
  container_sum_exp_signed.setZero();
  
  { // OK
    
    const int t = n_tests - 1;  ///// then second-to-last term (test T - 1)
    
    if (n_problem_array[t] > 0) {
    
        //Eigen::Ref<Eigen::Matrix<double, -1, 1>> col_ref_1 = log_abs_u_grad_array_CM_chunk.col(n_tests - 2);
        log_abs_u_grad_array_CM_chunk.col(n_tests - 2)(problem_index_array[t]) =      log_common_grad_term_1.col(t)(problem_index_array[t]).array()  + 
                                                                                      log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t)(problem_index_array[t]).array()  + 
                                                                                      log_abs_L_Omega_double(t, t - 1) + 
                                                                                      log_phi_Z_recip.col(t - 1)(problem_index_array[t]).array()  +
                                                                                      y1_log_prob.col(t - 1)(problem_index_array[t]).array()  ;
        
        //Eigen::Ref<Eigen::Matrix<double, -1, 1>> col_ref_2 = u_grad_array_CM_chunk.col(n_tests - 2);
        u_grad_array_CM_chunk.col(n_tests - 2)(problem_index_array[t]).array()  =  fn_EIGEN_double(log_abs_u_grad_array_CM_chunk.col(n_tests - 2)(problem_index_array[t]),   "exp",  vect_type).array() *
                                                                                 ( sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t)(problem_index_array[t]).array()  * 
                                                                                   stan::math::sign(  L_Omega_double(t, t - 1)  ) ).array() ;
        
    }
    
  }
  
  { ///// then third-to-last term (test T - 2)

    const int t = n_tests - 2;
    
    if (n_problem_array[t] > 0) {
    
        //// resize containers to match problem index
        log_abs_z_grad_term.resize(n_problem_array[t], n_tests);
        sign_z_grad_term.resize(n_problem_array[t], n_tests);
        log_abs_grad_prob.resize(n_problem_array[t], n_tests);
        sign_grad_prob.resize(n_problem_array[t], n_tests);
        log_terms.resize(n_problem_array[t], n_tests);
        sign_terms.resize(n_problem_array[t], n_tests);
    
        log_abs_prod_container_or_inc_array.resize(n_problem_array[t]);
        sign_prod_container_or_inc_array.resize(n_problem_array[t]);
        log_abs_a.resize(n_problem_array[t]);
        log_abs_b.resize(n_problem_array[t]);
        sign_a.resize(n_problem_array[t]);
        sign_b.resize(n_problem_array[t]);
        log_sum_result.resize(n_problem_array[t]);
        sign_sum_result.resize(n_problem_array[t]);
        container_max_logs.resize(n_problem_array[t]);
        container_sum_exp_signed.resize(n_problem_array[t]);
    
        log_abs_z_grad_term.setConstant(-700.0);
        log_abs_grad_prob.setConstant(-700.0);
        log_abs_prod_container_or_inc_array.setConstant(-700.0);
        log_sum_result.setConstant(-700.0);
        log_terms.setConstant(-700.0);
        log_abs_a.setConstant(-700.0);
        log_abs_b.setConstant(-700.0);
        container_max_logs.setConstant(-700.0);
    
        sign_z_grad_term.setOnes();
        sign_grad_prob.setOnes();
        sign_prod_container_or_inc_array.setOnes();
        sign_sum_result.setOnes();
        sign_terms.setOnes();
        sign_a.setOnes();
        sign_b.setOnes();
        
        // OK
    
        // /// 1st z term
        log_abs_z_grad_term.col(0)  = log_phi_Z_recip.col(t - 1)(problem_index_array[t])  + y1_log_prob.col(t - 1)(problem_index_array[t]); // causing aborted session
        sign_z_grad_term.col(0).setOnes();     // since densities are always +'ve!!
    
        // 2nd z term
        log_abs_z_grad_term.col(1).array() = log_abs_L_Omega_double(t, t - 1) +   log_abs_z_grad_term.col(0).array() +
                                             log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t)(problem_index_array[t]).array();
        sign_z_grad_term.col(1).array() =  (  stan::math::sign(L_Omega_double(t, t - 1)) *   sign_z_grad_term.col(0).array() *
                                             sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t)(problem_index_array[t]).array());
    
        // 1st prob term
        log_abs_prod_container_or_inc_array.array() = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t + 0, t - 1);
        sign_prod_container_or_inc_array.array()  =   sign_z_grad_term.col(0).array() * stan::math::sign(L_Omega_double(t + 0, t - 1));
        log_abs_grad_prob.col(0).array() = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 0)(problem_index_array[t]).array() +   log_abs_prod_container_or_inc_array.array();
        sign_grad_prob.col(0).array() =       sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 0)(problem_index_array[t]).array() *   sign_prod_container_or_inc_array.array();
    
        // 2nd prob term
        log_abs_a.setConstant(-700.0);
        log_abs_b.setConstant(-700.0);
        sign_a.setOnes();
        sign_b.setOnes();
    
        log_abs_a.array() =  log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t + 1, t - 1);
        log_abs_b.array() =  log_abs_z_grad_term.col(1).array() + log_abs_L_Omega_double(t + 1, t + 0);
        sign_a.array() =     sign_z_grad_term.col(0).array() * stan::math::sign( L_Omega_double(t + 1, t - 1) );
        sign_b.array() =     -1.0 *  sign_z_grad_term.col(1).array() * stan::math::sign(  L_Omega_double(t + 1, t + 0) ); // need to mult by -1
    
        log_terms.setConstant(-700.0);
        sign_terms.setOnes();
        log_sum_result.setConstant(-700.0);
        sign_sum_result.setOnes();
        log_terms.col(0) = log_abs_a;  log_terms.col(1) = log_abs_b;
        sign_terms.col(0) = sign_a; sign_terms.col(1) = sign_b;
    
        log_abs_sum_exp_general_v2( log_terms.leftCols(2), 
                                    sign_terms.leftCols(2),
                                    vect_type, vect_type,
                                    log_abs_prod_container_or_inc_array,
                                    sign_prod_container_or_inc_array,
                                    container_max_logs,
                                    container_sum_exp_signed);
    
        // then compute 2nd grad_prob term
        log_abs_grad_prob.col(1).array() = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1)(problem_index_array[t]).array() +   log_abs_prod_container_or_inc_array.array();
        sign_grad_prob.col(1).array() =       sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1)(problem_index_array[t]).array() *   sign_prod_container_or_inc_array.array();
    
        ////// then final computation for u_grad
        // u_grad_array_CM_chunk.col(n_tests - 3)  +=  ( A_common_grad_term_1.col(t).array()   *  (  A_grad_prob.col(1).array() *  A_prob.col(t).array()  +      A_grad_prob.col(0).array() *   A_prob.col(t+1).array()  )  )   ;
        log_abs_a = log_abs_grad_prob.col(0) + y1_log_prob.col(t + 1)(problem_index_array[t]);
        log_abs_b = log_abs_grad_prob.col(1) + y1_log_prob.col(t + 0)(problem_index_array[t]);
        sign_a = sign_grad_prob.col(0) ;
        sign_b = sign_grad_prob.col(1) ;
    
        log_terms.setConstant(-700.0);
        sign_terms.setOnes();
        log_sum_result.setConstant(-700.0);
        sign_sum_result.setOnes();
        log_terms.col(0) = log_abs_a;  log_terms.col(1) = log_abs_b;
        sign_terms.col(0) = sign_a;    sign_terms.col(1) = sign_b;
    
        log_abs_sum_exp_general_v2( log_terms.leftCols(2), 
                                    sign_terms.leftCols(2),
                                    vect_type, vect_type,
                                    log_sum_result,
                                    sign_sum_result,
                                    container_max_logs,
                                    container_sum_exp_signed);
    
        log_abs_u_grad_array_CM_chunk.col(n_tests - 3)(problem_index_array[t]) =      log_common_grad_term_1.col(t)(problem_index_array[t]) +  log_sum_result;
        Eigen::Matrix<double, -1, 1> temp = log_abs_u_grad_array_CM_chunk.col(n_tests - 3)(problem_index_array[t]);
        temp = fn_EIGEN_double( temp, "exp", vect_type);
        u_grad_array_CM_chunk.col(n_tests - 3)(problem_index_array[t]) =   temp.array() *  sign_sum_result.array();
        
    }

  }

  
       
      // compute remaining terms
      for (int i = 1; i < n_tests - 2; i++) {

            const int t = n_tests - (i + 2);

      if (n_problem_array[t] > 0) {

            //// resize containers to match problem index
            log_abs_z_grad_term.resize(n_problem_array[t], n_tests);
            sign_z_grad_term.resize(n_problem_array[t], n_tests);
            log_abs_grad_prob.resize(n_problem_array[t], n_tests);
            sign_grad_prob.resize(n_problem_array[t], n_tests);
            log_terms.resize(n_problem_array[t], n_tests);
            sign_terms.resize(n_problem_array[t], n_tests);
            
            log_abs_prod_container_or_inc_array.resize(n_problem_array[t]);
            sign_prod_container_or_inc_array.resize(n_problem_array[t]);
            log_abs_a.resize(n_problem_array[t]);
            log_abs_b.resize(n_problem_array[t]);
            sign_a.resize(n_problem_array[t]);
            sign_b.resize(n_problem_array[t]);
            log_sum_result.resize(n_problem_array[t]);
            sign_sum_result.resize(n_problem_array[t]);
            container_max_logs.resize(n_problem_array[t]);
            container_sum_exp_signed.resize(n_problem_array[t]);
            
            log_abs_z_grad_term.setConstant(-700.0);
            log_abs_grad_prob.setConstant(-700.0);
            log_abs_prod_container_or_inc_array.setConstant(-700.0);
            log_sum_result.setConstant(-700.0);
            log_terms.setConstant(-700.0);
            log_abs_a.setConstant(-700.0);
            log_abs_b.setConstant(-700.0);
            container_max_logs.setConstant(-700.0);
            
            sign_z_grad_term.setOnes();
            sign_grad_prob.setOnes();
            sign_prod_container_or_inc_array.setOnes();
            sign_sum_result.setOnes();
            sign_terms.setOnes();
            sign_a.setOnes();
            sign_b.setOnes();

            //// 1st z term
            log_abs_z_grad_term.col(0)  = log_phi_Z_recip.col(t - 1)(problem_index_array[t]) +   y1_log_prob.col(t - 1)(problem_index_array[t]);
            sign_z_grad_term.col(0).setOnes(); // since densities are always +'ve!!

            //// 1st prob term
            log_abs_prod_container_or_inc_array = log_abs_z_grad_term.col(0).array() +    log_abs_L_Omega_double(t, t - 1);
            sign_prod_container_or_inc_array  = sign_z_grad_term.col(0).array() *     stan::math::sign((L_Omega_double(t, t - 1)));

            log_abs_grad_prob.col(0) = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t)(problem_index_array[t]).array() +    log_abs_prod_container_or_inc_array.array();
            sign_grad_prob.col(0) = sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t)(problem_index_array[t]).array() *    sign_prod_container_or_inc_array.array();

            //// subsequent terms
            for (int ii = 1; ii < i + 2; ii++) {

                    //// next z term
                    log_abs_z_grad_term.col(ii).array()  = log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii - 1)(problem_index_array[t]).array() +    log_abs_prod_container_or_inc_array.array();
                    sign_z_grad_term.col(ii).array()  = sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii - 1)(problem_index_array[t]).array() *   sign_prod_container_or_inc_array.array();
  
                    //// 1st term (positive)
                    log_terms.col(0).array()  = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t + ii, t - 1);
                    sign_terms.col(0).array()  = sign_z_grad_term.col(0).array() *   stan::math::sign(L_Omega_double(t + ii, t - 1));
  
                    //// subsequent terms (negative)
                    for (int j = 1; j <= ii; j++) {
                      log_terms.col(j).array() =         log_abs_z_grad_term.col(j).array() +  log_abs_L_Omega_double(t + ii, t + j - 1);
                      sign_terms.col(j).array() = -1.0 * sign_z_grad_term.col(j).array() *     stan::math::sign(L_Omega_double(t + ii, t + j - 1));
                    }
  
                    //// combine terms
                    log_abs_sum_exp_general_v2( log_terms.leftCols(ii), 
                                                sign_terms.leftCols(ii),
                                                vect_type, vect_type,
                                                log_abs_prod_container_or_inc_array,
                                                sign_prod_container_or_inc_array,
                                                container_max_logs,
                                                container_sum_exp_signed);
  
                    //// compute prob term
                    log_abs_grad_prob.col(ii).array()  = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii)(problem_index_array[t]).array() +  log_abs_prod_container_or_inc_array.array();
                    sign_grad_prob.col(ii).array()  = sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii)(problem_index_array[t]).array() *   sign_prod_container_or_inc_array.array();

            }

            ////// final derivatives computation
            log_terms.setConstant(-700.0);
            sign_terms.setOnes();
            log_sum_result.setConstant(-700.0);
            sign_sum_result.setOnes();

            for (int ii = 0; ii < i + 2; ii++) {
              log_terms.col(ii)   = log_abs_grad_prob.col(ii).array() +  log_prob_rowwise_prod_temp.col(t)(problem_index_array[t]).array() + log_prob_recip.col(t + ii)(problem_index_array[t]).array();
              sign_terms.col(ii)  = sign_grad_prob.col(ii);
            }

            //// combine final terms
            log_abs_sum_exp_general_v2( log_terms.leftCols(i + 2), 
                                        sign_terms.leftCols(i + 2),
                                        vect_type, vect_type,
                                        log_sum_result,
                                        sign_sum_result,
                                        container_max_logs,
                                        container_sum_exp_signed);

            //// update final gradients
            log_abs_u_grad_array_CM_chunk.col(n_tests - (i + 3))(problem_index_array[t])   =      log_common_grad_term_1.col(t)(problem_index_array[t]).array() +   log_sum_result.array();
            u_grad_array_CM_chunk.col(n_tests - (i + 3))(problem_index_array[t])  =    fn_EIGEN_double( log_abs_u_grad_array_CM_chunk.col(n_tests - (i + 3))(problem_index_array[t]), "exp", vect_type).array() * sign_sum_result.array();


      }

      }
      
      //// once done, resize containers back to chunk_size
      log_abs_z_grad_term.resize(chunk_size, n_tests);
      sign_z_grad_term.resize(chunk_size, n_tests);
      log_abs_grad_prob.resize(chunk_size, n_tests);
      sign_grad_prob.resize(chunk_size, n_tests);
      log_terms.resize(chunk_size, n_tests);
      sign_terms.resize(chunk_size, n_tests);
      log_abs_prod_container_or_inc_array.resize(chunk_size);
      sign_prod_container_or_inc_array.resize(chunk_size);
      log_abs_a.resize(chunk_size);
      log_abs_b.resize(chunk_size);
      sign_a.resize(chunk_size);
      sign_b.resize(chunk_size);
      log_sum_result.resize(chunk_size);
      sign_sum_result.resize(chunk_size);
      container_max_logs.resize(chunk_size);
      container_sum_exp_signed.resize(chunk_size);
  
      
}
  





 
 
  
 
 
 
 
 
 
  
  
  
  
  
  
  

inline  void      fn_MVP_compute_coefficients_grad_log_scale(      const std::vector<int> &n_problem_array,
                                                                          const std::vector<std::vector<int>> &problem_index_array,
                                                                          Eigen::Matrix<double, -1, -1> &beta_grad_array,
                                                                          std::vector<Eigen::Matrix<double, -1, -1>> &sign_beta_grad_array_for_each_n, 
                                                                          std::vector<Eigen::Matrix<double, -1, -1>> &log_abs_beta_grad_array_for_each_n,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prob_rowwise_prod_temp,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_common_grad_term_1,
                                                                          Eigen::Matrix<double, -1, -1>  &log_abs_z_grad_term, /// NOT Eigen::Ref as can't resize them !!!
                                                                          Eigen::Matrix<double, -1, -1>  &sign_z_grad_term,
                                                                          Eigen::Matrix<double, -1, -1>  &log_abs_grad_prob,
                                                                          Eigen::Matrix<double, -1, -1>  &sign_grad_prob,
                                                                          Eigen::Matrix<double, -1, 1>   &log_abs_prod_container_or_inc_array,
                                                                          Eigen::Matrix<double, -1, 1>   &sign_prod_container_or_inc_array,
                                                                          Eigen::Matrix<double, -1, -1>  &log_abs_prod_container_or_inc_array_comp,
                                                                          Eigen::Matrix<double, -1, -1>  &sign_prod_container_or_inc_array_comp,
                                                                          Eigen::Matrix<double, -1, 1>   &log_sum_result,
                                                                          Eigen::Matrix<double, -1, 1>   &sign_sum_result,
                                                                          Eigen::Matrix<double, -1, -1>  &log_terms,
                                                                          Eigen::Matrix<double, -1, -1>  &sign_terms,
                                                                          Eigen::Matrix<double, -1, 1>   &container_max_logs,
                                                                          Eigen::Matrix<double, -1, 1>   &container_sum_exp_signed,
                                                                          const Model_fn_args_struct &Model_args_as_cpp_struct
) {
  
  
  const int  n_class = Model_args_as_cpp_struct.Model_args_ints(1);
  const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
  const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
  
  const int chunk_size = log_phi_Z_recip.rows();
  const int n_tests = log_phi_Z_recip.cols();
 
      log_abs_z_grad_term.setConstant(-700.0);
      log_abs_grad_prob.setConstant(-700.0);
      log_abs_prod_container_or_inc_array.setConstant(-700.0);
      log_sum_result.setConstant(-700.0);
      log_terms.setConstant(-700.0);
      container_max_logs.setConstant(-700.0);
      
      sign_z_grad_term.setOnes();
      sign_grad_prob.setOnes();
      sign_prod_container_or_inc_array.setOnes();
      sign_sum_result.setOnes();
      sign_terms.setOnes();
      
      container_sum_exp_signed.setZero();
      
      log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);
      sign_prod_container_or_inc_array_comp.setOnes();
  
  
  {   ///// first (last t) term 
    
      int t = n_tests - 1;
        
      if (n_problem_array[t] > 0) {
            
            log_abs_beta_grad_array_for_each_n[0].col(t)(problem_index_array[t]).array()  =  log_common_grad_term_1.col(t)(problem_index_array[t]).array() + log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t)(problem_index_array[t]).array();
            sign_beta_grad_array_for_each_n[0].col(t)(problem_index_array[t]).array()   = sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t)(problem_index_array[t]).array();
      
      }
    
  }
  
  for (int i = 0; i < n_tests - 1; i++) {
            
            int t = n_tests - (i + 2);
            
            if (n_problem_array[t] > 0) {
              
              //// resize containers to match problem index
              log_abs_z_grad_term.resize(n_problem_array[t], n_tests);
              sign_z_grad_term.resize(n_problem_array[t], n_tests);
              log_abs_grad_prob.resize(n_problem_array[t], n_tests); 
              sign_grad_prob.resize(n_problem_array[t], n_tests);
              log_terms.resize(n_problem_array[t], n_tests);
              sign_terms.resize(n_problem_array[t], n_tests);
              
              log_abs_prod_container_or_inc_array.resize(n_problem_array[t]);
              sign_prod_container_or_inc_array.resize(n_problem_array[t]);
              log_sum_result.resize(n_problem_array[t]); 
              sign_sum_result.resize(n_problem_array[t]);
              container_max_logs.resize(n_problem_array[t]);
              container_sum_exp_signed.resize(n_problem_array[t]);
              
              log_abs_prod_container_or_inc_array_comp.resize(n_problem_array[t], n_tests);
              sign_prod_container_or_inc_array_comp.resize(n_problem_array[t], n_tests);
              
              log_abs_z_grad_term.setConstant(-700.0);
              log_abs_grad_prob.setConstant(-700.0);
              log_abs_prod_container_or_inc_array.setConstant(-700.0); 
              log_sum_result.setConstant(-700.0);
              log_terms.setConstant(-700.0);
              container_max_logs.setConstant(-700.0);
              
              sign_z_grad_term.setOnes();
              sign_grad_prob.setOnes();
              sign_prod_container_or_inc_array.setOnes();
              sign_sum_result.setOnes();
              sign_terms.setOnes();
              
              log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);
              sign_prod_container_or_inc_array_comp.setOnes();
            
            // 1st grad_prob term
            log_abs_grad_prob.col(0) = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t)(problem_index_array[t]);
            sign_grad_prob.col(0) =    sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t)(problem_index_array[t]);
            
            // 1st z_grad term
            log_abs_z_grad_term.col(0).array() = log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t)(problem_index_array[t]).array() ;// +  stan::math::log(stan::math::abs(L_Omega_double(t, t))) ;
            sign_z_grad_term.col(0).array() =   -1.0 * sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t)(problem_index_array[t]).array() ;// * stan::math::sign(1.0  / L_Omega_double(t, t)) ;
              
              int ii = 0;  
              for (int iii = 0; iii < ii + 1; iii++) {
                    log_abs_prod_container_or_inc_array_comp.col(iii).array() =  log_abs_z_grad_term.col(iii).array() + log_abs_L_Omega_double.row(t + ii + 1).segment(t, ii + 1).transpose().eval()(iii); 
                    sign_prod_container_or_inc_array_comp.col(iii).array() =   sign_z_grad_term.col(iii).array() *  stan::math::sign(L_Omega_double.row(t + ii + 1).segment(t, ii + 1).transpose().eval()(iii));
              }
              
              sign_prod_container_or_inc_array = sign_prod_container_or_inc_array_comp.col(0);
              log_abs_prod_container_or_inc_array = log_abs_prod_container_or_inc_array_comp.col(0);
              
              // 2nd grad_prob term
              log_abs_grad_prob.col(1).array() = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1)(problem_index_array[t]).array() +   log_abs_prod_container_or_inc_array.array();
              sign_grad_prob.col(1).array() =    sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1)(problem_index_array[t]).array() *  sign_prod_container_or_inc_array.array();
         
           if (i > 0)  {
             
              for (int ii = 1; ii < i + 1; ii++) {
                
                    // compute next z term
                    log_abs_z_grad_term.col(ii).array()  = log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii)(problem_index_array[t]).array() +  log_abs_prod_container_or_inc_array.array();
                    sign_z_grad_term.col(ii).array() =  -1.0 * sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii)(problem_index_array[t]).array() *   sign_prod_container_or_inc_array.array();
                          
                          for (int iii = 0; iii < ii + 1; iii++) {
                            
                            log_abs_prod_container_or_inc_array_comp.col(iii).array() =  log_abs_z_grad_term.col(iii).array() + log_abs_L_Omega_double.row(t + ii + 1).segment(t, ii + 1).transpose().eval()(iii);
                            sign_prod_container_or_inc_array_comp.col(iii).array() =   sign_z_grad_term.col(iii).array() *  stan::math::sign(L_Omega_double.row(t + ii + 1).segment(t, ii + 1).transpose().eval()(iii));
                            
                          }
                          
                          log_abs_sum_exp_general_v2(  log_abs_prod_container_or_inc_array_comp.leftCols(ii + 1),
                                                       sign_prod_container_or_inc_array_comp.leftCols(ii + 1),
                                                       vect_type,  vect_type,
                                                       log_abs_prod_container_or_inc_array,
                                                       sign_prod_container_or_inc_array,
                                                       container_max_logs,
                                                       container_sum_exp_signed);
                    // then compute enxt prob term
                    log_abs_grad_prob.col(ii + 1) = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii + 1)(problem_index_array[t]).array() +  log_abs_prod_container_or_inc_array.array();
                    sign_grad_prob.col(ii + 1) =  sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii + 1)(problem_index_array[t]).array() *   sign_prod_container_or_inc_array.array();
        
              }
              
           }
        
              for (int ii = 0; ii < i + 2; ii++) {
                    log_terms.col(ii).array() = log_abs_grad_prob.col(ii).array() + log_prob_rowwise_prod_temp.col(t)(problem_index_array[t]).array() + (-y1_log_prob.array()).col(t + ii)(problem_index_array[t]).array();
                    sign_terms.col(ii).array() = sign_grad_prob.col(ii).array();
              }
        
              log_abs_sum_exp_general_v2( log_terms.leftCols(i + 2), 
                                          sign_terms.leftCols(i + 2),
                                          vect_type, vect_type,  
                                          log_sum_result, 
                                          sign_sum_result,
                                          container_max_logs,
                                          container_sum_exp_signed);
        
             log_abs_beta_grad_array_for_each_n[0].col(t)(problem_index_array[t]).array() = log_common_grad_term_1.col(t)(problem_index_array[t]).array() + log_sum_result.array(); 
             sign_beta_grad_array_for_each_n[0].col(t)(problem_index_array[t]).array() = sign_sum_result.array();
          
            }

  }
  
  
  //// once done, resize containers back to chunk_size
  log_abs_z_grad_term.resize(chunk_size, n_tests);
  sign_z_grad_term.resize(chunk_size, n_tests);
  log_abs_grad_prob.resize(chunk_size, n_tests);
  sign_grad_prob.resize(chunk_size, n_tests);
  log_terms.resize(chunk_size, n_tests);
  sign_terms.resize(chunk_size, n_tests);
  log_abs_prod_container_or_inc_array.resize(chunk_size);
  sign_prod_container_or_inc_array.resize(chunk_size);
  log_sum_result.resize(chunk_size);
  sign_sum_result.resize(chunk_size);
  container_max_logs.resize(chunk_size);
  container_sum_exp_signed.resize(chunk_size);
  
  log_abs_prod_container_or_inc_array_comp.resize(chunk_size, n_tests);
  sign_prod_container_or_inc_array_comp.resize(chunk_size, n_tests);
  
}

 
 
 
 
 
 
 
 
 
 
 
 
 
 








inline  void fn_MVP_compute_L_Omega_grad_log_scale(      const std::vector<int> &n_problem_array,
                                                                const std::vector<std::vector<int>> &problem_index_array,
                                                                Eigen::Ref<Eigen::Matrix<double, -1, -1>> L_Omega_grad_array,
                                                                std::vector<Eigen::Matrix<double, -1, -1>>  &sign_L_Omega_grad_array_col_for_each_n,
                                                                std::vector<Eigen::Matrix<double, -1, -1>>  &log_abs_L_Omega_grad_array_col_for_each_n,
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_Bound_Z,  ////
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_Bound_Z,  ////
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_Z_std_norm,  ////
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_Z_std_norm, ////
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double,
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prop_rowwise_prod_temp,
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_common_grad_term_1,
                                                                Eigen::Matrix<double, -1, -1>  &log_abs_z_grad_term,  /// NOT Eigen::Ref as can't resize them !!!
                                                                Eigen::Matrix<double, -1, -1>  &sign_z_grad_term,
                                                                Eigen::Matrix<double, -1, -1>  &log_abs_grad_prob,
                                                                Eigen::Matrix<double, -1, -1>  &sign_grad_prob,
                                                                Eigen::Matrix<double, -1, 1>   &log_abs_prod_container_or_inc_array,
                                                                Eigen::Matrix<double, -1, 1>   &sign_prod_container_or_inc_array,
                                                                Eigen::Matrix<double, -1, -1>  &log_abs_prod_container_or_inc_array_comp,
                                                                Eigen::Matrix<double, -1, -1>  &sign_prod_container_or_inc_array_comp,
                                                                Eigen::Matrix<double, -1, -1>  &log_abs_derivs_chain_container_vec_comp,
                                                                Eigen::Matrix<double, -1, -1>  &sign_derivs_chain_container_vec_comp,
                                                                Eigen::Matrix<double, -1, 1>   &log_sum_result,
                                                                Eigen::Matrix<double, -1, 1>   &sign_sum_result,
                                                                Eigen::Matrix<double, -1, -1>  &log_terms,
                                                                Eigen::Matrix<double, -1, -1>  &sign_terms,
                                                                Eigen::Matrix<double, -1, 1>   &log_abs_a,
                                                                Eigen::Matrix<double, -1, 1>   &log_abs_b,
                                                                Eigen::Matrix<double, -1, 1>   &sign_a,
                                                                Eigen::Matrix<double, -1, 1>   &sign_b,
                                                                Eigen::Matrix<double, -1, 1>   &container_max_logs,
                                                                Eigen::Matrix<double, -1, 1>   &container_sum_exp_signed,
                                                                const Model_fn_args_struct &Model_args_as_cpp_struct
) {

  
  
     const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
     const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
     const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
  
     const int n_tests = log_Bound_Z.cols();
     const int chunk_size = log_Bound_Z.rows();
     
     
    
       //  for (int t1 = 1; t1 < n_tests; t1++) {
       //   for (int t2 = 0; t2 < t1; t2++) {
       //     sign_L_Omega_grad_array_col_for_each_n[t1].col(t2)(problem_index_array[t1]).array() = 1.0;
       //     log_abs_L_Omega_grad_array_col_for_each_n[t1].col(t2)(problem_index_array[t1]).array() = -700.0;
       //   }
       // }
   
    
        ///////////////////////// deriv of diagonal elements (not needed if using the "standard" or "Stan" Cholesky parameterisation of Omega)
        //////// w.r.t last diagonal first
        {
          
            int  t1 = n_tests - 1;
            
            sign_L_Omega_grad_array_col_for_each_n[t1].col(t1)(problem_index_array[t1]).array() =    sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array() *  sign_Bound_Z.col(t1)(problem_index_array[t1]).array() ; 
            log_abs_L_Omega_grad_array_col_for_each_n[t1].col(t1)(problem_index_array[t1])   = log_common_grad_term_1.col(t1)(problem_index_array[t1]);  // +  log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array() + log_Bound_Z.col(t1)(problem_index_array[t1]).array();
            log_abs_L_Omega_grad_array_col_for_each_n[t1].col(t1)(problem_index_array[t1]).array() += log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array();
            log_abs_L_Omega_grad_array_col_for_each_n[t1].col(t1)(problem_index_array[t1]).array() += log_Bound_Z.col(t1)(problem_index_array[t1]).array();
        }
        
        // //////// then w.r.t the second-to-last diagonal
        {

        int t1 = n_tests - 2;
          
        if (n_problem_array[t1] > 0) {
          
            //// resize containers to match problem index
            log_abs_z_grad_term.resize(n_problem_array[t1], n_tests);
            sign_z_grad_term.resize(n_problem_array[t1], n_tests);
            log_abs_grad_prob.resize(n_problem_array[t1], n_tests); 
            sign_grad_prob.resize(n_problem_array[t1], n_tests);
            log_terms.resize(n_problem_array[t1], n_tests);
            sign_terms.resize(n_problem_array[t1], n_tests);
            
            log_abs_prod_container_or_inc_array.resize(n_problem_array[t1]);
            sign_prod_container_or_inc_array.resize(n_problem_array[t1]);
            log_sum_result.resize(n_problem_array[t1]); 
            sign_sum_result.resize(n_problem_array[t1]);
            container_max_logs.resize(n_problem_array[t1]);
            container_sum_exp_signed.resize(n_problem_array[t1]);
            
            log_abs_prod_container_or_inc_array_comp.resize(n_problem_array[t1], n_tests);
            sign_prod_container_or_inc_array_comp.resize(n_problem_array[t1], n_tests);
            
            log_abs_derivs_chain_container_vec_comp.resize(n_problem_array[t1], n_tests);
            sign_derivs_chain_container_vec_comp.resize(n_problem_array[t1], n_tests);
            
            log_abs_z_grad_term.setConstant(-700.0);
            log_abs_grad_prob.setConstant(-700.0);
            log_abs_prod_container_or_inc_array.setConstant(-700.0); 
            log_sum_result.setConstant(-700.0);
            log_terms.setConstant(-700.0);
            container_max_logs.setConstant(-700.0);
            
            sign_z_grad_term.setOnes();
            sign_grad_prob.setOnes();
            sign_prod_container_or_inc_array.setOnes();
            sign_sum_result.setOnes();
            sign_terms.setOnes();
            
            log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);
            sign_prod_container_or_inc_array_comp.setOnes();
            
            log_abs_derivs_chain_container_vec_comp.setConstant(-700.0);
            sign_derivs_chain_container_vec_comp.setOnes();
            
            ///// 1st grad_prob term 
            log_abs_grad_prob.col(0) = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]) +  log_Bound_Z.col(t1)(problem_index_array[t1]) ;
            sign_grad_prob.col(0).array() = sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array() *   sign_Bound_Z.col(t1)(problem_index_array[t1]).array();
    
            // 1st z_grad term
            log_abs_z_grad_term.col(0) = log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1)(problem_index_array[t1]) +  log_Bound_Z.col(t1)(problem_index_array[t1]);
            sign_z_grad_term.col(0).array() = -sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array() *  sign_Bound_Z.col(t1)(problem_index_array[t1]).array();
    
            log_abs_prod_container_or_inc_array.array() = log_abs_z_grad_term.col(0).array() +    log_abs_L_Omega_double(t1 + 1, t1);
            sign_prod_container_or_inc_array = sign_z_grad_term.col(0) *  stan::math::sign(L_Omega_double(t1 + 1, t1));
    
            // 2nd grad_prob term
            log_abs_grad_prob.col(1) = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1)(problem_index_array[t1]) +    log_abs_prod_container_or_inc_array ;
            sign_grad_prob.col(1).array() = sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1)(problem_index_array[t1]).array()  *   sign_prod_container_or_inc_array.array() ;
    
            log_abs_a = log_abs_grad_prob.col(1)  + y1_log_prob.col(t1)(problem_index_array[t1]);
            sign_a = sign_grad_prob.col(1)  ;  // sign of probs always +'ve
            log_abs_b = log_abs_grad_prob.col(0)  + y1_log_prob.col(t1 + 1)(problem_index_array[t1]) ;
            sign_b = sign_grad_prob.col(0) ;  // sign of probs always +'ve
            
            log_terms.setConstant(-700.0);
            sign_terms.setOnes();
            log_sum_result.setConstant(-700.0);
            sign_sum_result.setOnes();
            log_terms.col(0) = log_abs_a;  log_terms.col(1) = log_abs_b;
            sign_terms.col(0) = sign_a; sign_terms.col(1) = sign_b;
            
            log_abs_sum_exp_general_v2( log_terms.leftCols(2), 
                                        sign_terms.leftCols(2),
                                        vect_type, vect_type,  
                                        log_sum_result,
                                        sign_sum_result,
                                        container_max_logs,
                                        container_sum_exp_signed);
    
            log_abs_L_Omega_grad_array_col_for_each_n[t1].col(t1)(problem_index_array[t1]) = log_common_grad_term_1.col(t1)(problem_index_array[t1]) + log_sum_result;
            sign_L_Omega_grad_array_col_for_each_n[t1].col(t1)(problem_index_array[t1]) = sign_sum_result;


        }
        
        }
      
      //////// then w.r.t the third-to-last diagonal .... etc
      {

        for (int i = 3; i < n_tests + 1; i++) {

          int t1 = n_tests - i;

          if (n_problem_array[t1] > 0) {

            //// resize containers to match problem index
            log_abs_z_grad_term.resize(n_problem_array[t1], n_tests);
            sign_z_grad_term.resize(n_problem_array[t1], n_tests);
            log_abs_grad_prob.resize(n_problem_array[t1], n_tests);
            sign_grad_prob.resize(n_problem_array[t1], n_tests);
            log_terms.resize(n_problem_array[t1], n_tests);
            sign_terms.resize(n_problem_array[t1], n_tests);

            log_abs_prod_container_or_inc_array.resize(n_problem_array[t1]);
            sign_prod_container_or_inc_array.resize(n_problem_array[t1]);
            log_sum_result.resize(n_problem_array[t1]);
            sign_sum_result.resize(n_problem_array[t1]);
            container_max_logs.resize(n_problem_array[t1]);
            container_sum_exp_signed.resize(n_problem_array[t1]);

            log_abs_prod_container_or_inc_array_comp.resize(n_problem_array[t1], n_tests);
            sign_prod_container_or_inc_array_comp.resize(n_problem_array[t1], n_tests);

            log_abs_derivs_chain_container_vec_comp.resize(n_problem_array[t1], n_tests);
            sign_derivs_chain_container_vec_comp.resize(n_problem_array[t1], n_tests);

            log_abs_z_grad_term.setConstant(-700.0);
            log_abs_grad_prob.setConstant(-700.0);
            log_abs_prod_container_or_inc_array.setConstant(-700.0);
            log_sum_result.setConstant(-700.0);
            log_terms.setConstant(-700.0);
            container_max_logs.setConstant(-700.0);

            sign_z_grad_term.setOnes();
            sign_grad_prob.setOnes();
            sign_prod_container_or_inc_array.setOnes();
            sign_sum_result.setOnes();
            sign_terms.setOnes();

            log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);
            sign_prod_container_or_inc_array_comp.setOnes();

            log_abs_derivs_chain_container_vec_comp.setConstant(-700.0);
            sign_derivs_chain_container_vec_comp.setOnes();

            log_abs_grad_prob.col(0) = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]) + log_Bound_Z.col(t1)(problem_index_array[t1]) ;
            sign_grad_prob.col(0).array() = sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array() *   sign_Bound_Z.col(t1)(problem_index_array[t1]).array() ;

            // 1st z_grad term
            log_abs_z_grad_term.col(0) = log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1)(problem_index_array[t1]) +  log_Bound_Z.col(t1)(problem_index_array[t1]) ;
            sign_z_grad_term.col(0).array() = -1.0*sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array() *   sign_Bound_Z.col(t1)(problem_index_array[t1]).array() ;

            const double log_abs_L_Omega_double_element = log_abs_L_Omega_double(t1 + 1, t1);
            log_abs_prod_container_or_inc_array.array() = log_abs_L_Omega_double_element +  log_abs_z_grad_term.col(0).array();
            const double sign_L_Omega_double_element = stan::math::sign(L_Omega_double(t1 + 1, t1));
            sign_prod_container_or_inc_array = sign_L_Omega_double_element * sign_z_grad_term.col(0) ; // prod_container_or_inc_array.array().sign();

            // 1st grad_prob update
            log_abs_grad_prob.col(1) = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1)(problem_index_array[t1]) +    log_abs_prod_container_or_inc_array;
            sign_grad_prob.col(1).array() = sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1)(problem_index_array[t1]).array() *   sign_prod_container_or_inc_array.array();

            for (int ii = 1; ii < i - 1; ii++) {

                    log_abs_z_grad_term.col(ii) =  log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + ii)(problem_index_array[t1]) +    log_abs_prod_container_or_inc_array ;
                    sign_z_grad_term.col(ii).array() = -1.0 * sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + ii)(problem_index_array[t1]).array() * sign_prod_container_or_inc_array.array() ;

                    for (int iii = 0; iii < ii + 1; iii++) {
                        // log_abs_prod_container_or_inc_array_comp.col(iii).array() = log_abs_z_grad_term.col(iii).array() +   log_abs_L_Omega_double.row(t1 + ii + 1).segment(t1, ii + 1).transpose().eval()(iii);
                        // sign_prod_container_or_inc_array_comp.col(iii).array() = sign_z_grad_term.col(iii).array() *   stan::math::sign(L_Omega_double.row(t1 + ii + 1).segment(t1, ii + 1).transpose().eval()(iii));
                        const Eigen::Matrix<double, 1, -1> log_abs_L_Omega_row = log_abs_L_Omega_double.row(t1 + ii + 1);
                        const double log_abs_L_Omega_row_iii = log_abs_L_Omega_row.segment(t1, ii + 1).eval()(iii);
                        log_abs_prod_container_or_inc_array_comp.col(iii).array() = log_abs_z_grad_term.col(iii).array() + log_abs_L_Omega_row_iii;
                        const Eigen::Matrix<double, 1, -1> sign_L_Omega_row = stan::math::sign(L_Omega_double.row(t1 + ii + 1));
                        const double sign_L_Omega_row_iii = sign_L_Omega_row.segment(t1, ii + 1).eval()(iii);
                        sign_prod_container_or_inc_array_comp.col(iii) = sign_z_grad_term.col(iii) * sign_L_Omega_row_iii;
                    }

                    log_abs_sum_exp_general_v2(log_abs_prod_container_or_inc_array_comp.leftCols(ii + 1),
                                               sign_prod_container_or_inc_array_comp.leftCols(ii + 1),
                                               vect_type, vect_type,
                                               log_abs_prod_container_or_inc_array,  // computing this
                                               sign_prod_container_or_inc_array,     // computing this
                                               container_max_logs,
                                               container_sum_exp_signed);

                    // Update grad_prob
                    log_abs_grad_prob.col(ii + 1) =   log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + ii + 1)(problem_index_array[t1]) +  log_abs_prod_container_or_inc_array;
                    sign_grad_prob.col(ii + 1).array() = sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + ii + 1)(problem_index_array[t1]).array() *  sign_prod_container_or_inc_array.array();

            }

            {

              for (int iii = 0; iii < i; iii++) {
                log_abs_derivs_chain_container_vec_comp.col(iii) = log_abs_grad_prob.col(iii) + log_prop_rowwise_prod_temp.col(t1)(problem_index_array[t1]) - 1.0*y1_log_prob.col(t1 + iii)(problem_index_array[t1]);
                sign_derivs_chain_container_vec_comp.col(iii) = sign_grad_prob.col(iii);
              }

              log_abs_sum_exp_general_v2(log_abs_derivs_chain_container_vec_comp.leftCols(i),
                                         sign_derivs_chain_container_vec_comp.leftCols(i),
                                         vect_type,  vect_type,
                                         log_sum_result,  // computing this
                                         sign_sum_result, // computing this
                                         container_max_logs,
                                         container_sum_exp_signed);

              log_abs_L_Omega_grad_array_col_for_each_n[t1].col(t1)(problem_index_array[t1]) = log_common_grad_term_1.col(t1)(problem_index_array[t1]) + log_sum_result;
              sign_L_Omega_grad_array_col_for_each_n[t1].col(t1)(problem_index_array[t1]) = sign_sum_result;

            }

          }

        }


      }
        
        

        
        ////  OK up to here -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        

      {
              int t1_dash = 0;  // t1 = n_tests - 1
              int t1 = n_tests - (t1_dash + 1); //  starts at n_tests - 1;  // if t1_dash = 0 -> t1 = T - 1

              if (n_problem_array[t1] > 0) {

                //// resize containers to match problem index
                log_abs_z_grad_term.resize(n_problem_array[t1], n_tests);
                sign_z_grad_term.resize(n_problem_array[t1], n_tests);
                log_abs_grad_prob.resize(n_problem_array[t1], n_tests);
                sign_grad_prob.resize(n_problem_array[t1], n_tests);
                log_terms.resize(n_problem_array[t1], n_tests);
                sign_terms.resize(n_problem_array[t1], n_tests);

                log_abs_prod_container_or_inc_array.resize(n_problem_array[t1]);
                sign_prod_container_or_inc_array.resize(n_problem_array[t1]);
                log_sum_result.resize(n_problem_array[t1]);
                sign_sum_result.resize(n_problem_array[t1]);
                container_max_logs.resize(n_problem_array[t1]);
                container_sum_exp_signed.resize(n_problem_array[t1]);

                log_abs_prod_container_or_inc_array_comp.resize(n_problem_array[t1], n_tests);
                sign_prod_container_or_inc_array_comp.resize(n_problem_array[t1], n_tests);

                log_abs_derivs_chain_container_vec_comp.resize(n_problem_array[t1], n_tests);
                sign_derivs_chain_container_vec_comp.resize(n_problem_array[t1], n_tests);

                log_abs_z_grad_term.setConstant(-700.0);
                log_abs_grad_prob.setConstant(-700.0);
                log_abs_prod_container_or_inc_array.setConstant(-700.0);
                log_sum_result.setConstant(-700.0);
                log_terms.setConstant(-700.0);
                container_max_logs.setConstant(-700.0);

                sign_z_grad_term.setOnes();
                sign_grad_prob.setOnes();
                sign_prod_container_or_inc_array.setOnes();
                sign_sum_result.setOnes();
                sign_terms.setOnes();

                log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);
                sign_prod_container_or_inc_array_comp.setOnes();

                log_abs_derivs_chain_container_vec_comp.setConstant(-700.0);
                sign_derivs_chain_container_vec_comp.setOnes();

                  int t2 = n_tests - (t1_dash + 2); //  starts at n_tests - 2;

                    sign_L_Omega_grad_array_col_for_each_n[t1].col(t2)(problem_index_array[t1]).array()  =      sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array() *  sign_Z_std_norm.col(t2)(problem_index_array[t1]).array()  ;
                    log_abs_L_Omega_grad_array_col_for_each_n[t1].col(t2)(problem_index_array[t1]) = log_common_grad_term_1.col(t1)(problem_index_array[t1]) + log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]) +  log_Z_std_norm.col(t2)(problem_index_array[t1]);

                  if (t1 > 1) { // starts at  L_{T, T-2}
                    {
                      t2 =   n_tests - (t1_dash + 3); // starts at n_tests - 3;
                      sign_L_Omega_grad_array_col_for_each_n[t1].col(t2)(problem_index_array[t1]).array()  =   sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array() *  sign_Z_std_norm.col(t2)(problem_index_array[t1]).array()  ;
                      log_abs_L_Omega_grad_array_col_for_each_n[t1].col(t2)(problem_index_array[t1]) = log_common_grad_term_1.col(t1)(problem_index_array[t1]) + log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]) +  log_Z_std_norm.col(t2)(problem_index_array[t1]);
                    }
                  }

                  if (t1 > 2) {// starts at  L_{T, T-3}
                    for (int t2_dash = 3; t2_dash < n_tests; t2_dash++ ) { // t2 < t1
                      t2 = n_tests - (t1_dash + t2_dash + 1); // starts at T - 4
                      if (t2 < n_tests - 1) {
                        sign_L_Omega_grad_array_col_for_each_n[t1].col(t2)(problem_index_array[t1]).array() =  sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array() *  sign_Z_std_norm.col(t2)(problem_index_array[t1]).array()  ;
                        log_abs_L_Omega_grad_array_col_for_each_n[t1].col(t2)(problem_index_array[t1]) =  log_common_grad_term_1.col(t1)(problem_index_array[t1]) + log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]) +  log_Z_std_norm.col(t2)(problem_index_array[t1]);
                      }
                    }
                  }

              }
      }


        
        
        
      /////////////////// then rest of rows (second-to-last row, third-to-last row, ..., first row)
      for (int t1_dash = 1; t1_dash < n_tests - 1; t1_dash++) {
        int t1 = n_tests - (t1_dash + 1);

        if (n_problem_array[t1] > 0) {

          //// resize containers to match problem index
          log_abs_z_grad_term.resize(n_problem_array[t1], n_tests);
          sign_z_grad_term.resize(n_problem_array[t1], n_tests);
          log_abs_grad_prob.resize(n_problem_array[t1], n_tests);
          sign_grad_prob.resize(n_problem_array[t1], n_tests);
          log_terms.resize(n_problem_array[t1], n_tests);
          sign_terms.resize(n_problem_array[t1], n_tests);

          log_abs_prod_container_or_inc_array.resize(n_problem_array[t1]);
          sign_prod_container_or_inc_array.resize(n_problem_array[t1]);
          log_sum_result.resize(n_problem_array[t1]);
          sign_sum_result.resize(n_problem_array[t1]);
          container_max_logs.resize(n_problem_array[t1]);
          container_sum_exp_signed.resize(n_problem_array[t1]);

          log_abs_prod_container_or_inc_array_comp.resize(n_problem_array[t1], n_tests);
          sign_prod_container_or_inc_array_comp.resize(n_problem_array[t1], n_tests);

          log_abs_derivs_chain_container_vec_comp.resize(n_problem_array[t1], n_tests);
          sign_derivs_chain_container_vec_comp.resize(n_problem_array[t1], n_tests);

          log_abs_z_grad_term.setConstant(-700.0);
          log_abs_grad_prob.setConstant(-700.0);
          log_abs_prod_container_or_inc_array.setConstant(-700.0);
          log_sum_result.setConstant(-700.0);
          log_terms.setConstant(-700.0);
          container_max_logs.setConstant(-700.0);

          sign_z_grad_term.setOnes();
          sign_grad_prob.setOnes();
          sign_prod_container_or_inc_array.setOnes();
          sign_sum_result.setOnes();
          sign_terms.setOnes();

          log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);
          sign_prod_container_or_inc_array_comp.setOnes();

          log_abs_derivs_chain_container_vec_comp.setConstant(-700.0);
          sign_derivs_chain_container_vec_comp.setOnes();


            for (int t2_dash = t1_dash + 1; t2_dash < n_tests; t2_dash++) {
              int t2 = n_tests - (t2_dash + 1); // starts at t1 - 1, then t1 - 2, up to 0

              {

                log_abs_prod_container_or_inc_array = log_Z_std_norm.col(t2)(problem_index_array[t1]) ;
                sign_prod_container_or_inc_array = sign_Z_std_norm.col(t2)(problem_index_array[t1]) ;
                // prod_container.array()  =  Z_std_norm.col(t2).array() ; // block(0, t2, index_size, t1 - t2) * deriv_L_t1.head(t1 - t2) ;

                // 1st grad_prob setup
                log_abs_grad_prob.col(0) = log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]) +    log_abs_prod_container_or_inc_array;
                sign_grad_prob.col(0).array() = sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array() *  sign_prod_container_or_inc_array.array();

                // 1st z_grad term
                log_abs_z_grad_term.col(0) = log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1)(problem_index_array[t1]) + log_abs_prod_container_or_inc_array ;
                sign_z_grad_term.col(0).array() =    sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1)(problem_index_array[t1]).array() *   -1.0 * sign_prod_container_or_inc_array.array();
                //
                // log_abs_prod_container_or_inc_array.array() =   (log_abs_L_Omega_double.row(t1 + 1).segment(t1 + 0, 1).transpose()).array() + log_abs_z_grad_term.col(0).array();  /// ?
                // sign_prod_container_or_inc_array.array() = stan::math::sign(L_Omega_double.row(t1 + 1).segment(t1 + 0, 1).transpose()).array() * sign_grad_prob.col(0).array();  /// ?

                if (t1_dash > 0) {
                  for (int t1_dash_dash = 1; t1_dash_dash < t1_dash + 1; t1_dash_dash++) {

                    if (t1_dash_dash > 1) {
                      log_abs_z_grad_term.col(t1_dash_dash - 1) =   log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + t1_dash_dash - 1)(problem_index_array[t1]) +  log_abs_prod_container_or_inc_array;
                      sign_z_grad_term.col(t1_dash_dash - 1).array() = sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + t1_dash_dash - 1)(problem_index_array[t1]).array() * -1.0 * sign_prod_container_or_inc_array.array() ;
                    }


                    // prod_container  =    z_grad_term.leftCols(t1_dash_dash) *   L_Omega_double.row(t1 + t1_dash_dash).segment(t1, t1_dash_dash).transpose()   ;
                    for (int iii = 0; iii < t1_dash_dash; iii++) {
                      const Eigen::Matrix<double, 1, -1> log_abs_L_Omega_row = log_abs_L_Omega_double.row(t1 + t1_dash_dash);
                      const double log_abs_L_Omega_row_iii = log_abs_L_Omega_row.segment(t1, t1_dash_dash).eval()(iii);
                      log_abs_prod_container_or_inc_array_comp.col(iii).array() =  log_abs_z_grad_term.col(iii).array() + log_abs_L_Omega_row_iii;
                      const Eigen::Matrix<double, 1, -1> sign_L_Omega_row = stan::math::sign(L_Omega_double.row(t1 + t1_dash_dash));
                      const double sign_L_Omega_row_iii = sign_L_Omega_row.segment(t1, t1_dash_dash).eval()(iii);
                      sign_prod_container_or_inc_array_comp.col(iii).array() =   sign_z_grad_term.col(iii).array() * sign_L_Omega_row_iii;
                    }

                    log_abs_sum_exp_general_v2(  log_abs_prod_container_or_inc_array_comp.leftCols(t1_dash_dash),
                                                 sign_prod_container_or_inc_array_comp.leftCols(t1_dash_dash),
                                                 vect_type,  vect_type,
                                                 log_abs_prod_container_or_inc_array, // computing this
                                                 sign_prod_container_or_inc_array,    // computing this
                                                 container_max_logs,
                                                 container_sum_exp_signed);

                    log_abs_grad_prob.col(t1_dash_dash) =    log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + t1_dash_dash)(problem_index_array[t1])  +  log_abs_prod_container_or_inc_array;
                    sign_grad_prob.col(t1_dash_dash).array() =  sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + t1_dash_dash)(problem_index_array[t1]).array() *  sign_prod_container_or_inc_array.array();

                  }
                }

                /////// final computations
                {
                  for (int ii = 0; ii < t1_dash + 1; ii++) {
                    log_abs_derivs_chain_container_vec_comp.col(ii) =  log_abs_grad_prob.col(ii) +  log_prop_rowwise_prod_temp.col(t1)(problem_index_array[t1]) +  (-1.0*y1_log_prob).col(t1 + ii)(problem_index_array[t1]);
                    sign_derivs_chain_container_vec_comp.col(ii) = sign_grad_prob.col(ii);
                  }

                  log_abs_sum_exp_general_v2( log_abs_derivs_chain_container_vec_comp.leftCols(t1_dash + 1),
                                              sign_derivs_chain_container_vec_comp.leftCols(t1_dash + 1),
                                              vect_type, vect_type,
                                              log_sum_result,    // computing this
                                              sign_sum_result,   // computing this
                                              container_max_logs,
                                              container_sum_exp_signed);

                  log_abs_L_Omega_grad_array_col_for_each_n[t1].col(t2)(problem_index_array[t1]) = log_common_grad_term_1.col(t1)(problem_index_array[t1]) + log_sum_result;
                  sign_L_Omega_grad_array_col_for_each_n[t1].col(t2)(problem_index_array[t1]) = sign_sum_result;

                }

              }

            }

      }

      }
      
        //// once done, resize containers back to chunk_size
        log_abs_z_grad_term.resize(chunk_size, n_tests);
        sign_z_grad_term.resize(chunk_size, n_tests);
        log_abs_grad_prob.resize(chunk_size, n_tests);
        sign_grad_prob.resize(chunk_size, n_tests);
        log_terms.resize(chunk_size, n_tests);
        sign_terms.resize(chunk_size, n_tests);
        log_abs_prod_container_or_inc_array.resize(chunk_size);
        sign_prod_container_or_inc_array.resize(chunk_size);
        log_abs_a.resize(chunk_size);
        log_abs_b.resize(chunk_size);
        sign_a.resize(chunk_size);
        sign_b.resize(chunk_size);
        log_sum_result.resize(chunk_size);
        sign_sum_result.resize(chunk_size);
        container_max_logs.resize(chunk_size);
        container_sum_exp_signed.resize(chunk_size); 
        
        log_abs_prod_container_or_inc_array_comp.resize(chunk_size, n_tests);
        sign_prod_container_or_inc_array_comp.resize(chunk_size, n_tests);
        
        log_abs_derivs_chain_container_vec_comp.resize(chunk_size, n_tests);
        sign_derivs_chain_container_vec_comp.resize(chunk_size, n_tests);

}
  

 