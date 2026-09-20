
//// MVP_log_scale_grad_calc_fns_T.hpp
////
//// Templated (<Vec vec>) versions of everything in MVP_log_scale_grad_calc_fns.hpp, plus the
//// three signed log-sum-exp utilities those functions need. Every function here ends in _T and
//// takes NO Model_fn_args_struct / vect_type strings; your originals are untouched.
////
//// Requires: fn_dispatch_templated.hpp, MVP_helpers_migrated.hpp (KernelChoice, apply_*).
////
//// Maths is identical to your originals. Only the element-wise calls changed:
////   fn_EIGEN_double(x, "log", vt)  ->  apply_inplace<vec, Fn::log>(x)      (in place)
////   log_abs_sum_exp_general_v2(A, S, vt, vt, out_log, out_sign, mx, sm) -> log_abs_sum_exp_general_v2_T<vec>(A, S, out_log, out_sign, mx, sm)
////   log_sum_vec_signed_v1(la, sg, vt)  ->  log_sum_vec_signed_T<vec>(la, sg)
////   fn_log_sum_exp_2d_double(M, vt)    ->  fn_log_sum_exp_2d_T<vec>(M)

#pragma once
// #include "MVP_helpers_migrated.hpp"

//// =====================================================================================
//// 0. Signed log-sum-exp utilities
//// =====================================================================================

//// Row-wise: out_log = log|sum_j sign_j exp(log_j)|, out_sign = sign(sum). All k columns of
//// log_terms/sign_terms are used (pass .leftCols(k) at the call site). out_log doubles as scratch.
template <Vec vec>
inline void log_abs_sum_exp_general_v2_T(  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_terms,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_terms,
                                           Eigen::Matrix<double, -1, 1> &out_log,
                                           Eigen::Matrix<double, -1, 1> &out_sign,
                                           Eigen::Matrix<double, -1, 1> &container_max_logs,
                                           Eigen::Matrix<double, -1, 1> &container_sum_exp_signed
) {
        
        const int n = static_cast<int>(log_terms.rows());
        const int k = static_cast<int>(log_terms.cols());
        container_max_logs = log_terms.rowwise().maxCoeff();
        container_sum_exp_signed.setZero();
        for (int j = 0; j < k; ++j) {
          out_log = log_terms.col(j) - container_max_logs;                 //// scratch
          apply_inplace<vec, Fn::exp>(out_log);
          container_sum_exp_signed.array() += sign_terms.col(j).array() * out_log.array();
        }
        for (int i = 0; i < n; ++i) {
          const double s = container_sum_exp_signed(i);
          out_sign(i) = (s > 0.0) ? 1.0 : ((s < 0.0) ? -1.0 : 1.0);
          out_log(i)  = std::abs(s);
        }
        apply_inplace<vec, Fn::log>(out_log);                                //// log(0) -> -Inf is fine (Stan log)
        for (int i = 0; i < n; ++i) if (!std::isfinite(out_log(i))) out_log(i) = -700.0;
        out_log.array() += container_max_logs.array();
  
}




//// Whole-vector signed sum: log|sum_i sign_i exp(log_i)| and its sign.
struct LogSumSignedResult_T { double log_sum; double sign; };

template <Vec vec>
inline LogSumSignedResult_T log_sum_vec_signed_T(  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> log_abs,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> sign
) {
  
        const double mx = log_abs.maxCoeff();
        Eigen::Matrix<double, -1, 1> tmp = log_abs.array() - mx;
        apply_inplace<vec, Fn::exp>(tmp);
        const double s = (sign.array() * tmp.array()).sum();
        LogSumSignedResult_T r;
        r.sign    = (s < 0.0) ? -1.0 : 1.0;
        r.log_sum = (s == 0.0) ? -700.0 : std::log(std::abs(s)) + mx;
        return r;
  
}




//// Row-wise log-sum-exp of an n x 2 matrix (all positive terms).
template <Vec vec>
inline Eigen::Matrix<double, -1, 1> fn_log_sum_exp_2d_T( const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> M) {
  
        Eigen::Matrix<double, -1, 1> mx = M.rowwise().maxCoeff();
        Eigen::Matrix<double, -1, 1> a = M.col(0) - mx;  apply_inplace<vec, Fn::exp>(a);
        Eigen::Matrix<double, -1, 1> b = M.col(1) - mx;  apply_inplace<vec, Fn::exp>(b);
        Eigen::Matrix<double, -1, 1> out = a + b;
        apply_inplace<vec, Fn::log>(out);
        out.array() += mx.array();
        return out;
  
}




//// =====================================================================================
//// 1. GHK log-scale fix-ups for the problem rows (index). Eigen indexed views need a
////    temporary per expression; those allocations are unavoidable and small (|index| rows).
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_lp_GHK_cols_log_scale_underflow_T(  const int t,
                                                               const std::vector<int> &index,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_Z_std_norm,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array
) {
  
        const double a_times_3 = 3.0 * 0.07056, b = 1.5976;
        typedef Eigen::Matrix<double, -1, 1> V;
      
        V log_Bound_U_Phi_Bound_Z = Bound_Z(index, t);
        apply_inplace<vec, Fn::log_Phi_approx>(log_Bound_U_Phi_Bound_Z);
        V tmp = log_Bound_U_Phi_Bound_Z;  apply_inplace<vec, Fn::exp>(tmp);
        Bound_U_Phi_Bound_Z(index, t) = tmp;
      
        V u_log = u_array(index, t);  apply_inplace<vec, Fn::log>(u_log);
        V log_Phi_Z = u_log + log_Bound_U_Phi_Bound_Z;
        tmp = log_Phi_Z;  apply_inplace<vec, Fn::exp>(tmp);
        Phi_Z(index, t) = tmp;
      
        V log_1m_Phi_Z = Bound_U_Phi_Bound_Z(index, t).array() * u_array(index, t).array();
        apply_inplace<vec, Fn::log1m>(log_1m_Phi_Z);
        V logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
        apply_inplace<vec, Fn::inv_Phi_approx_from_logit_prob>(logit_Phi_Z);
        Z_std_norm(index, t) = logit_Phi_Z;
      
        tmp = logit_Phi_Z.cwiseAbs();  apply_inplace<vec, Fn::log>(tmp);
        log_Z_std_norm(index, t) = tmp;
      
        y1_log_prob(index, t) = log_Bound_U_Phi_Bound_Z;
        prob(index, t)        = Bound_U_Phi_Bound_Z(index, t);
      
        V log_Bound_U_Phi_Bound_Z_1m = Bound_U_Phi_Bound_Z(index, t);
        apply_inplace<vec, Fn::log1m>(log_Bound_U_Phi_Bound_Z_1m);
      
        tmp = Bound_Z(index, t);
        tmp.array() = a_times_3 * tmp.array().square() + b;
        apply_inplace<vec, Fn::log>(tmp);
        tmp.array() += log_Bound_U_Phi_Bound_Z.array() + log_Bound_U_Phi_Bound_Z_1m.array();
        log_phi_Bound_Z(index, t) = tmp;
      
        tmp = Z_std_norm(index, t);
        tmp.array() = a_times_3 * tmp.array().square() + b;
        apply_inplace<vec, Fn::log>(tmp);
        tmp.array() += log_Phi_Z.array() + log_1m_Phi_Z.array();
        log_phi_Z_recip(index, t) = -tmp;
  
}




template <Vec vec>
inline void fn_MVP_compute_lp_GHK_cols_log_scale_overflow_T(  const int t,
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
                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array
) {
  
        const double a_times_3 = 3.0 * 0.07056, b = 1.5976;
        typedef Eigen::Matrix<double, -1, 1> V;
      
        V log_Bound_U_Phi_Bound_Z_1m = -Bound_Z(index, t);
        apply_inplace<vec, Fn::log_Phi_approx>(log_Bound_U_Phi_Bound_Z_1m);
        V Bound_U_Phi_Bound_Z_1m = log_Bound_U_Phi_Bound_Z_1m;  apply_inplace<vec, Fn::exp>(Bound_U_Phi_Bound_Z_1m);
        V log_Bound_U_Phi_Bound_Z = Bound_U_Phi_Bound_Z_1m;     apply_inplace<vec, Fn::log1m>(log_Bound_U_Phi_Bound_Z);
      
        Bound_U_Phi_Bound_Z(index, t).array() = 1.0 - Bound_U_Phi_Bound_Z_1m.array();
      
        Eigen::Matrix<double, -1, -1> lse2(num_overflows, 2);
        V u_log = u_array(index, t);  apply_inplace<vec, Fn::log>(u_log);
        lse2.col(0) = log_Bound_U_Phi_Bound_Z_1m + u_log;
        lse2.col(1) = log_Bound_U_Phi_Bound_Z;
        V log_Phi_Z = fn_log_sum_exp_2d_T<vec>(lse2);
      
        V tmp = log_Phi_Z;  apply_inplace<vec, Fn::exp>(tmp);
        Phi_Z(index, t) = tmp;
      
        V log_1m_Phi_Z = u_array(index, t);  apply_inplace<vec, Fn::log1m>(log_1m_Phi_Z);
        log_1m_Phi_Z.array() += log_Bound_U_Phi_Bound_Z_1m.array();
      
        V logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
        apply_inplace<vec, Fn::inv_Phi_approx_from_logit_prob>(logit_Phi_Z);
        Z_std_norm(index, t) = logit_Phi_Z;
        tmp = logit_Phi_Z.cwiseAbs();  apply_inplace<vec, Fn::log>(tmp);
        log_Z_std_norm(index, t) = tmp;
      
        y1_log_prob(index, t) = log_Bound_U_Phi_Bound_Z_1m;
        prob(index, t)        = Bound_U_Phi_Bound_Z_1m;
      
        tmp = Bound_Z(index, t);
        tmp.array() = a_times_3 * tmp.array().square() + b;
        apply_inplace<vec, Fn::log>(tmp);
        tmp.array() += log_Bound_U_Phi_Bound_Z.array() + log_Bound_U_Phi_Bound_Z_1m.array();
        log_phi_Bound_Z(index, t) = tmp;
      
        tmp = Z_std_norm(index, t);
        tmp.array() = a_times_3 * tmp.array().square() + b;
        apply_inplace<vec, Fn::log>(tmp);
        tmp.array() += log_Phi_Z.array() + log_1m_Phi_Z.array();
        log_phi_Z_recip(index, t) = -tmp;
  
}

//// =====================================================================================
//// 2. grad prep on the log scale
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_grad_prep_log_scale_T(  Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_prob_rowwise_prod_temp,
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
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  log_prev_per_obs_given_c,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_recip_double,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_L_Omega_recip_double,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_sign_chunk,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_m_y_sign_x_u,
                                           const int n_class
) {
  
        const int n_tests = y1_log_prob.cols();
        const int chunk_size = y1_log_prob.rows();
      
        for (int i = 0; i < n_tests; i++) {
          const int t = n_tests - (i + 1);
          log_prob_rowwise_prod_temp.col(t)       = y1_log_prob.block(0, t, chunk_size, i + 1).rowwise().sum();
          log_prob_recip_rowwise_prod_temp.col(t) = y1_log_prob_recip.block(0, t, chunk_size, i + 1).rowwise().sum();
        }
        log_prob_rowwise_prod_temp_all = y1_log_prob.rowwise().sum();
      
        if (n_class > 1) {
          for (int i = 0; i < n_tests; i++) {
            const int t = n_tests - (i + 1);
            log_common_grad_term_1.col(t).array() = log_prob_n_recip.array() + log_prev_per_obs_given_c.array()
                                                  + log_prob_rowwise_prod_temp_all.array()
                                                  + log_prob_recip_rowwise_prod_temp.col(t).array();
          }
        } else {
          log_common_grad_term_1.setConstant(-700.0);
        }
      
        Eigen::Matrix<double, -1, 1> log_abs_col(chunk_size);
        for (int t = 0; t < n_tests; t++) {
          const double L_log_abs = log_abs_L_Omega_recip_double(t, t);
          const double L_sign    = sign_L_Omega_recip_double(t, t);
      
          log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array() = log_phi_Bound_Z.col(t).array() + L_log_abs;
      
          log_abs_col = y_m_y_sign_x_u.col(t).cwiseAbs();
          apply_inplace<vec, Fn::log>(log_abs_col);
          log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array() =
              log_abs_col.array() + log_phi_Z_recip.col(t).array() + log_phi_Bound_Z.col(t).array() + L_log_abs;
      
          sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t) = y_sign_chunk.col(t).array().sign() * L_sign;
          sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t) = y_m_y_sign_x_u.col(t).array().sign() * L_sign;
        }
  
}




//// Small helper used by the three grad functions: exp() of an indexed view, times a sign, into an indexed view.
template <Vec vec>
ALWAYS_INLINE void exp_indexed_times_sign_T(Eigen::Matrix<double, -1, -1> &dst, const int col, const std::vector<int> &idx,
                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_src, const int src_col,
                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> sign) {
  
        Eigen::Matrix<double, -1, 1> tmp = log_abs_src.col(src_col)(idx);
        apply_inplace<vec, Fn::exp>(tmp);
        dst.col(col)(idx) = tmp.array() * sign.array();
  
}




//// =====================================================================================
//// 3. nuisance grad on the log scale (problem rows only)
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_nuisance_grad_log_scale_T(  const std::vector<int> &n_problem_array,
                                                       const std::vector<std::vector<int>> &problem_index_array,
                                                       Eigen::Matrix<double, -1, -1> &log_abs_u_grad_array_CM_chunk,
                                                       Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prob_recip,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prob_rowwise_prod_temp,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ys,     //// log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ys,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ym,     //// log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ym,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_common_grad_term_1,
                                                       Eigen::Matrix<double, -1, -1> &log_abs_z_grad_term,
                                                       Eigen::Matrix<double, -1, -1> &sign_z_grad_term,
                                                       Eigen::Matrix<double, -1, -1> &log_abs_grad_prob,
                                                       Eigen::Matrix<double, -1, -1> &sign_grad_prob,
                                                       Eigen::Matrix<double, -1, 1>  &log_abs_prod,
                                                       Eigen::Matrix<double, -1, 1>  &sign_prod,
                                                       Eigen::Matrix<double, -1, 1>  &log_sum_result,
                                                       Eigen::Matrix<double, -1, 1>  &sign_sum_result,
                                                       Eigen::Matrix<double, -1, -1> &log_terms,
                                                       Eigen::Matrix<double, -1, -1> &sign_terms,
                                                       Eigen::Matrix<double, -1, 1>  &log_abs_a,
                                                       Eigen::Matrix<double, -1, 1>  &log_abs_b,
                                                       Eigen::Matrix<double, -1, 1>  &sign_a,
                                                       Eigen::Matrix<double, -1, 1>  &sign_b,
                                                       Eigen::Matrix<double, -1, 1>  &container_max_logs,
                                                       Eigen::Matrix<double, -1, 1>  &container_sum_exp_signed
) {
  
        const int chunk_size = u_grad_array_CM_chunk.rows();
        const int n_tests    = u_grad_array_CM_chunk.cols();
      
        auto resize_all = [&](const int m) {
          log_abs_z_grad_term.resize(m, n_tests); sign_z_grad_term.resize(m, n_tests);
          log_abs_grad_prob.resize(m, n_tests);   sign_grad_prob.resize(m, n_tests);
          log_terms.resize(m, n_tests);           sign_terms.resize(m, n_tests);
          log_abs_prod.resize(m); sign_prod.resize(m); log_abs_a.resize(m); log_abs_b.resize(m); sign_a.resize(m); sign_b.resize(m);
          log_sum_result.resize(m); sign_sum_result.resize(m); container_max_logs.resize(m); container_sum_exp_signed.resize(m);
          log_abs_z_grad_term.setConstant(-700.0); log_abs_grad_prob.setConstant(-700.0); log_abs_prod.setConstant(-700.0);
          log_sum_result.setConstant(-700.0); log_terms.setConstant(-700.0); log_abs_a.setConstant(-700.0); log_abs_b.setConstant(-700.0);
          container_max_logs.setConstant(-700.0);
          sign_z_grad_term.setOnes(); sign_grad_prob.setOnes(); sign_prod.setOnes(); sign_sum_result.setOnes(); sign_terms.setOnes();
          sign_a.setOnes(); sign_b.setOnes(); container_sum_exp_signed.setZero();
        };
        resize_all(chunk_size);
      
        {  //// second-to-last column
          const int t = n_tests - 1;
          if (n_problem_array[t] > 0) {
            const std::vector<int> &idx = problem_index_array[t];
            log_abs_u_grad_array_CM_chunk.col(n_tests - 2)(idx) =
                log_common_grad_term_1.col(t)(idx).array() + log_abs_ys.col(t)(idx).array() + log_abs_L_Omega_double(t, t - 1)
                + log_phi_Z_recip.col(t - 1)(idx).array() + y1_log_prob.col(t - 1)(idx).array();
            Eigen::Matrix<double, -1, 1> sg = sign_ys.col(t)(idx).array() * stan::math::sign(L_Omega_double(t, t - 1));
            exp_indexed_times_sign_T<vec>(u_grad_array_CM_chunk, n_tests - 2, idx, log_abs_u_grad_array_CM_chunk, n_tests - 2, sg);
          }
        }
      
        {  //// third-to-last column
          const int t = n_tests - 2;
          if (n_problem_array[t] > 0) {
            const std::vector<int> &idx = problem_index_array[t];
            resize_all(n_problem_array[t]);
      
            log_abs_z_grad_term.col(0) = log_phi_Z_recip.col(t - 1)(idx) + y1_log_prob.col(t - 1)(idx);
            sign_z_grad_term.col(0).setOnes();
            log_abs_z_grad_term.col(1).array() = log_abs_L_Omega_double(t, t - 1) + log_abs_z_grad_term.col(0).array() + log_abs_ym.col(t)(idx).array();
            sign_z_grad_term.col(1).array()    = stan::math::sign(L_Omega_double(t, t - 1)) * sign_z_grad_term.col(0).array() * sign_ym.col(t)(idx).array();
      
            log_abs_prod.array() = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t, t - 1);
            sign_prod.array()    = sign_z_grad_term.col(0).array() * stan::math::sign(L_Omega_double(t, t - 1));
            log_abs_grad_prob.col(0).array() = log_abs_ys.col(t)(idx).array() + log_abs_prod.array();
            sign_grad_prob.col(0).array()    = sign_ys.col(t)(idx).array() * sign_prod.array();
      
            log_abs_a.array() = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t + 1, t - 1);
            log_abs_b.array() = log_abs_z_grad_term.col(1).array() + log_abs_L_Omega_double(t + 1, t);
            sign_a.array() =  sign_z_grad_term.col(0).array() * stan::math::sign(L_Omega_double(t + 1, t - 1));
            sign_b.array() = -sign_z_grad_term.col(1).array() * stan::math::sign(L_Omega_double(t + 1, t));
            log_terms.col(0) = log_abs_a; log_terms.col(1) = log_abs_b; sign_terms.col(0) = sign_a; sign_terms.col(1) = sign_b;
            log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(2), sign_terms.leftCols(2), log_abs_prod, sign_prod, container_max_logs, container_sum_exp_signed);
      
            log_abs_grad_prob.col(1).array() = log_abs_ys.col(t + 1)(idx).array() + log_abs_prod.array();
            sign_grad_prob.col(1).array()    = sign_ys.col(t + 1)(idx).array() * sign_prod.array();
      
            log_abs_a = log_abs_grad_prob.col(0) + y1_log_prob.col(t + 1)(idx);
            log_abs_b = log_abs_grad_prob.col(1) + y1_log_prob.col(t)(idx);
            sign_a = sign_grad_prob.col(0); sign_b = sign_grad_prob.col(1);
            log_terms.col(0) = log_abs_a; log_terms.col(1) = log_abs_b; sign_terms.col(0) = sign_a; sign_terms.col(1) = sign_b;
            log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(2), sign_terms.leftCols(2), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
      
            log_abs_u_grad_array_CM_chunk.col(n_tests - 3)(idx) = log_common_grad_term_1.col(t)(idx) + log_sum_result;
            exp_indexed_times_sign_T<vec>(u_grad_array_CM_chunk, n_tests - 3, idx, log_abs_u_grad_array_CM_chunk, n_tests - 3, sign_sum_result);
          }
        }
      
        for (int i = 1; i < n_tests - 2; i++) {
          const int t = n_tests - (i + 2);
          if (n_problem_array[t] > 0) {
            const std::vector<int> &idx = problem_index_array[t];
            resize_all(n_problem_array[t]);
      
            log_abs_z_grad_term.col(0) = log_phi_Z_recip.col(t - 1)(idx) + y1_log_prob.col(t - 1)(idx);
            sign_z_grad_term.col(0).setOnes();
            log_abs_prod = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t, t - 1);
            sign_prod    = sign_z_grad_term.col(0).array() * stan::math::sign(L_Omega_double(t, t - 1));
            log_abs_grad_prob.col(0) = log_abs_ys.col(t)(idx).array() + log_abs_prod.array();
            sign_grad_prob.col(0)    = sign_ys.col(t)(idx).array() * sign_prod.array();
      
            for (int ii = 1; ii < i + 2; ii++) {
              log_abs_z_grad_term.col(ii).array() = log_abs_ym.col(t + ii - 1)(idx).array() + log_abs_prod.array();
              sign_z_grad_term.col(ii).array()    = sign_ym.col(t + ii - 1)(idx).array() * sign_prod.array();
              log_terms.col(0).array()  = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t + ii, t - 1);
              sign_terms.col(0).array() = sign_z_grad_term.col(0).array() * stan::math::sign(L_Omega_double(t + ii, t - 1));
              for (int j = 1; j <= ii; j++) {
                log_terms.col(j).array()  = log_abs_z_grad_term.col(j).array() + log_abs_L_Omega_double(t + ii, t + j - 1);
                sign_terms.col(j).array() = -sign_z_grad_term.col(j).array() * stan::math::sign(L_Omega_double(t + ii, t + j - 1));
              }
              log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(ii), sign_terms.leftCols(ii), log_abs_prod, sign_prod, container_max_logs, container_sum_exp_signed);
              log_abs_grad_prob.col(ii).array() = log_abs_ys.col(t + ii)(idx).array() + log_abs_prod.array();
              sign_grad_prob.col(ii).array()    = sign_ys.col(t + ii)(idx).array() * sign_prod.array();
            }
      
            log_terms.setConstant(-700.0); sign_terms.setOnes();
            for (int ii = 0; ii < i + 2; ii++) {
              log_terms.col(ii)  = log_abs_grad_prob.col(ii).array() + log_prob_rowwise_prod_temp.col(t)(idx).array() + log_prob_recip.col(t + ii)(idx).array();
              sign_terms.col(ii) = sign_grad_prob.col(ii);
            }
            log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(i + 2), sign_terms.leftCols(i + 2), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
      
            log_abs_u_grad_array_CM_chunk.col(n_tests - (i + 3))(idx) = log_common_grad_term_1.col(t)(idx).array() + log_sum_result.array();
            exp_indexed_times_sign_T<vec>(u_grad_array_CM_chunk, n_tests - (i + 3), idx, log_abs_u_grad_array_CM_chunk, n_tests - (i + 3), sign_sum_result);
          }
        }
      
        resize_all(chunk_size);
  
}




//// =====================================================================================
//// 4. coefficients grad on the log scale (intercept-only, problem rows only)
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_coefficients_grad_log_scale_T(  const std::vector<int> &n_problem_array,
                                                           const std::vector<std::vector<int>> &problem_index_array,
                                                           Eigen::Matrix<double, -1, -1> &beta_grad_array,
                                                           std::vector<Eigen::Matrix<double, -1, -1>> &sign_beta_grad_array_for_each_n,
                                                           std::vector<Eigen::Matrix<double, -1, -1>> &log_abs_beta_grad_array_for_each_n,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prob_rowwise_prod_temp,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ys,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ys,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ym,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ym,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_common_grad_term_1,
                                                           Eigen::Matrix<double, -1, -1> &log_abs_z_grad_term,
                                                           Eigen::Matrix<double, -1, -1> &sign_z_grad_term,
                                                           Eigen::Matrix<double, -1, -1> &log_abs_grad_prob,
                                                           Eigen::Matrix<double, -1, -1> &sign_grad_prob,
                                                           Eigen::Matrix<double, -1, 1>  &log_abs_prod,
                                                           Eigen::Matrix<double, -1, 1>  &sign_prod,
                                                           Eigen::Matrix<double, -1, -1> &log_abs_prod_comp,
                                                           Eigen::Matrix<double, -1, -1> &sign_prod_comp,
                                                           Eigen::Matrix<double, -1, 1>  &log_sum_result,
                                                           Eigen::Matrix<double, -1, 1>  &sign_sum_result,
                                                           Eigen::Matrix<double, -1, -1> &log_terms,
                                                           Eigen::Matrix<double, -1, -1> &sign_terms,
                                                           Eigen::Matrix<double, -1, 1>  &container_max_logs,
                                                           Eigen::Matrix<double, -1, 1>  &container_sum_exp_signed
) {
  
        const int chunk_size = log_phi_Z_recip.rows();
        const int n_tests    = log_phi_Z_recip.cols();
        (void)beta_grad_array;
      
        auto resize_all = [&](const int m) {
          log_abs_z_grad_term.resize(m, n_tests); sign_z_grad_term.resize(m, n_tests);
          log_abs_grad_prob.resize(m, n_tests);   sign_grad_prob.resize(m, n_tests);
          log_terms.resize(m, n_tests);           sign_terms.resize(m, n_tests);
          log_abs_prod.resize(m); sign_prod.resize(m); log_sum_result.resize(m); sign_sum_result.resize(m);
          container_max_logs.resize(m); container_sum_exp_signed.resize(m);
          log_abs_prod_comp.resize(m, n_tests); sign_prod_comp.resize(m, n_tests);
          log_abs_z_grad_term.setConstant(-700.0); log_abs_grad_prob.setConstant(-700.0); log_abs_prod.setConstant(-700.0);
          log_sum_result.setConstant(-700.0); log_terms.setConstant(-700.0); container_max_logs.setConstant(-700.0);
          log_abs_prod_comp.setConstant(-700.0);
          sign_z_grad_term.setOnes(); sign_grad_prob.setOnes(); sign_prod.setOnes(); sign_sum_result.setOnes(); sign_terms.setOnes();
          sign_prod_comp.setOnes(); container_sum_exp_signed.setZero();
        };
        resize_all(chunk_size);
      
        {
          const int t = n_tests - 1;
          if (n_problem_array[t] > 0) {
            const std::vector<int> &idx = problem_index_array[t];
            log_abs_beta_grad_array_for_each_n[0].col(t)(idx).array() = log_common_grad_term_1.col(t)(idx).array() + log_abs_ys.col(t)(idx).array();
            sign_beta_grad_array_for_each_n[0].col(t)(idx).array()    = sign_ys.col(t)(idx).array();
          }
        }
      
        for (int i = 0; i < n_tests - 1; i++) {
          const int t = n_tests - (i + 2);
          if (n_problem_array[t] > 0) {
            const std::vector<int> &idx = problem_index_array[t];
            resize_all(n_problem_array[t]);
      
            log_abs_grad_prob.col(0) = log_abs_ys.col(t)(idx);
            sign_grad_prob.col(0)    = sign_ys.col(t)(idx);
            log_abs_z_grad_term.col(0).array() = log_abs_ym.col(t)(idx).array();
            sign_z_grad_term.col(0).array()    = -sign_ym.col(t)(idx).array();
      
            {
              const int ii = 0;
              for (int iii = 0; iii < ii + 1; iii++) {
                const double lo = log_abs_L_Omega_double(t + ii + 1, t + iii);
                const double so = stan::math::sign(L_Omega_double(t + ii + 1, t + iii));
                log_abs_prod_comp.col(iii).array() = log_abs_z_grad_term.col(iii).array() + lo;
                sign_prod_comp.col(iii).array()    = sign_z_grad_term.col(iii).array() * so;
              }
            }
            sign_prod    = sign_prod_comp.col(0);
            log_abs_prod = log_abs_prod_comp.col(0);
      
            log_abs_grad_prob.col(1).array() = log_abs_ys.col(t + 1)(idx).array() + log_abs_prod.array();
            sign_grad_prob.col(1).array()    = sign_ys.col(t + 1)(idx).array() * sign_prod.array();
      
            if (i > 0) {
              for (int ii = 1; ii < i + 1; ii++) {
                log_abs_z_grad_term.col(ii).array() = log_abs_ym.col(t + ii)(idx).array() + log_abs_prod.array();
                sign_z_grad_term.col(ii).array()    = -sign_ym.col(t + ii)(idx).array() * sign_prod.array();
                for (int iii = 0; iii < ii + 1; iii++) {
                  const double lo = log_abs_L_Omega_double(t + ii + 1, t + iii);
                  const double so = stan::math::sign(L_Omega_double(t + ii + 1, t + iii));
                  log_abs_prod_comp.col(iii).array() = log_abs_z_grad_term.col(iii).array() + lo;
                  sign_prod_comp.col(iii).array()    = sign_z_grad_term.col(iii).array() * so;
                }
                log_abs_sum_exp_general_v2_T<vec>(log_abs_prod_comp.leftCols(ii + 1), sign_prod_comp.leftCols(ii + 1), log_abs_prod, sign_prod, container_max_logs, container_sum_exp_signed);
                log_abs_grad_prob.col(ii + 1) = log_abs_ys.col(t + ii + 1)(idx).array() + log_abs_prod.array();
                sign_grad_prob.col(ii + 1)    = sign_ys.col(t + ii + 1)(idx).array() * sign_prod.array();
              }
            }
      
            for (int ii = 0; ii < i + 2; ii++) {
              log_terms.col(ii).array()  = log_abs_grad_prob.col(ii).array() + log_prob_rowwise_prod_temp.col(t)(idx).array() - y1_log_prob.col(t + ii)(idx).array();
              sign_terms.col(ii).array() = sign_grad_prob.col(ii).array();
            }
            log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(i + 2), sign_terms.leftCols(i + 2), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
      
            log_abs_beta_grad_array_for_each_n[0].col(t)(idx).array() = log_common_grad_term_1.col(t)(idx).array() + log_sum_result.array();
            sign_beta_grad_array_for_each_n[0].col(t)(idx).array()    = sign_sum_result.array();
          }
        }
      
        resize_all(chunk_size);
  
}




//// =====================================================================================
//// 5. L_Omega grad on the log scale (problem rows only)
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_L_Omega_grad_log_scale_T(  const std::vector<int> &n_problem_array,
                                                      const std::vector<std::vector<int>> &problem_index_array,
                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>> L_Omega_grad_array,
                                                      std::vector<Eigen::Matrix<double, -1, -1>> &sign_LO,      //// sign_L_Omega_grad_array_col_for_each_n
                                                      std::vector<Eigen::Matrix<double, -1, -1>> &log_abs_LO,   //// log_abs_L_Omega_grad_array_col_for_each_n
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_Bound_Z,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_Bound_Z,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_Z_std_norm,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_Z_std_norm,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prop_rowwise_prod_temp,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ys,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ys,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ym,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ym,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_common_grad_term_1,
                                                      Eigen::Matrix<double, -1, -1> &log_abs_z_grad_term,
                                                      Eigen::Matrix<double, -1, -1> &sign_z_grad_term,
                                                      Eigen::Matrix<double, -1, -1> &log_abs_grad_prob,
                                                      Eigen::Matrix<double, -1, -1> &sign_grad_prob,
                                                      Eigen::Matrix<double, -1, 1>  &log_abs_prod,
                                                      Eigen::Matrix<double, -1, 1>  &sign_prod,
                                                      Eigen::Matrix<double, -1, -1> &log_abs_prod_comp,
                                                      Eigen::Matrix<double, -1, -1> &sign_prod_comp,
                                                      Eigen::Matrix<double, -1, -1> &log_abs_dcc,      //// log_abs_derivs_chain_container_vec_comp
                                                      Eigen::Matrix<double, -1, -1> &sign_dcc,
                                                      Eigen::Matrix<double, -1, 1>  &log_sum_result,
                                                      Eigen::Matrix<double, -1, 1>  &sign_sum_result,
                                                      Eigen::Matrix<double, -1, -1> &log_terms,
                                                      Eigen::Matrix<double, -1, -1> &sign_terms,
                                                      Eigen::Matrix<double, -1, 1>  &log_abs_a,
                                                      Eigen::Matrix<double, -1, 1>  &log_abs_b,
                                                      Eigen::Matrix<double, -1, 1>  &sign_a,
                                                      Eigen::Matrix<double, -1, 1>  &sign_b,
                                                      Eigen::Matrix<double, -1, 1>  &container_max_logs,
                                                      Eigen::Matrix<double, -1, 1>  &container_sum_exp_signed
) {
  
        const int n_tests    = log_Bound_Z.cols();
        const int chunk_size = log_Bound_Z.rows();
        (void)L_Omega_grad_array;
      
        auto resize_all = [&](const int m) {
          log_abs_z_grad_term.resize(m, n_tests); sign_z_grad_term.resize(m, n_tests);
          log_abs_grad_prob.resize(m, n_tests);   sign_grad_prob.resize(m, n_tests);
          log_terms.resize(m, n_tests);           sign_terms.resize(m, n_tests);
          log_abs_prod.resize(m); sign_prod.resize(m); log_sum_result.resize(m); sign_sum_result.resize(m);
          container_max_logs.resize(m); container_sum_exp_signed.resize(m);
          log_abs_prod_comp.resize(m, n_tests); sign_prod_comp.resize(m, n_tests);
          log_abs_dcc.resize(m, n_tests); sign_dcc.resize(m, n_tests);
          log_abs_a.resize(m); log_abs_b.resize(m); sign_a.resize(m); sign_b.resize(m);
          log_abs_z_grad_term.setConstant(-700.0); log_abs_grad_prob.setConstant(-700.0); log_abs_prod.setConstant(-700.0);
          log_sum_result.setConstant(-700.0); log_terms.setConstant(-700.0); container_max_logs.setConstant(-700.0);
          log_abs_prod_comp.setConstant(-700.0); log_abs_dcc.setConstant(-700.0); log_abs_a.setConstant(-700.0); log_abs_b.setConstant(-700.0);
          sign_z_grad_term.setOnes(); sign_grad_prob.setOnes(); sign_prod.setOnes(); sign_sum_result.setOnes(); sign_terms.setOnes();
          sign_prod_comp.setOnes(); sign_dcc.setOnes(); sign_a.setOnes(); sign_b.setOnes(); container_sum_exp_signed.setZero();
        };
      
        //// ---- last diagonal ----
        {
          const int t1 = n_tests - 1;
          const std::vector<int> &idx = problem_index_array[t1];
          sign_LO[t1].col(t1)(idx).array()    = sign_ys.col(t1)(idx).array() * sign_Bound_Z.col(t1)(idx).array();
          log_abs_LO[t1].col(t1)(idx).array() = log_common_grad_term_1.col(t1)(idx).array() + log_abs_ys.col(t1)(idx).array() + log_Bound_Z.col(t1)(idx).array();
        }
      
        //// ---- second-to-last diagonal ----
        {
          const int t1 = n_tests - 2;
          if (n_problem_array[t1] > 0) {
            const std::vector<int> &idx = problem_index_array[t1];
            resize_all(n_problem_array[t1]);
      
            log_abs_grad_prob.col(0)         = log_abs_ys.col(t1)(idx) + log_Bound_Z.col(t1)(idx);
            sign_grad_prob.col(0).array()    = sign_ys.col(t1)(idx).array() * sign_Bound_Z.col(t1)(idx).array();
            log_abs_z_grad_term.col(0)       = log_abs_ym.col(t1)(idx) + log_Bound_Z.col(t1)(idx);
            sign_z_grad_term.col(0).array()  = -sign_ym.col(t1)(idx).array() * sign_Bound_Z.col(t1)(idx).array();
            log_abs_prod.array() = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t1 + 1, t1);
            sign_prod            = sign_z_grad_term.col(0) * stan::math::sign(L_Omega_double(t1 + 1, t1));
            log_abs_grad_prob.col(1)         = log_abs_ys.col(t1 + 1)(idx) + log_abs_prod;
            sign_grad_prob.col(1).array()    = sign_ys.col(t1 + 1)(idx).array() * sign_prod.array();
      
            log_abs_a = log_abs_grad_prob.col(1) + y1_log_prob.col(t1)(idx);      sign_a = sign_grad_prob.col(1);
            log_abs_b = log_abs_grad_prob.col(0) + y1_log_prob.col(t1 + 1)(idx);  sign_b = sign_grad_prob.col(0);
            log_terms.col(0) = log_abs_a; log_terms.col(1) = log_abs_b; sign_terms.col(0) = sign_a; sign_terms.col(1) = sign_b;
            log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(2), sign_terms.leftCols(2), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
      
            log_abs_LO[t1].col(t1)(idx) = log_common_grad_term_1.col(t1)(idx) + log_sum_result;
            sign_LO[t1].col(t1)(idx)    = sign_sum_result;
          }
        }
      
        //// ---- remaining diagonals ----
        for (int i = 3; i < n_tests + 1; i++) {
          const int t1 = n_tests - i;
          if (n_problem_array[t1] > 0) {
            const std::vector<int> &idx = problem_index_array[t1];
            resize_all(n_problem_array[t1]);
      
            log_abs_grad_prob.col(0)        = log_abs_ys.col(t1)(idx) + log_Bound_Z.col(t1)(idx);
            sign_grad_prob.col(0).array()   = sign_ys.col(t1)(idx).array() * sign_Bound_Z.col(t1)(idx).array();
            log_abs_z_grad_term.col(0)      = log_abs_ym.col(t1)(idx) + log_Bound_Z.col(t1)(idx);
            sign_z_grad_term.col(0).array() = -sign_ym.col(t1)(idx).array() * sign_Bound_Z.col(t1)(idx).array();
            log_abs_prod.array() = log_abs_L_Omega_double(t1 + 1, t1) + log_abs_z_grad_term.col(0).array();
            sign_prod            = stan::math::sign(L_Omega_double(t1 + 1, t1)) * sign_z_grad_term.col(0);
            log_abs_grad_prob.col(1)        = log_abs_ys.col(t1 + 1)(idx) + log_abs_prod;
            sign_grad_prob.col(1).array()   = sign_ys.col(t1 + 1)(idx).array() * sign_prod.array();
      
            for (int ii = 1; ii < i - 1; ii++) {
              log_abs_z_grad_term.col(ii)      = log_abs_ym.col(t1 + ii)(idx) + log_abs_prod;
              sign_z_grad_term.col(ii).array() = -sign_ym.col(t1 + ii)(idx).array() * sign_prod.array();
              for (int iii = 0; iii < ii + 1; iii++) {
                log_abs_prod_comp.col(iii).array() = log_abs_z_grad_term.col(iii).array() + log_abs_L_Omega_double(t1 + ii + 1, t1 + iii);
                sign_prod_comp.col(iii)            = sign_z_grad_term.col(iii) * stan::math::sign(L_Omega_double(t1 + ii + 1, t1 + iii));
              }
              log_abs_sum_exp_general_v2_T<vec>(log_abs_prod_comp.leftCols(ii + 1), sign_prod_comp.leftCols(ii + 1), log_abs_prod, sign_prod, container_max_logs, container_sum_exp_signed);
              log_abs_grad_prob.col(ii + 1)        = log_abs_ys.col(t1 + ii + 1)(idx) + log_abs_prod;
              sign_grad_prob.col(ii + 1).array()   = sign_ys.col(t1 + ii + 1)(idx).array() * sign_prod.array();
            }
            for (int iii = 0; iii < i; iii++) {
              log_abs_dcc.col(iii) = log_abs_grad_prob.col(iii) + log_prop_rowwise_prod_temp.col(t1)(idx) - y1_log_prob.col(t1 + iii)(idx);
              sign_dcc.col(iii)    = sign_grad_prob.col(iii);
            }
            log_abs_sum_exp_general_v2_T<vec>(log_abs_dcc.leftCols(i), sign_dcc.leftCols(i), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
            log_abs_LO[t1].col(t1)(idx) = log_common_grad_term_1.col(t1)(idx) + log_sum_result;
            sign_LO[t1].col(t1)(idx)    = sign_sum_result;
          }
        }
      
        //// ---- off-diagonals: last row ----
        {
          const int t1 = n_tests - 1;
          if (n_problem_array[t1] > 0) {
            const std::vector<int> &idx = problem_index_array[t1];
            auto fill = [&](const int t2) {
              sign_LO[t1].col(t2)(idx).array() = sign_ys.col(t1)(idx).array() * sign_Z_std_norm.col(t2)(idx).array();
              log_abs_LO[t1].col(t2)(idx)      = log_common_grad_term_1.col(t1)(idx) + log_abs_ys.col(t1)(idx) + log_Z_std_norm.col(t2)(idx);
            };
            fill(n_tests - 2);
            if (t1 > 1) fill(n_tests - 3);
            if (t1 > 2) {
              for (int t2_dash = 3; t2_dash < n_tests; t2_dash++) {
                const int t2 = n_tests - (t2_dash + 1);
                if (t2 < n_tests - 1) fill(t2);
              }
            }
          }
        }
      
        //// ---- off-diagonals: remaining rows ----
        for (int t1_dash = 1; t1_dash < n_tests - 1; t1_dash++) {
          const int t1 = n_tests - (t1_dash + 1);
          if (n_problem_array[t1] > 0) {
            const std::vector<int> &idx = problem_index_array[t1];
            resize_all(n_problem_array[t1]);
            for (int t2_dash = t1_dash + 1; t2_dash < n_tests; t2_dash++) {
              const int t2 = n_tests - (t2_dash + 1);
              log_abs_prod = log_Z_std_norm.col(t2)(idx);
              sign_prod    = sign_Z_std_norm.col(t2)(idx);
              log_abs_grad_prob.col(0)        = log_abs_ys.col(t1)(idx) + log_abs_prod;
              sign_grad_prob.col(0).array()   = sign_ys.col(t1)(idx).array() * sign_prod.array();
              log_abs_z_grad_term.col(0)      = log_abs_ym.col(t1)(idx) + log_abs_prod;
              sign_z_grad_term.col(0).array() = -sign_ym.col(t1)(idx).array() * sign_prod.array();
              for (int t1_dash_dash = 1; t1_dash_dash < t1_dash + 1; t1_dash_dash++) {
                if (t1_dash_dash > 1) {
                  log_abs_z_grad_term.col(t1_dash_dash - 1)      = log_abs_ym.col(t1 + t1_dash_dash - 1)(idx) + log_abs_prod;
                  sign_z_grad_term.col(t1_dash_dash - 1).array() = -sign_ym.col(t1 + t1_dash_dash - 1)(idx).array() * sign_prod.array();
                }
                for (int iii = 0; iii < t1_dash_dash; iii++) {
                  log_abs_prod_comp.col(iii).array() = log_abs_z_grad_term.col(iii).array() + log_abs_L_Omega_double(t1 + t1_dash_dash, t1 + iii);
                  sign_prod_comp.col(iii).array()    = sign_z_grad_term.col(iii).array() * stan::math::sign(L_Omega_double(t1 + t1_dash_dash, t1 + iii));
                }
                log_abs_sum_exp_general_v2_T<vec>(log_abs_prod_comp.leftCols(t1_dash_dash), sign_prod_comp.leftCols(t1_dash_dash), log_abs_prod, sign_prod, container_max_logs, container_sum_exp_signed);
                log_abs_grad_prob.col(t1_dash_dash)        = log_abs_ys.col(t1 + t1_dash_dash)(idx) + log_abs_prod;
                sign_grad_prob.col(t1_dash_dash).array()   = sign_ys.col(t1 + t1_dash_dash)(idx).array() * sign_prod.array();
              }
              for (int ii = 0; ii < t1_dash + 1; ii++) {
                log_abs_dcc.col(ii) = log_abs_grad_prob.col(ii) + log_prop_rowwise_prod_temp.col(t1)(idx) - y1_log_prob.col(t1 + ii)(idx);
                sign_dcc.col(ii)    = sign_grad_prob.col(ii);
              }
              log_abs_sum_exp_general_v2_T<vec>(log_abs_dcc.leftCols(t1_dash + 1), sign_dcc.leftCols(t1_dash + 1), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
              log_abs_LO[t1].col(t2)(idx) = log_common_grad_term_1.col(t1)(idx) + log_sum_result;
              sign_LO[t1].col(t2)(idx)    = sign_sum_result;
            }
          }
        }
      
        resize_all(chunk_size);
  
}



