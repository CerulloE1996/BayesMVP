
//// MVP_helpers_migrated.hpp
////
//// Replaces (in MVP_manual_grad_calc_fns.hpp and wherever the nuisance fns live):
////   fn_MVP_compute_phi_Bound_Z_cols     -> fn_MVP_compute_phi_Bound_Z_cols_T<vec>
////   fn_MVP_compute_phi_Z_recip_cols     -> fn_MVP_compute_phi_Z_recip_cols_T<vec>
////   fn_MVP_compute_nuisance             -> fn_MVP_compute_nuisance_T<vec>
////   fn_MVP_compute_nuisance_log_jac_u   -> fn_MVP_compute_nuisance_log_jac_u_T<vec>
////   fn_MVP_nuisance_first_deriv         -> fn_MVP_nuisance_first_deriv_T<vec>
////   fn_MVP_nuisance_deriv_of_log_det_J  -> fn_MVP_nuisance_deriv_of_log_det_J_T<vec>
////   log_sum_exp_general                 -> log_sum_exp_general_T<vec>
////
//// UNCHANGED (they never called fn_EIGEN_double — keep your existing versions as they are):
////   fn_MVP_grad_prep, compute_rowwise_products, compute_latent_class_terms, compute_final_terms,
////   fn_MVP_compute_nuisance_grad_v2, fn_MVP_compute_coefficients_grad_v2/v3,
////   fn_MVP_compute_L_Omega_grad_v2/v3.
////   (Optional tidy: delete the dead `const std::string vect_type = ...` line at the top of each;
////    it is a string copy per call and is never used.)
////
//// The nuisance-transform functions are written from the formulas, not from your source (you
//// didn't paste it). They cover nuisance_transformation = "Phi", "Phi_approx", "inv_logit", "tanh".
//// RUN check_nuisance_migration() BELOW against your old functions before deleting the old ones.

#pragma once
//#include "fn_dispatch_templated.hpp"

//// =====================================================================================
//// 0. KernelChoice, extended with the nuisance transform. Replaces the version in
////    migration_MVOP_helpers_and_MVP_driver.hpp (same name, superset of fields).
//// =====================================================================================
enum class NuisTf { Phi, Phi_approx, inv_logit, tanh };

struct KernelChoice {
  bool   Phi_approx;       //// true: Phi_approx, false: exact Phi
  bool   inv_Phi_approx;   //// true: inv_Phi_approx, false: exact inv_Phi (AS241)
  NuisTf nuisance;         //// u = f(u_unc)
};




inline KernelChoice kernel_choice_from_args(const Model_fn_args_struct &A) {
  const std::string &Phi_type     = A.Model_args_strings(1);
  const std::string &inv_Phi_type = A.Model_args_strings(2);
  const std::string &nuis         = A.Model_args_strings(12);
  KernelChoice k;
  k.Phi_approx     = (Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2");
  k.inv_Phi_approx = (inv_Phi_type == "inv_Phi_approx");
  //// 2026-09-22 (assistant): any other Phi_type / inv_Phi_type string used to be treated silently as the exact setting; now it stops.
  if ((k.Phi_approx == false) && (Phi_type != "Phi"))
    throw std::runtime_error("kernel_choice_from_args: unknown Phi_type '" + Phi_type + "' (allowed: Phi, Phi_approx, Phi_approx_2)");
  if ((k.inv_Phi_approx == false) && (inv_Phi_type != "inv_Phi"))
    throw std::runtime_error("kernel_choice_from_args: unknown inv_Phi_type '" + inv_Phi_type + "' (allowed: inv_Phi, inv_Phi_approx)");
  if      (nuis == "Phi")        k.nuisance = NuisTf::Phi;
  else if (nuis == "Phi_approx") k.nuisance = NuisTf::Phi_approx;
  else if (nuis == "inv_logit")  k.nuisance = NuisTf::inv_logit;
  else if (nuis == "tanh")       k.nuisance = NuisTf::tanh;
  else throw std::runtime_error("kernel_choice_from_args: unknown nuisance_transformation '" + nuis + "'");
  return k;
}




//// (apply_col_inplace now lives in fn_dispatch_templated.hpp)
template <Vec vec, typename Derived>
ALWAYS_INLINE void Phi_col_inplace(Eigen::DenseBase<Derived> &M, const int col, const KernelChoice &k) {
  if (k.Phi_approx) apply_col_inplace<vec, Fn::Phi_approx>(M, col);
  else              apply_col_inplace<vec, Fn::Phi>(M, col);
}



template <Vec vec, typename Derived>
ALWAYS_INLINE void inv_Phi_col_inplace(Eigen::DenseBase<Derived> &M, const int col, const KernelChoice &k) {
  if (k.inv_Phi_approx) apply_col_inplace<vec, Fn::inv_Phi_approx>(M, col);
  else                  apply_col_inplace<vec, Fn::inv_Phi>(M, col);
}




//// =====================================================================================
//// 1. phi(Bound_Z) column. Same maths as before: writes -x^2/2 into the output column,
////    exp's it in place, scales — no temporary.
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_phi_Bound_Z_cols_T(  const int t,
                                              Eigen::Matrix<double, -1, -1> &phi_Bound_Z,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                              const KernelChoice &k
) {
  const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
  const double a_times_3 = 3.0 * 0.07056;
  const double b = 1.5976;

  if (k.Phi_approx) {
        phi_Bound_Z.col(t).array() = (a_times_3 * Bound_Z.col(t).array().square() + b)
                                   * Bound_U_Phi_Bound_Z.col(t).array()
                                   * (1.0 - Bound_U_Phi_Bound_Z.col(t).array());
  } else {
        phi_Bound_Z.col(t).array() = -0.5 * Bound_Z.col(t).array().square();
        apply_col_inplace<vec, Fn::exp>(phi_Bound_Z, t);
        phi_Bound_Z.col(t).array() *= sqrt_2_pi_recip;
  }
}




//// =====================================================================================
//// 2. 1 / phi(Z) column.
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_phi_Z_recip_cols_T(  const int t,
                                              Eigen::Matrix<double, -1, -1> &phi_Z_recip,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Phi_Z,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                              const KernelChoice &k
) {
  const double sqrt_2_pi = std::sqrt(2.0 * M_PI);
  const double a_times_3 = 3.0 * 0.07056;
  const double b = 1.5976;

  if (k.Phi_approx) {
        phi_Z_recip.col(t).array() = 1.0 / ( (a_times_3 * Z_std_norm.col(t).array().square() + b)
                                             * Phi_Z.col(t).array() * (1.0 - Phi_Z.col(t).array()) );
  } else {
        //// 1/phi(z) = sqrt(2 pi) * exp(+z^2/2)
        phi_Z_recip.col(t).array() = 0.5 * Z_std_norm.col(t).array().square();
        apply_col_inplace<vec, Fn::exp>(phi_Z_recip, t);
        phi_Z_recip.col(t).array() *= sqrt_2_pi;
  }
}




//// =====================================================================================
//// 3. Nuisance transform u = f(u_unc) and its derivatives. All in place, no temporaries.
////
////   Phi:        u = Phi(x)          du/dx = phi(x)              dlogJ/dx = -x
////   Phi_approx: u = s(g), g=x(ax^2+b), g'=3ax^2+b
////                                   du/dx = u(1-u)g'            dlogJ/dx = (1-2u)g' + 6ax/g'
////   inv_logit:  u = s(x)            du/dx = u(1-u)              dlogJ/dx = 1-2u
////   tanh:       u = (1+tanh x)/2    du/dx = 2u(1-u)             dlogJ/dx = 2(1-2u)
////
////   (log J = sum log(du/dx); for Phi that is sum(-x^2/2) - n/2 log(2 pi).)
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_nuisance_T(  Eigen::Matrix<double, -1, 1> &u_vec,
                                      const Eigen::Matrix<double, -1, 1> &u_unc_vec,
                                      const KernelChoice &k
) {
  const double a = 0.07056, b = 1.5976;
  u_vec = u_unc_vec;
  switch (k.nuisance) {
    case NuisTf::Phi:        apply_inplace<vec, Fn::Phi>(u_vec);        break;
    case NuisTf::Phi_approx: apply_inplace<vec, Fn::Phi_approx>(u_vec); break;
    case NuisTf::inv_logit:  apply_inplace<vec, Fn::inv_logit>(u_vec);  break;
    case NuisTf::tanh:
      apply_inplace<vec, Fn::tanh>(u_vec);
      u_vec.array() = 0.5 * (1.0 + u_vec.array());
      break;
  }
  (void)a; (void)b;
}




template <Vec vec>
inline double fn_MVP_compute_nuisance_log_jac_u_T(  const Eigen::Matrix<double, -1, 1> &u_vec,
                                                  const Eigen::Matrix<double, -1, 1> &u_unc_vec,
                                                  Eigen::Matrix<double, -1, 1> &scratch,     //// same length as u_vec; contents destroyed
                                                  const KernelChoice &k
) {
  const int n = static_cast<int>(u_vec.size());
  const double a_times_3 = 3.0 * 0.07056, b = 1.5976;
  const double half_log_2pi = 0.5 * std::log(2.0 * M_PI);

  switch (k.nuisance) {
    case NuisTf::Phi:
      return -0.5 * u_unc_vec.squaredNorm() - n * half_log_2pi;

    case NuisTf::Phi_approx: {
      //// sum log(u) + log(1-u) + log(3ax^2+b)
      scratch.array() = u_vec.array() * (1.0 - u_vec.array()) * (a_times_3 * u_unc_vec.array().square() + b);
      apply_inplace<vec, Fn::log>(scratch);
      return scratch.sum();
    }
    case NuisTf::inv_logit: {
      scratch.array() = u_vec.array() * (1.0 - u_vec.array());
      apply_inplace<vec, Fn::log>(scratch);
      return scratch.sum();
    }
    case NuisTf::tanh: {
      scratch.array() = 2.0 * u_vec.array() * (1.0 - u_vec.array());
      apply_inplace<vec, Fn::log>(scratch);
      return scratch.sum();
    }
  }
  return 0.0;
}




template <Vec vec>
inline void fn_MVP_nuisance_first_deriv_T(  Eigen::Matrix<double, -1, 1> &du_wrt_duu,
                                          const Eigen::Matrix<double, -1, 1> &u_vec,
                                          const Eigen::Matrix<double, -1, 1> &u_unc_vec,
                                          const KernelChoice &k
) {
  const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
  const double a_times_3 = 3.0 * 0.07056, b = 1.5976;

  switch (k.nuisance) {
    case NuisTf::Phi:
      du_wrt_duu.array() = -0.5 * u_unc_vec.array().square();
      apply_inplace<vec, Fn::exp>(du_wrt_duu);
      du_wrt_duu.array() *= sqrt_2_pi_recip;
      break;
    case NuisTf::Phi_approx:
      du_wrt_duu.array() = u_vec.array() * (1.0 - u_vec.array()) * (a_times_3 * u_unc_vec.array().square() + b);
      break;
    case NuisTf::inv_logit:
      du_wrt_duu.array() = u_vec.array() * (1.0 - u_vec.array());
      break;
    case NuisTf::tanh:
      du_wrt_duu.array() = 2.0 * u_vec.array() * (1.0 - u_vec.array());
      break;
  }
}




template <Vec vec>
inline void fn_MVP_nuisance_deriv_of_log_det_J_T(  Eigen::Matrix<double, -1, 1> &d_J_wrt_duu,
                                                 const Eigen::Matrix<double, -1, 1> &u_vec,
                                                 const Eigen::Matrix<double, -1, 1> &u_unc_vec,
                                                 const Eigen::Matrix<double, -1, 1> &du_wrt_duu,   //// unused for these transforms; kept for signature parity
                                                 const KernelChoice &k
) {
  const double a = 0.07056, a_times_3 = 3.0 * a, a_times_6 = 6.0 * a, b = 1.5976;
  (void)du_wrt_duu;

  switch (k.nuisance) {
    case NuisTf::Phi:
      d_J_wrt_duu.array() = -u_unc_vec.array();
      break;
    case NuisTf::Phi_approx: {
      //// (1-2u) g' + g''/g',  g' = 3ax^2+b, g'' = 6ax
      const auto gp = (a_times_3 * u_unc_vec.array().square() + b);
      d_J_wrt_duu.array() = (1.0 - 2.0 * u_vec.array()) * gp + (a_times_6 * u_unc_vec.array()) / gp;
      break;
    }
    case NuisTf::inv_logit:
      d_J_wrt_duu.array() = 1.0 - 2.0 * u_vec.array();
      break;
    case NuisTf::tanh:
      d_J_wrt_duu.array() = 2.0 * (1.0 - 2.0 * u_vec.array());
      break;
  }
}




//// =====================================================================================
//// 4. Row-wise log-sum-exp over the columns of lp_array (chunk_size x n_class).
////    NOTE: modifies lp_array in place (it is per-chunk workspace, rewritten next chunk).
//// =====================================================================================
template <Vec vec>
inline void log_sum_exp_general_T(  Eigen::Matrix<double, -1, -1> &lp_array,
                                  Eigen::Matrix<double, -1, 1> &log_sum_result,
                                  Eigen::Matrix<double, -1, 1> &container_max_logs
) {
  const int n_cols = static_cast<int>(lp_array.cols());
  container_max_logs = lp_array.rowwise().maxCoeff();
  for (int c = 0; c < n_cols; ++c) {
    lp_array.col(c).array() -= container_max_logs.array();
    apply_col_inplace<vec, Fn::exp>(lp_array, c);
  }
  log_sum_result = lp_array.rowwise().sum();
  apply_inplace<vec, Fn::log>(log_sum_result);
  log_sum_result.array() += container_max_logs.array();
}




