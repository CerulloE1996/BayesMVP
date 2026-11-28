

//// MVOP_lp_grad_common_T.hpp
////
//// Everything the three MVOP drivers (NoLog serial, NoLog WCP, PartialLog serial) do BEFORE and
//// AFTER the chunk loop, factored into one place. The three copies in the files were identical
//// except for two discrepancies in the WCP copy (the double-side std_normal_lpdf(C) and the
//// var-side Phi(C) were missing "- anchor"); the serial version (anchored in both, value and
//// gradient consistent) is used here.
////
//// Requires: MVP_helpers_migrated.hpp (KernelChoice), the Pinkney_corr_master(_dbl), vec_of_mats*,
//// fn_MVP_prev_multi_pop_AD, chain_rule_C_to_C_raw, calculate_chunk_sizes.

#pragma once
// #include "MVP_helpers_migrated.hpp"
#include "MVOP_Se_Sp_prior_fns.hpp"




struct MVOP_prep {
  //// sizes
  int N;
  int n_tests;
  int n_us;
  int n_params_main;
  int n_params;
  int n_class;
  int n_pops;
  int n_corrs;
  int dim_choose_2;
  int n_covariates_total;
  int n_covariates_max;
  int n_binary_tests;
  int n_ordinal_tests;
  int n_cutpoints_total;
  int n_cutpoints_max;
  bool exclude_priors;
  double overflow_threshold;
  double underflow_threshold;
  //// chunking
  int chunk_size_orig;
  int normal_chunk_size;
  int last_chunk_size;
  int n_total_chunks;
  int n_full_chunks;
  //// kernel choice
  KernelChoice kchoice;
  //// pointers into Model_args (valid for the lifetime of the driver call)
  const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> *X;
  const Eigen::Matrix<int, -1, -1> *n_covariates_per_outcome_vec;
  const Eigen::Matrix<int, -1, 1> *n_cat_per_ord_test;
  const Eigen::Matrix<int, -1, 1> *n_thr_per_ord_test;
  const Eigen::Matrix<int, -1, 1> *pop_ind;
  const std::vector<Eigen::Matrix<double, -1, -1>> *prior_coeffs_mean;
  const std::vector<Eigen::Matrix<double, -1, -1>> *prior_coeffs_sd;
  //// derived
  Eigen::Matrix<int, -1, 1> ord_idx_of_test;
  //// for each ordinal slot t_ord (0-based, in fitted order) the index of the fitted test (column of y)
  //// that holds it, i.e. the inverse of ord_idx_of_test; every test-indexed read for an ordinal slot
  //// goes through it, so that no read assumes that the binary tests come first in y:
  Eigen::Matrix<int, -1, 1> fitted_test_index_of_ordinal_slot;
  std::vector<Eigen::Matrix<double, -1, -1>> beta_double_array;
  std::vector<Eigen::Matrix<double, -1, -1>> C_raw;
  std::vector<Eigen::Matrix<double, -1, -1>> C;
  std::vector<Eigen::Matrix<double, -1, -1>> dC_raw_dunc;
  std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double;
  std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double;
  std::vector<Eigen::Matrix<double, -1, -1>> deriv_L_wrt_unc_full;
  Eigen::Matrix<double, -1, -1> prev_mat;
  Eigen::Matrix<double, -1, -1> log_prev_mat_small;
  Eigen::Matrix<double, -1, 1> tanh_u_prev_vec;
  Eigen::Matrix<double, -1, 1> deriv_p_wrt_u_vec;
  double prior_densities;
  double log_det_J_main;
};




//// -------------------------------------------------------------------------------------
//// Everything before the chunk loop. Writes the AD-derived gradient pieces (cutpoint priors/
//// Jacobians, Omega priors/Jacobians, prev) straight into out_mat, as the originals did.
//// -------------------------------------------------------------------------------------
inline MVOP_prep fn_MVOP_prep(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                const Model_fn_args_struct &A,
                                const int n_chunks_multiplier
) {

        MVOP_prep P;
        out_mat.setZero();
        P.N = y_ref.rows(); P.n_tests = y_ref.cols(); P.n_us = theta_us_vec_ref.rows();
        P.n_params_main = theta_main_vec_ref.rows(); P.n_params = P.n_params_main + P.n_us;
        const int N = P.N;
        const int n_tests = P.n_tests;
        const int n_us = P.n_us;

        P.X = &A.Model_args_2_layer_vecs_of_mats_double[0];
        P.exclude_priors = A.Model_args_bools(0);
        const bool corr_force_positive = A.Model_args_bools(2);
        const bool corr_prior_beta = A.Model_args_bools(3);
        const bool corr_prior_norm = A.Model_args_bools(4);
        fn_stop_if_corr_prior_beta_and_corr_prior_norm_are_both_true( corr_prior_beta,
                                                                      corr_prior_norm,
                                                                      "MVOP_lp_grad_common_T.hpp (fn_MVOP_prep)");
        P.n_class = A.Model_args_ints(1); const int n_class = P.n_class;
        const int n_chunks = A.Model_args_ints(3);
        P.overflow_threshold = A.Model_args_doubles(0); P.underflow_threshold = A.Model_args_doubles(1);
        const double C_raw_lower = A.Model_args_doubles(2);
        const double C_raw_upper = A.Model_args_doubles(3);
        const std::string &vect_type = A.Model_args_strings(0);
        const std::string &J_grad_option = A.Model_args_strings(11);
        P.kchoice = kernel_choice_from_args(A);

        const Eigen::Matrix<double, -1, 1> &lkj_cholesky_eta = A.Model_args_col_vecs_double[0];
        const Eigen::Matrix<double, -1, 1> &prev_prior_a = A.Model_args_col_vecs_double[1];
        const Eigen::Matrix<double, -1, 1> &prev_prior_b = A.Model_args_col_vecs_double[2];
        P.n_covariates_per_outcome_vec = &A.Model_args_mats_int[0];
        const Eigen::Matrix<int, -1, -1> &ncov = *P.n_covariates_per_outcome_vec;
        P.prior_coeffs_mean = &A.Model_args_vecs_of_mats_double[0];
        P.prior_coeffs_sd   = &A.Model_args_vecs_of_mats_double[1];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_mean = *P.prior_coeffs_mean;
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_sd = *P.prior_coeffs_sd;
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_a = A.Model_args_vecs_of_mats_double[2];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_b = A.Model_args_vecs_of_mats_double[3];
        std::vector<Eigen::Matrix<double, -1, -1>>        lb_corr          = A.Model_args_vecs_of_mats_double[4];
        const std::vector<Eigen::Matrix<double, -1, -1>> &ub_corr          = A.Model_args_vecs_of_mats_double[5];
        const std::vector<Eigen::Matrix<double, -1, -1>> &known_values     = A.Model_args_vecs_of_mats_double[6];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_dirichlet_alpha = A.Model_args_vecs_of_mats_double[7];
        const std::vector<Eigen::Matrix<int, -1, -1>>    &known_values_indicator = A.Model_args_vecs_of_mats_int[0];
        ////
        //// ---- priors directly on Sp (c = 0) and Se (c = 1) at chosen cutoffs (off unless use_Se_Sp_prior = 1):
        ////
        const int use_Se_Sp_prior = (A.Model_args_ints.rows() > 7) ? A.Model_args_ints(7) : 0;
        const int Se_Sp_prior_type = (use_Se_Sp_prior == 1) ? A.Model_args_ints(8) : 0;
        Se_Sp_prior_args_struct Se_Sp_args;
        if (use_Se_Sp_prior == 1) {
             Se_Sp_args = fn_Se_Sp_prior_args(A.Model_args_vecs_of_mats_int[1],
                                              A.Model_args_vecs_of_mats_double[8],
                                              A.Model_args_vecs_of_mats_double[9],
                                              A.Model_args_vecs_of_mats_double[10],
                                              A.Model_args_vecs_of_mats_double[11],
                                              A.Model_args_vecs_of_mats_double[12],
                                              prior_dirichlet_alpha,
                                              A.Model_args_col_vecs_int[1],
                                              Se_Sp_prior_type);
        }
        if (corr_force_positive) for (int c = 0; c < n_class; c++) for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++) lb_corr[c](i, j) = 0.0;

        //// ---- slot types ----
        const Eigen::Matrix<int, -1, 1> &n_cat_per_test = A.Model_args_col_vecs_int[3];
        P.ord_idx_of_test.resize(n_tests); P.n_binary_tests = 0; P.n_ordinal_tests = 0;
        for (int t = 0; t < n_tests; t++) {
          if (n_cat_per_test(t) > 2) { P.ord_idx_of_test(t) = P.n_ordinal_tests; ++P.n_ordinal_tests; }
          else                       { P.ord_idx_of_test(t) = -1;                ++P.n_binary_tests;  }
        }
        // const int n_ordinal_tests = P.n_ordinal_tests, n_binary_tests = P.n_binary_tests;
        const int n_ordinal_tests = P.n_ordinal_tests;
        //// ---- ordinal slot -> fitted test (the inverse of ord_idx_of_test). The cutpoint-prior anchor
        ////      below read the intercept prior mean of test n_binary_tests + t_ord, which is the test of
        ////      ordinal slot t_ord only when the binary tests come first in y:
        P.fitted_test_index_of_ordinal_slot.resize(n_ordinal_tests);
        for (int t = 0; t < n_tests; t++) {
          if (P.ord_idx_of_test(t) >= 0) P.fitted_test_index_of_ordinal_slot(P.ord_idx_of_test(t)) = t;
        }
        P.n_cat_per_ord_test = &A.Model_args_col_vecs_int[0];
        P.n_thr_per_ord_test = &A.Model_args_col_vecs_int[1];
        const Eigen::Matrix<int, -1, 1> &n_thr_per_ord_test = *P.n_thr_per_ord_test;
        P.n_cutpoints_total = n_class * n_thr_per_ord_test.sum();
        P.n_cutpoints_max = (n_ordinal_tests > 0) ? n_thr_per_ord_test.maxCoeff() : 0;
        const int n_cutpoints_total = P.n_cutpoints_total;
        const int n_cutpoints_max = P.n_cutpoints_max;

        P.n_pops = A.Model_args_ints(6); const int n_pops = P.n_pops;
        P.pop_ind = &A.Model_args_col_vecs_int[2];
        P.n_corrs = n_class * n_tests * (n_tests - 1) / 2; const int n_corrs = P.n_corrs;
        if (n_class > 1) { P.n_covariates_total = ncov.row(0).sum() + ncov.row(1).sum(); P.n_covariates_max = std::max(ncov.row(0).maxCoeff(), ncov.row(1).maxCoeff()); }
        else             { P.n_covariates_total = ncov.sum(); P.n_covariates_max = ncov.array().maxCoeff(); }
        const int n_covariates_total = P.n_covariates_total;
        const int n_covariates_max = P.n_covariates_max;

        //// ---- chunking ----
        int vec_size; if (vect_type == "AVX512") vec_size = 8; else if (vect_type == "AVX2") vec_size = 4; else if (vect_type == "AVX") vec_size = 2; else vec_size = 1;
        //// n_chunks_multiplier is 1 except in the PartialLog evaluations (the fallback, attempt 2 in
        //// MVP_lp_grad_multi_attempts.hpp, and the forced ones), which use the sample() argument
        //// n_chunks_multiplier_for_PartialLog_log_scale_evaluation (default 3) so that each
        //// (larger-per-individual) PartialLog chunk stays about the same size in memory as a NoLog chunk;
        //// the nuisance vector and its gradient are re-laid out between the stored chunk layout and this
        //// one (PartialLog_nuisance_chunk_relayout.hpp).
        //// Requested chunks are not capped; calculate_chunk_sizes preserves the existing SIMD packet width.
        const int n_chunks_used = n_chunks * std::max(1, n_chunks_multiplier);
        ChunkSizeInfo csi = calculate_chunk_sizes(N, vec_size, n_chunks_used);
        P.chunk_size_orig = csi.chunk_size_orig; P.normal_chunk_size = csi.normal_chunk_size; P.last_chunk_size = csi.last_chunk_size;
        P.n_total_chunks = csi.n_total_chunks; P.n_full_chunks = csi.n_full_chunks;

        //// ---- corrs / coeffs / prev (doubles) ----
        const Eigen::Matrix<double, -1, 1> Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs);
        P.beta_double_array = vec_of_mats(n_covariates_max, n_tests, n_class);
        { int i = n_corrs; for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < ncov(c, t); k++) P.beta_double_array[c](k, t) = theta_main_vec_ref(i++); }
        Eigen::Matrix<double, -1, 1> u_prev_raw(n_pops);
        if (n_class > 1) for (int g = 0; g < n_pops; g++) u_prev_raw(g) = theta_main_vec_ref(n_corrs + n_covariates_total + g);

        //// ---- cutpoints (doubles): theta -> C_raw (bounded via tanh) -> C ----
        const double C_raw_range = C_raw_upper - C_raw_lower;
        const double log_half_range = std::log(0.5 * C_raw_range);
        P.C_raw = vec_of_mats(n_cutpoints_max, n_ordinal_tests, n_class); P.C = P.C_raw; P.dC_raw_dunc = P.C_raw;
        double log_det_J_unc_to_C_raw_double = 0.0;
        { int i = n_corrs + n_covariates_total + (n_class > 1 ? n_pops : 0);
          for (int c = 0; c < n_class; c++) for (int t_ord = 0; t_ord < n_ordinal_tests; t_ord++) for (int k = 0; k < n_thr_per_ord_test(t_ord); k++) {
            const double x = theta_main_vec_ref(i++);
            const double th = std::tanh(x);
            P.C_raw[c](k, t_ord) = C_raw_lower + C_raw_range * 0.5 * (1.0 + th);
            P.dC_raw_dunc[c](k, t_ord) = C_raw_range * 0.5 * (1.0 - th * th);
            const double ax = std::abs(x);
            log_det_J_unc_to_C_raw_double += log_half_range + 2.0 * (0.6931471805599453 - ax - std::log1p(std::exp(-2.0 * ax)));
          } }
        double prior_density_induced_Dirichlet_double = 0.0;
        double log_det_J_C_raw_to_C_double = 0.0;
        for (int c = 0; c < n_class; c++) for (int t_ord = 0; t_ord < n_ordinal_tests; t_ord++) {
          const int n_thr_t = n_thr_per_ord_test(t_ord);
          const int n_cat_t = n_thr_t + 1;
          P.C[c](0, t_ord) = P.C_raw[c](0, t_ord);
          for (int k = 1; k < n_thr_t; k++) { P.C[c](k, t_ord) = P.C[c](k - 1, t_ord) + std::exp(P.C_raw[c](k, t_ord)); log_det_J_C_raw_to_C_double += P.C_raw[c](k, t_ord); }
          // const int t = n_binary_tests + t_ord; const double anchor = prior_coeffs_mean[c](0, t);
          //// anchor = the intercept prior mean of the fitted test holding ordinal slot t_ord (any order):
          const int t = P.fitted_test_index_of_ordinal_slot(t_ord);
          const double anchor = prior_coeffs_mean[c](0, t);
          if (use_Se_Sp_prior == 1 && Se_Sp_args.n_chosen[c](t_ord) > 0) continue;
          Eigen::Matrix<double, -1, 1> cp(n_thr_t); for (int k = 0; k < n_thr_t; k++) cp(k) = stan::math::Phi(P.C[c](k, t_ord) - anchor);
          Eigen::Matrix<double, -1, 1> p_ord(n_cat_t); p_ord(0) = cp(0); for (int k = 1; k < n_thr_t; k++) p_ord(k) = cp(k) - cp(k - 1); p_ord(n_cat_t - 1) = 1.0 - cp(n_thr_t - 1);
          const Eigen::Matrix<double, -1, 1> alpha_t = prior_dirichlet_alpha[c].col(t_ord).head(n_cat_t);
          prior_density_induced_Dirichlet_double += stan::math::lgamma(alpha_t.sum());
          for (int k = 0; k < n_cat_t; k++) { prior_density_induced_Dirichlet_double -= stan::math::lgamma(alpha_t(k));
            const double coeff = alpha_t(k) - 1.0; if (std::abs(coeff) > 1e-15) prior_density_induced_Dirichlet_double += coeff * std::log(p_ord(k)); }
          for (int k = 0; k < n_thr_t; k++) prior_density_induced_Dirichlet_double += stan::math::std_normal_lpdf(P.C[c](k, t_ord) - anchor);
        }

        //// ---- Omega (doubles) ----
        double prior_densities_L_Omega_double = 0.0;
        double log_det_J_L_Omega_double = 0.0;
        P.dim_choose_2 = n_tests * (n_tests - 1) / 2; const int dim_choose_2 = P.dim_choose_2;
        P.deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
        P.L_Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
        P.L_Omega_recip_double = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_unconstrained_double = fn_convert_std_vec_of_corrs_to_3d_array_double( Eigen_vec_to_std_vec(Omega_raw_vec_double),
                                                                                                                                n_tests,
                                                                                                                                n_class);
        for (int c = 0; c < n_class; c++) {
          P.L_Omega_double[c] = Pinkney_corr_master_dbl(n_tests,
                           lb_corr[c],
                           ub_corr[c],
                           Omega_unconstrained_double[c],
                           known_values_indicator[c],
                           known_values[c]).block(1,
                           0,
                           n_tests,
                           n_tests);
          for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++) P.L_Omega_recip_double[c](i,
               j) = (std::abs(P.L_Omega_double[c](i, j)) > 1e-10) ? 1.0 / P.L_Omega_double[c](i,
               j) : 0.0;
        }
        if (J_grad_option == "num_diff") {
          for (int c = 0; c < n_class; c++) {
            auto get_L = [&](const Eigen::Matrix<double, -1, -1> &Ou) { return Pinkney_corr_master_dbl(n_tests, lb_corr[c], ub_corr[c], Ou, known_values_indicator[c], known_values[c]).block(1, 0, n_tests, n_tests).eval(); };
            int cnt_2 = 0;
            for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++) {
              double eps = std::max(1e-8, 1e-6 * std::abs(Omega_unconstrained_double[c](i, j))); if (std::abs(Omega_unconstrained_double[c](i, j)) > 2.0) eps = 1e-4;
              Omega_unconstrained_double[c](i, j) += eps; Eigen::Matrix<double, -1, -1> Lp = get_L(Omega_unconstrained_double[c]); Omega_unconstrained_double[c](i, j) -= eps;
              int cnt_1 = 0;
           // for (int k = 0; k < n_tests; k++) for (int l = 0; l <= k; l++) { const double d = (Lp(k, l) - P.L_Omega_double[c](k, l)) / eps; P.deriv_L_wrt_unc_full[c](cnt_1++, cnt_2) = std::isfinite(d) ? d : 0.0; }
              //// a non-finite derivative is stored as NaN, not as 0 (flagged, so the fallback runs):
              for (int k = 0; k < n_tests; k++) for (int l = 0; l <= k; l++) {
                double deriv = (Lp(k, l) - P.L_Omega_double[c](k, l)) / eps;
                if (!std::isfinite(deriv)) deriv = std::numeric_limits<double>::quiet_NaN();
                P.deriv_L_wrt_unc_full[c](cnt_1++, cnt_2) = deriv;
              }
              cnt_2++;
            }
          }
        }

        //// ---- AD block: cutpoint priors/Jacobians, Omega priors/Jacobians, prev ----
        double prior_densities_prev_double = 0.0;
        double log_det_J_prev_from_AD = 0.0;
        Eigen::Matrix<double, -1, 1> grad_prev_raw(n_pops);
        {
          stan::math::start_nested();
          stan::math::var target_AD = 0.0;
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> C_raw_var = vec_of_mats_var(n_cutpoints_max, n_ordinal_tests, n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> C_var = C_raw_var;
          Eigen::Matrix<stan::math::var, -1, 1> C_unc_vec_var(n_cutpoints_total);
          { int i = n_corrs + n_covariates_total + (n_class > 1 ? n_pops : 0), j = 0;
            for (int c = 0; c < n_class; c++) for (int t_ord = 0; t_ord < n_ordinal_tests; t_ord++) for (int k = 0; k < n_thr_per_ord_test(t_ord); k++) {
              C_unc_vec_var(j) = stan::math::to_var(theta_main_vec_ref(i));
              stan::math::var th = stan::math::tanh(C_unc_vec_var(j));
              C_raw_var[c](k, t_ord) = C_raw_lower + C_raw_range * 0.5 * (1.0 + th);
              target_AD += std::log(0.5 * C_raw_range) + stan::math::log1m(stan::math::square(th));
              i++; j++;
            } }
          {
            stan::math::var pdD = 0.0;
            stan::math::var ldJ = 0.0;
            for (int c = 0; c < n_class; c++) for (int t_ord = 0; t_ord < n_ordinal_tests; t_ord++) {
              const int n_thr_t = n_thr_per_ord_test(t_ord);
              const int n_cat_t = n_thr_t + 1;
              C_var[c](0, t_ord) = C_raw_var[c](0, t_ord);
              for (int k = 1; k < n_thr_t; k++) { C_var[c](k, t_ord) = C_var[c](k - 1, t_ord) + stan::math::exp(C_raw_var[c](k, t_ord)); ldJ += C_raw_var[c](k, t_ord); }
              // const int t = n_binary_tests + t_ord;
              // stan::math::var anchor = stan::math::to_var(prior_coeffs_mean[c](0, t));
              //// the same anchor as the double-side prior above (the test of ordinal slot t_ord):
              const int t = P.fitted_test_index_of_ordinal_slot(t_ord);
              stan::math::var anchor = stan::math::to_var(prior_coeffs_mean[c](0, t));
              if (use_Se_Sp_prior == 1 && Se_Sp_args.n_chosen[c](t_ord) > 0) {
                   Eigen::Matrix<stan::math::var, -1, 1> C_centred(n_thr_t);
                   for (int k = 0; k < n_thr_t; k++) {
                        C_centred(k) = C_var[c](k, t_ord) - anchor;
                   }
                   stan::math::var lp_Se_Sp = fn_induced_dirichlet_with_Se_Sp_prior_lpdf(
                                                    C_centred,
                                                    C_centred,
                                                    prior_dirichlet_alpha[c].col(t_ord).head(n_cat_t),
                                                    Se_Sp_args.alpha_seg[c][t_ord],
                                                    Se_Sp_args.chosen[c][t_ord],
                                                    Se_Sp_prior_type,
                                                    Se_Sp_args.A[c][t_ord],
                                                    Se_Sp_args.beta_ab[c][t_ord],
                                                    Se_Sp_args.link_mean_SD[c][t_ord]);
                   pdD += lp_Se_Sp;
                   prior_density_induced_Dirichlet_double += lp_Se_Sp.val();
                   continue;
              }
              Eigen::Matrix<stan::math::var, -1, 1> cp(n_thr_t); for (int k = 0; k < n_thr_t; k++) cp(k) = stan::math::Phi(C_var[c](k, t_ord) - anchor);
              Eigen::Matrix<stan::math::var, -1, 1> p_ord(n_cat_t); p_ord(0) = cp(0); for (int k = 1; k < n_thr_t; k++) p_ord(k) = cp(k) - cp(k - 1); p_ord(n_cat_t - 1) = 1.0 - cp(n_thr_t - 1);
              const Eigen::Matrix<double, -1, 1> alpha_t = prior_dirichlet_alpha[c].col(t_ord).head(n_cat_t);
              pdD += stan::math::lgamma(alpha_t.sum());
              if (!prior_dirichlet_alpha[c].col(t_ord).head(n_cat_t).isOnes()) {
                for (int k = 0; k < n_cat_t; k++) { pdD -= stan::math::lgamma(alpha_t(k)); const double coeff = alpha_t(k) - 1.0; if (std::abs(coeff) > 1e-15) pdD += coeff * stan::math::log(p_ord(k)); }
              }
              for (int k = 0; k < n_thr_t; k++) pdD += stan::math::std_normal_lpdf(C_var[c](k, t_ord) - anchor);
            }
            //// target_AD += ldJ + pdD;
            //// ---- the induced Dirichlet cutpoint prior pdD is a prior: left out of the gradient with
            ////      exclude_priors, as it is left out of the log density; the C_raw -> C Jacobian ldJ stays:
            if (!P.exclude_priors) target_AD += ldJ + pdD;
            else                   target_AD += ldJ;
          }
          target_AD.grad();
          out_mat.segment(1 + n_us + n_corrs + n_covariates_total + (n_class > 1 ? n_pops : 0),
                          n_cutpoints_total) = C_unc_vec_var.adj();
          stan::math::set_zero_all_adjoints_nested();

          Eigen::Matrix<stan::math::var, -1, 1> Omega_raw_vec_var = stan::math::to_var(Omega_raw_vec_double);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var( Eigen_vec_to_std_vec_var(Omega_raw_vec_var),
                                                                                                                                     n_tests,
                                                                                                                                     n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Omega_var = L_Omega_var;
          {
            stan::math::var ldJ = 0.0;
            for (int c = 0; c < n_class; c++) {
              Eigen::Matrix<stan::math::var, -1, -1> CS = Pinkney_corr_master(n_tests,
                                                                              lb_corr[c],
                                                                              ub_corr[c],
                                                                              Omega_unconstrained_var[c],
                                                                              known_values_indicator[c],
                                                                              known_values[c]);
              L_Omega_var[c] = CS.block(1, 0, n_tests, n_tests); target_AD += CS(0, 0); ldJ += CS(0, 0);
              Omega_var[c] = L_Omega_var[c] * L_Omega_var[c].transpose();
              for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++) if (known_values_indicator[c](i, j) == 1) {
                //// stan::math::var kv = stan::math::normal_lpdf(Omega_var[c](i, j), 0.0, 10.0); target_AD += kv;
                ////       prior_densities_L_Omega_double += kv.val(); }
                stan::math::var kv = stan::math::normal_lpdf(Omega_var[c](i, j), 0.0, 10.0);
                //// ---- a prior: left out of the gradient with exclude_priors, as it is left out of the log
                ////      density:
                if (!P.exclude_priors) target_AD += kv;
                prior_densities_L_Omega_double += kv.val(); }
            }
            log_det_J_L_Omega_double += ldJ.val();
          }
          {
            stan::math::var pd = 0.0;
            for (int c = 0; c < n_class; c++) {
              //// if (!corr_prior_beta && !corr_prior_norm) pd +=
              ////       stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c));
              if (!corr_prior_beta && !corr_prior_norm) {
                pd += stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c));
                //// ---- with known (fixed) correlations, the terms of the known pairs in the
                ////      Jacobian that lkj_corr_cholesky_lpdf contains are removed (the prior is
                ////      on the free correlations):
                fn_subtract_jacobian_terms_of_known_correlations_from_lkj_corr_cholesky_density( pd,
                                                                                                 L_Omega_var[c],
                                                                                                 known_values_indicator[c]);
              }
              else {
                for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++)
                  pd += corr_prior_beta ? stan::math::beta_lpdf((Omega_var[c](i, j) + 1) / 2,
                                                                prior_for_corr_a[c](i, j),
                                                                prior_for_corr_b[c](i, j))
                                        : stan::math::normal_lpdf(Omega_var[c](i, j),
                                                                  prior_for_corr_a[c](i, j),
                                                                  prior_for_corr_b[c](i, j));
                //// ---- log Jacobian of the map from the free entries of L_Omega to the off-diagonal
                ////      entries of Omega: sum over i of (n_tests - 1 - i) * log(L_Omega(i, i)) for
                ////      0-indexed i, i.e. prod_r L_rr^(K - r) for 1-indexed r. The exponent
                ////      n_tests + 1 - (i + 1) of the commented-out lines below has one power of L_Omega(i, i)
                ////      too many, i.e. an extra det(Omega)^(1/2) factor that pulls the correlations towards 0.
                //// Eigen::Matrix<stan::math::var, -1, 1> jd(n_tests); for (int i = 0; i < n_tests; ++i) jd(i) =
                ////       (n_tests + 1 - (i + 1)) * stan::math::log(L_Omega_var[c](i, i));
                Eigen::Matrix<stan::math::var, -1, 1> jd(n_tests);
                //// ---- with known (fixed) correlations the map is to the FREE off-diagonal
                ////      correlations only: the power of L_Omega(i, i) is the number of free
                ////      correlations below the diagonal in column i (n_tests - 1 - i when no
                ////      correlation is known):
                //// for (int i = 0; i < n_tests; ++i) {
                ////       jd(i) = (n_tests - 1 - i) *
                ////             stan::math::log(L_Omega_var[c](i, i));
                //// }
                for (int i = 0; i < n_tests; i++) {
                      const int number_of_free_correlations_below_diagonal_in_column_i =
                            fn_number_of_free_correlations_below_diagonal_in_column( known_values_indicator[c],
                                                                                     i);
                      jd(i) = number_of_free_correlations_below_diagonal_in_column_i *
                             stan::math::log(L_Omega_var[c](i, i));
                }
                //// ---- constant: -log(2) per correlation for the Beta prior on (rho + 1) / 2, i.e.
                ////      n_tests * (n_tests - 1) / 2 per class; no log(2) term for the normal prior on
                ////      rho (the commented-out line below adds + n_tests * log(2) for both; a constant, so the
                ////      draws are the same):
                //// pd += (n_tests * stan::math::log(2) + jd.sum());
                pd += corr_prior_beta ? (- 0.5 * n_tests * (n_tests - 1) * stan::math::log(2.0) + jd.sum())
                                      : jd.sum();
              }
            }
            //// target_AD += pd; prior_densities_L_Omega_double += pd.val();
            //// ---- the correlation prior: left out of the gradient with exclude_priors, as it is left out
            ////      of the log density (prior_densities):
            if (!P.exclude_priors) target_AD += pd;
            prior_densities_L_Omega_double += pd.val();
            target_AD.grad();
            out_mat.segment(1 + n_us, n_corrs) = Omega_raw_vec_var.adj();
            stan::math::set_zero_all_adjoints_nested();
          }
          if (n_class > 1) {
            fn_MVP_prev_multi_pop_AD(u_prev_raw,
                                     prev_prior_a,
                                     prev_prior_b,
                                     n_pops,
                                     prior_densities_prev_double,
                                     log_det_J_prev_from_AD,
                                     grad_prev_raw);
            //// ---- the prevalence prior is a prior: with exclude_priors its gradient is left out, as its
            ////      density is left out of the log density (the Jacobian gradient is added below):
            if (P.exclude_priors) grad_prev_raw.setZero();
            out_mat.segment(1 + n_us + n_corrs + n_covariates_total, n_pops) = grad_prev_raw;
          }
          if (J_grad_option == "autodiff") {
            for (int c = 0; c < n_class; c++) { int cnt_1 = 0;
              for (int k = 0; k < n_tests; k++) for (int l = 0; l < k + 1; l++) {
                (L_Omega_var[c](k, l)).grad(); int cnt_2 = 0;
                for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++) P.deriv_L_wrt_unc_full[c](cnt_1,
                     cnt_2++) = Omega_unconstrained_var[c](i,
                     j).adj();
                stan::math::set_zero_all_adjoints_nested(); cnt_1++;
              } }
          }
          for (int c = 0; c < n_class; c++) for (int t2 = 0; t2 < n_tests; t2++) for (int t1 = 0; t1 < n_tests; t1++) {
            P.L_Omega_double[c](t1,
                             t2) = L_Omega_var[c](t1,
                             t2).val(); P.L_Omega_recip_double[c](t1,
                             t2) = 1.0 / P.L_Omega_double[c](t1,
                             t2); }
          stan::math::recover_memory_nested();
        }

        //// ---- prev (doubles) ----
        P.prev_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class); P.log_prev_mat_small = P.prev_mat;
        P.tanh_u_prev_vec.resize(n_pops); P.deriv_p_wrt_u_vec.resize(n_pops);
        double log_det_J_prev_double_total = 0.0;
        if (n_class > 1) {
          for (int g = 0; g < n_pops; g++) {
            P.tanh_u_prev_vec(g) = stan::math::tanh(u_prev_raw(g)); const double pg = 0.5 * (P.tanh_u_prev_vec(g) + 1.0);
            P.prev_mat(g, 1) = pg; P.prev_mat(g, 0) = 1.0 - pg;
            P.deriv_p_wrt_u_vec(g) = 0.5 * (1.0 - P.tanh_u_prev_vec(g) * P.tanh_u_prev_vec(g));
            log_det_J_prev_double_total += std::log(P.deriv_p_wrt_u_vec(g));
          }
          P.log_prev_mat_small = stan::math::log(P.prev_mat);
        }

        //// ---- priors / Jacobians ----
        P.prior_densities = 0.0;
        if (!P.exclude_priors) {
          for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < ncov(c, t); k++)
            P.prior_densities += stan::math::normal_lpdf(P.beta_double_array[c](k, t),
                                                         prior_coeffs_mean[c](k, t),
                                                         prior_coeffs_sd[c](k, t));
          P.prior_densities += prior_densities_L_Omega_double + prior_densities_prev_double + prior_density_induced_Dirichlet_double;
        }
        P.log_det_J_main = log_det_J_prev_double_total + log_det_J_L_Omega_double + log_det_J_C_raw_to_C_double + log_det_J_unc_to_C_raw_double;
        return P;

}




//// -------------------------------------------------------------------------------------
//// Everything after the chunk loop.
//// -------------------------------------------------------------------------------------
inline void fn_MVOP_assemble(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                               const MVOP_prep &P,
                               const double log_jac_u,
                               const std::vector<Eigen::Matrix<double, -1, -1>> &beta_grad_array,
                               const std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
                               const std::vector<Eigen::Matrix<double, -1, -1>> &cutpoint_grad_array,
                               const Eigen::Matrix<double, -1, -1> &prev_grad_mat
) {

        const int N = P.N;
        const int n_tests = P.n_tests;
        const int n_us = P.n_us;
        const int n_class = P.n_class;
        const int n_pops = P.n_pops;
        const int n_corrs = P.n_corrs;
        const int n_covariates_total = P.n_covariates_total;
        const int dim_choose_2 = P.dim_choose_2;
        const Eigen::Matrix<int, -1, -1> &ncov = *P.n_covariates_per_outcome_vec;
        const Eigen::Matrix<int, -1, 1> &n_thr_per_ord_test = *P.n_thr_per_ord_test;

        //// cutpoint grads: C -> C_raw -> theta
        int out_idx = 1 + n_us + n_corrs + n_covariates_total + (n_class > 1 ? n_pops : 0);
        for (int c = 0; c < n_class; c++) for (int t_ord = 0; t_ord < P.n_ordinal_tests; t_ord++) {
          const int n_thr_t = n_thr_per_ord_test(t_ord);
          Eigen::Matrix<double, -1, 1> g_C = cutpoint_grad_array[c].col(t_ord).head(n_thr_t);
          Eigen::Matrix<double, -1, 1> g_raw = chain_rule_C_to_C_raw(g_C, P.C_raw[c].col(t_ord).head(n_thr_t));
          g_raw.array() *= P.dC_raw_dunc[c].col(t_ord).head(n_thr_t).array();
          out_mat.segment(out_idx, n_thr_t) += g_raw; out_idx += n_thr_t;
        }

        Eigen::Matrix<double, -1, 1> prev_unc_grad = Eigen::Matrix<double, -1, 1>::Zero(n_pops);
        if (n_class > 1) for (int g = 0; g < n_pops; g++) prev_unc_grad(g) = (prev_grad_mat(g, 1) - prev_grad_mat(g, 0)) * P.deriv_p_wrt_u_vec(g) - 2.0 * P.tanh_u_prev_vec(g);

        double log_prob_out = out_mat.tail(N).sum() + log_jac_u + P.log_det_J_main;
        if (!P.exclude_priors) log_prob_out += P.prior_densities;

        Eigen::Matrix<double, -1, 1> beta_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);
        { int i = 0; for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < ncov(c, t); k++) beta_grad_vec(i++) = beta_grad_array[c](k, t); }
        Eigen::Matrix<double, -1, 1> L_Omega_grad_vec(n_corrs + n_class * n_tests);
        Eigen::Matrix<double, -1, 1> U_Omega_grad_vec(n_corrs);
        { int i = 0; for (int c = 0; c < n_class; c++) for (int t1 = 0; t1 < n_tests; t1++) for (int t2 = 0; t2 < t1 + 1; t2++) L_Omega_grad_vec(i++) = U_Omega_grad_array[c](t1, t2); }
        if (n_class > 1) {
          U_Omega_grad_vec.head(dim_choose_2) = (L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests).transpose() * P.deriv_L_wrt_unc_full[0]).transpose();
          U_Omega_grad_vec.segment(dim_choose_2, dim_choose_2) = (L_Omega_grad_vec.segment(dim_choose_2 + n_tests, dim_choose_2 + n_tests).transpose() * P.deriv_L_wrt_unc_full[1]).transpose();
        } else {
          U_Omega_grad_vec.head(dim_choose_2) = (L_Omega_grad_vec.head(dim_choose_2 + n_tests).transpose() * P.deriv_L_wrt_unc_full[0]).transpose();
        }
        out_mat(0) = log_prob_out;
        out_mat.segment(1 + n_us, n_corrs) += U_Omega_grad_vec;
        out_mat.segment(1 + n_us + n_corrs, n_covariates_total) += beta_grad_vec;
        if (n_class > 1) out_mat.segment(1 + n_us + n_corrs + n_covariates_total, n_pops) += prev_unc_grad;
        if (!P.exclude_priors) {
          const std::vector<Eigen::Matrix<double, -1, -1>> &pm = *P.prior_coeffs_mean;
          const std::vector<Eigen::Matrix<double, -1, -1>> &ps = *P.prior_coeffs_sd;
          int i = n_us + n_corrs + 1;
          for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < ncov(c, t); k++)
          {
            out_mat(i++) += -((P.beta_double_array[c](k, t) - pm[c](k, t)) / ps[c](k, t)) * (1.0 / ps[c](k, t));
          }
        }

}






















