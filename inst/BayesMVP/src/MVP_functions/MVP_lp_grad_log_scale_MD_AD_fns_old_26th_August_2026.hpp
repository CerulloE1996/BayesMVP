
//// MVP_lp_grad_log_scale_MD_AD_fns.hpp






#pragma once



#include <Eigen/Dense>




#include <unsupported/Eigen/SpecialFunctions>









/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Internal per-chunk routine (lp + grad + log-scale fix-ups for ONE chunk). Called once per
//// full chunk with the SIMD-configured struct, and once for the remainder chunk with the
//// "Stan" (scalar) copy - so ALL helper fns run scalar on the remainder chunk:
inline void fn_lp_grad_MVP_LC_Pinkney_PartialLog_process_chunk(   Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec,
                                                                         const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                         const std::string &grad_option,
                                                                         const int chunk_counter,
                                                                         const int chunk_size,
                                                                         const int chunk_size_orig,
                                                                         const int N,
                                                                         const int n_params,
                                                                         const int n_tests,
                                                                         const int n_class,
                                                                         const int n_covariates_max,
                                                                         const int n_pops,
                                                                         const double overflow_threshold,
                                                                         const double underflow_threshold,
                                                                         const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec,
                                                                         const Eigen::Matrix<int, -1, 1> &pop_ind,
                                                                         const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X,
                                                                         const std::vector<Eigen::Matrix<double, -1, -1>> &beta_double_array,
                                                                         const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_double,
                                                                         const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_recip_double,
                                                                         const std::vector<Eigen::Matrix<double, -1, -1>> &log_abs_L_Omega_double,
                                                                         const Eigen::Matrix<double, -1, -1> &prev_mat,
                                                                         const Eigen::Matrix<double, -1, -1> &log_prev_mat_small,
                                                                         double &log_jac_u,
                                                                         std::vector<Eigen::Matrix<double, -1, -1>> &beta_grad_array,
                                                                         std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
                                                                         Eigen::Matrix<double, -1, -1> &prev_grad_mat,
                                                                         const Model_fn_args_struct &Model_args_for_chunk
) {

        const double sqrt_2_pi_recip = 1.0 / sqrt(2.0 * M_PI);
        const double sqrt_2_recip = 1.0 / std::sqrt(2.0);
        const double minus_sqrt_2_recip = -sqrt_2_recip;
        const double a = 0.07056;
        const double b = 1.5976;
        const double a_times_3 = 3.0 * a;
        const double s = 1.0 / 1.702;
        const double Inf = std::numeric_limits<double>::infinity();
        ////
        //// vect-type strings read from the PASSED struct - SIMD for full chunks, "Stan" for the
        //// remainder chunk (the driver passes a "Stan" copy of the struct), so the direct
        //// fn_EIGEN_double / log_sum_exp_general / log_sum_vec_signed_v1 calls below automatically
        //// match what the helper fns receive:
        const std::string vect_type = Model_args_for_chunk.Model_args_strings(0);
        const std::string &Phi_type = Model_args_for_chunk.Model_args_strings(1);
        const std::string &inv_Phi_type = Model_args_for_chunk.Model_args_strings(2);
        const std::string vect_type_exp = Model_args_for_chunk.Model_args_strings(3);
        const std::string vect_type_log = Model_args_for_chunk.Model_args_strings(4);
        const std::string vect_type_lse = Model_args_for_chunk.Model_args_strings(5);
        const std::string vect_type_tanh = Model_args_for_chunk.Model_args_strings(6);
        const std::string vect_type_Phi = Model_args_for_chunk.Model_args_strings(7);
        const std::string vect_type_log_Phi = Model_args_for_chunk.Model_args_strings(8);
        const std::string vect_type_inv_Phi = Model_args_for_chunk.Model_args_strings(9);
        const std::string vect_type_inv_Phi_approx_from_logit_prob = Model_args_for_chunk.Model_args_strings(10);

        ///////////////////////////////////////////////
        ///////////////////////////////////////////////
        ///////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //// containers allocated directly at last_chunk_size (no mid-loop resizing needed):
        std::vector<Eigen::Matrix<double, -1, -1>>   Z_std_norm =  vec_of_mats<double>(chunk_size, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>>   Bound_Z =  Z_std_norm ;
        std::vector<Eigen::Matrix<double, -1, -1>>   Bound_U_Phi_Bound_Z =  Z_std_norm ;
        std::vector<Eigen::Matrix<double, -1, -1>>   Phi_Z =  Z_std_norm ;
        std::vector<Eigen::Matrix<double, -1, -1>>   phi_Bound_Z =  Z_std_norm ;
        std::vector<Eigen::Matrix<double, -1, -1>>   y1_log_prob =  Z_std_norm ;
        std::vector<Eigen::Matrix<double, -1, -1>>   prob =  Z_std_norm ;
        std::vector<Eigen::Matrix<double, -1, -1>>   phi_Z_recip =  Z_std_norm ;
        ///////////////////////////////////////////////
        std::vector<Eigen::Matrix<double, -1, -1>>     log_phi_Z_recip =  vec_of_mats<double>(chunk_size, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>>     log_phi_Bound_Z =  vec_of_mats<double>(chunk_size, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>>     log_Z_std_norm =   vec_of_mats<double>(chunk_size, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>>     log_abs_Bound_Z =  vec_of_mats<double>(chunk_size, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>>     sign_Bound_Z =     vec_of_mats<double>(chunk_size, n_tests, n_class);
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, -1> y_chunk =        Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
        Eigen::Matrix<double, -1, -1> u_array =        Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
        Eigen::Matrix<double, -1, -1> y_sign_chunk =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
        Eigen::Matrix<double, -1, -1> y_m_y_sign_x_u = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, 1>  inc_array  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk   =           Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
        Eigen::Matrix<double, -1, -1> log_abs_u_grad_array_CM_chunk   =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, 1> log_sum_result =           Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
        Eigen::Matrix<double, -1, 1> log_sum_abs_result =       Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
        Eigen::Matrix<double, -1, 1> sign_result =              Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
        Eigen::Matrix<double, -1, 1> container_max_logs =       Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
        Eigen::Matrix<double, -1, 1> container_sum_exp_signed = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, 1> u_unc_vec_chunk =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
        Eigen::Matrix<double, -1, 1> u_vec_chunk =        Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
        Eigen::Matrix<double, -1, 1> du_wrt_duu_chunk =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
        Eigen::Matrix<double, -1, 1> d_J_wrt_duu_chunk =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, -1>     log_common_grad_term_1   =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0);
        Eigen::Matrix<double, -1, -1>     log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
        Eigen::Matrix<double, -1, -1>     log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
        Eigen::Matrix<double, -1, -1>     sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests) ;
        Eigen::Matrix<double, -1, -1>     sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests) ;
        Eigen::Matrix<double, -1, -1>     log_prob_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
        Eigen::Matrix<double, -1, -1>     log_prob_recip_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
        Eigen::Matrix<double, -1, -1>     log_abs_grad_prob =    Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0);
        Eigen::Matrix<double, -1, -1>     log_abs_z_grad_term =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0);
        Eigen::Matrix<double, -1, -1>     sign_grad_prob =    Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
        Eigen::Matrix<double, -1, -1>     sign_z_grad_term =  Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, -1>     log_abs_prod_container_or_inc_array_comp  =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests, -700.0);
        Eigen::Matrix<double, -1, -1>     sign_prod_container_or_inc_array_comp  =     Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
        Eigen::Matrix<double, -1, -1>     log_abs_derivs_chain_container_vec_comp =    Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests, -700.0);
        Eigen::Matrix<double, -1, -1>     sign_derivs_chain_container_vec_comp =       Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, 1>      log_abs_prod_container_or_inc_array  =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
        Eigen::Matrix<double, -1, 1>      sign_prod_container_or_inc_array  =     Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
        Eigen::Matrix<double, -1, 1>      log_abs_derivs_chain_container_vec  =   Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
        Eigen::Matrix<double, -1, 1>      sign_derivs_chain_container_vec  =      Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
        Eigen::Matrix<double, -1, 1>      log_prob_rowwise_prod_temp_all  =       Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, 1>  log_abs_prev_grad_array_col_for_each_n =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
        Eigen::Matrix<double, -1, 1>  sign_prev_grad_array_col_for_each_n =     Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, 1>  log_abs_a  =       Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
        Eigen::Matrix<double, -1, 1>  sign_a =           Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
        Eigen::Matrix<double, -1, 1>  log_abs_b =        Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
        Eigen::Matrix<double, -1, 1>  sign_b =           Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
        Eigen::Matrix<double, -1, 1>  sign_sum_result =  Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
        Eigen::Matrix<double, -1, -1> log_terms =        Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests, -700.0);
        Eigen::Matrix<double, -1, -1> sign_terms =       Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
        Eigen::Matrix<double, -1, 1>  final_log_sum =    Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
        Eigen::Matrix<double, -1, 1>  final_sign =       Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
        ///////////////////////////////////////////////
        Eigen::VectorXi overflow_mask(chunk_size);
        Eigen::VectorXi underflow_mask(chunk_size);
        Eigen::VectorXi OK_mask(chunk_size);
        /////////////////////////////////////////////  ////-----------------------------------------------
        std::vector<std::vector<std::vector<int>>> problem_index_array(n_class);
        std::vector<std::vector<int>> n_problem_array(n_class);
        for (int c = 0; c < n_class; c++) {
          problem_index_array[c].resize(n_tests); // initialise
          n_problem_array[c].resize(n_tests);     // initialise
          for (int t = 0; t < n_tests; t++) {
            n_problem_array[c][t] = 0; // initialise
          }
        }
        /////////////////////////////////////////////  ////-----------------------------------------------
        std::vector<Eigen::Matrix<double, -1, -1>>   Omega_grad_array_for_each_n =     vec_of_mats<double>(chunk_size, n_tests, n_tests);
        std::vector<Eigen::Matrix<double, -1, -1>>   sign_Omega_grad_array_for_each_n =     Omega_grad_array_for_each_n;
        std::vector<Eigen::Matrix<double, -1, -1>>   log_abs_Omega_grad_array_for_each_n =  Omega_grad_array_for_each_n;
        std::vector<Eigen::Matrix<double, -1, -1>>   beta_grad_array_for_each_n =      vec_of_mats<double>(chunk_size, n_tests, n_covariates_max);
        std::vector<Eigen::Matrix<double, -1, -1>>   sign_beta_grad_array_for_each_n =      beta_grad_array_for_each_n;
        std::vector<Eigen::Matrix<double, -1, -1>>   log_abs_beta_grad_array_for_each_n =   beta_grad_array_for_each_n;
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, 1>  rowwise_log_sum = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
        Eigen::Matrix<double, -1, 1>  rowwise_prod =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
        Eigen::Matrix<double, -1, 1>  rowwise_sum =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
        ///////////////////////////////////////////////
        Eigen::Matrix<double, -1, 1> log_prev_per_obs_given_c = Eigen::Matrix<double, -1, 1>::Zero(chunk_size); //// ----
        Eigen::Matrix<double, -1, 1> prev_per_obs_given_c     = Eigen::Matrix<double, -1, 1>::Zero(chunk_size); //// ----
        ///////////////////////////////////////////////

        Eigen::Matrix<double, -1, -1>    lp_array  =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_class);

        { ////-----------------------------------------------
          u_grad_array_CM_chunk.setZero() ; //// reset between chunks as re-using same container
          log_abs_u_grad_array_CM_chunk.setConstant(-700);  // reset between chunks as re-using same container

          y_chunk = y_ref.middleRows( chunk_size_orig * chunk_counter, chunk_size).cast<double>();

          //// Nuisance parameter transformation step
          u_unc_vec_chunk = u_unc_vec.segment( chunk_size_orig * n_tests * chunk_counter, chunk_size * n_tests);

          fn_MVP_compute_nuisance( u_vec_chunk, u_unc_vec_chunk, Model_args_for_chunk);
          log_jac_u +=    fn_MVP_compute_nuisance_log_jac_u(   u_vec_chunk, u_unc_vec_chunk, Model_args_for_chunk);

          u_array  =  u_vec_chunk.reshaped(chunk_size, n_tests);
          y_sign_chunk.array() =      y_chunk.array() + (y_chunk.array() - 1.0) ;
          y_m_y_sign_x_u.array() =  y_chunk.array() - (y_sign_chunk.array() * u_array.array());
        } ////-----------------------------------------------


        {
          // START of c loop
          for (int c = 0; c < n_class; c++) {

            inc_array.setZero();  //// needs to be reset to 0

            if (n_class > 1) { //// ----

              for (int n = 0; n < chunk_size; ++n) {

                int n_global = chunk_size_orig * chunk_counter + n;
                int g = pop_ind(n_global);
                log_prev_per_obs_given_c(n) = log_prev_mat_small(g, c);
                prev_per_obs_given_c(n) = prev_mat(g, c);

              }

            }

            for (int t = 0; t < n_tests; t++) {   //// start of t loop

              if (n_covariates_max > 1) {

                const Eigen::Matrix<double, -1, 1> Xbeta_given_class_c_col_t = X[c][t].block(chunk_size_orig * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>() * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
                Bound_Z[c].col(t) =     L_Omega_recip_double[c](t, t) * (  -1.0*( Xbeta_given_class_c_col_t + inc_array  )  ) ;
                sign_Bound_Z[c].col(t) =   stan::math::sign(Bound_Z[c].col(t));
                log_abs_Bound_Z[c].col(t) =    fn_EIGEN_double( stan::math::abs(Bound_Z[c].col(t)), "log", vect_type_log);

              } else {  //// intercept-only

                Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * ( -1.0*( beta_double_array[c](0, t) + inc_array.array() ) ) ;

                { ////-----------------------------------------------
                  Eigen::Matrix<double, -1, 1> Bound_Z_col_t = Bound_Z[c].col(t);
                  Eigen::Matrix<double, -1, 1> Bound_Z_col_t_sign = stan::math::sign(Bound_Z_col_t);
                  Eigen::Matrix<double, -1, 1> Bound_Z_col_t_abs =  stan::math::abs(Bound_Z_col_t);
                  ////
                  sign_Bound_Z[c].col(t) =   Bound_Z_col_t_sign;
                  log_abs_Bound_Z[c].col(t) =    fn_EIGEN_double( Bound_Z_col_t_abs, "log", vect_type_log);
                } ////-----------------------------------------------

              }

              //// create masks
              { ////-----------------------------------------------
                Eigen::Matrix<double, -1, 1> y_chunk_col_t_dbl = y_chunk.col(t);
                Eigen::Matrix<double, -1, 1> Bound_Z_col_t = Bound_Z[c].col(t);
                overflow_mask.array() =   ( (Bound_Z_col_t.array() > overflow_threshold)  && (y_chunk_col_t_dbl.array() == 1.0) ).cast<int>()  ;
                underflow_mask.array() =  ( (Bound_Z_col_t.array() < underflow_threshold) && (y_chunk_col_t_dbl.array() == 0.0) ).cast<int>()  ;
                OK_mask.array() = ( (overflow_mask.array() == 0) && (underflow_mask.array() == 0) ).cast<int>() ;
              } ////-----------------------------------------------

              //// counts
              const int n_overflows =   overflow_mask.sum();
              const int n_underflows =  underflow_mask.sum();
              const int n_problem = n_overflows + n_underflows;
              const int n_OK = OK_mask.sum();

              std::vector<int> over_index(n_overflows, 0);
              std::vector<int> under_index(n_underflows, 0);
              std::vector<int> problem_index(n_problem, 0);

              int counter_over  = 0;
              int counter_under  = 0;

              for (int n = 0; n < chunk_size; ++n) {

                if  (overflow_mask(n) == 1) {
                  over_index[counter_over] = n;
                  counter_over += 1;
                } else if  (underflow_mask(n) == 1)   {
                  under_index[counter_under] = n;
                  counter_under += 1;
                } else {
                  // OK_index[counter_ok] = n;
                  // counter_ok += 1;
                }

              }

              {
                int counter = 0;

                for (int i = 0; i < n_overflows; ++i) {
                  problem_index[counter] = over_index[i];
                  counter += 1;
                }

                for (int i = 0; i < n_underflows; ++i) {
                  problem_index[counter] = under_index[i];
                  counter += 1;
                }
              }

              //// fill the index arrays
              n_problem_array[c][t] = n_problem;
              problem_index_array[c][t].resize(n_problem);
              problem_index_array[c][t] = problem_index;

              //// compute/update important log-lik quantities for GHK-MVP
              fn_MVP_compute_lp_GHK_cols(  t,
                                           Bound_U_Phi_Bound_Z[c], // computing this
                                           Phi_Z[c], // computing this
                                           Z_std_norm[c], // computing this
                                           prob[c],        // computing this
                                           y1_log_prob[c], // computing this
                                           Bound_Z[c],
                                           y_chunk,
                                           u_array,
                                           Model_args_for_chunk);

              //// compute/update important grad quantities for GHK-MVP
              fn_MVP_compute_phi_Z_recip_cols(    t,
                                                  phi_Z_recip[c], // computing this
                                                  Phi_Z[c], Z_std_norm[c], Model_args_for_chunk);

              fn_MVP_compute_phi_Bound_Z_cols(     t,
                                                   phi_Bound_Z[c], // computing this
                                                   Bound_U_Phi_Bound_Z[c], Bound_Z[c], Model_args_for_chunk);

              //// compute log-scale quantities (for grad)
              { ////-----------------------------------------------
                    Eigen::Matrix<double, -1, -1> temp_mat =       Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
                    Eigen::Matrix<double, -1, 1>  temp_col_t =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
                    Eigen::Matrix<double, -1, 1>  temp_col_t_abs = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);

                    temp_mat = phi_Bound_Z[c];
                    temp_col_t = temp_mat.col(t);
                    temp_col_t_abs = stan::math::abs(temp_col_t);
                    log_phi_Bound_Z[c].col(t) =   fn_EIGEN_double( temp_col_t_abs,  "log", vect_type_log);

                    temp_mat = phi_Z_recip[c];
                    temp_col_t = temp_mat.col(t);
                    temp_col_t_abs = stan::math::abs(temp_col_t);
                    log_phi_Z_recip[c].col(t) =   fn_EIGEN_double( temp_col_t_abs,  "log", vect_type_log);

                    temp_mat = Z_std_norm[c];
                    temp_col_t = temp_mat.col(t);
                    temp_col_t_abs = stan::math::abs(temp_col_t);
                    log_Z_std_norm[c].col(t)  =   fn_EIGEN_double( temp_col_t_abs,  "log", vect_type_log);
              } ////-----------------------------------------------

              if (n_OK == chunk_size)  {
                    //// carry on as normal as (likely) no * problematic * overflows/underflows
              }  else if (n_OK < chunk_size)  {

                    //----------------------------------------------- currently testing this
                    if (n_underflows > 0) { //// underflow (w/ y == 0)

                      const std::vector<int> index = under_index;
                      const int index_size = index.size();

                      fn_MVP_compute_lp_GHK_cols_log_scale_underflow(  t,
                                                                       index,
                                                                       Bound_U_Phi_Bound_Z[c], // computing this
                                                                       Phi_Z[c],  // computing this
                                                                       Z_std_norm[c],  // computing this
                                                                       log_Z_std_norm[c],  // computing this
                                                                       prob[c],  // computing this
                                                                       y1_log_prob[c], // computing this
                                                                       log_phi_Bound_Z[c], // computing this
                                                                       log_phi_Z_recip[c], // computing this
                                                                       Bound_Z[c],
                                                                       u_array,
                                                                       Model_args_for_chunk);

                      phi_Bound_Z[c](index, t) =  fn_EIGEN_double( log_phi_Bound_Z[c](index, t),  "exp", vect_type_exp); // not needed if comp. grad on log-scale
                      phi_Z_recip[c](index, t) =  fn_EIGEN_double( log_phi_Z_recip[c](index, t),  "exp", vect_type_exp); // not needed if comp. grad on log-scale

                    }
                    //----------------------------------------------- currently testing this

                    //-----------------------------------------------
                    if (n_overflows > 0) { //// overflow (w/ y == 1)

                      const std::vector<int> index = over_index;
                      const int index_size = index.size();

                      fn_MVP_compute_lp_GHK_cols_log_scale_overflow( t,
                                                                     n_overflows,
                                                                     index,
                                                                     Bound_U_Phi_Bound_Z[c],  // computing this
                                                                     Phi_Z[c],  // computing this
                                                                     Z_std_norm[c],  // computing this
                                                                     log_Z_std_norm[c],  // computing this
                                                                     prob[c],  // computing this
                                                                     y1_log_prob[c],  // computing this
                                                                     log_phi_Bound_Z[c],  // computing this
                                                                     log_phi_Z_recip[c],  // computing this
                                                                     Bound_Z[c],
                                                                     u_array,
                                                                     Model_args_for_chunk);

                      phi_Bound_Z[c](index, t) =  fn_EIGEN_double( log_phi_Bound_Z[c](index, t),  "exp", vect_type_exp);  // not needed if comp. grad on log-scale
                      phi_Z_recip[c](index, t) =  fn_EIGEN_double( log_phi_Z_recip[c](index, t),  "exp", vect_type_exp);  // not needed if comp. grad on log-scale

                    }
                    //-----------------------------------------------

              }  ///// end of "if overflow or underflow" block

              if (t < n_tests - 1) {

                    //// -----------------------------------------------
                    Eigen::Matrix<double, 1, -1> L_Omega_row = L_Omega_double[c].row(t + 1);
                    inc_array = Z_std_norm[c].leftCols(t + 1) * L_Omega_row.head(t + 1).transpose();
                    //// -----------------------------------------------

              }

            }  //// end of t loop

            //// -----------------------------------------------
            if (n_class > 1) {  //// if latent class
              rowwise_sum = y1_log_prob[c].rowwise().sum();
              // rowwise_sum.array() += log_prev(0, c);
              rowwise_sum.array() += log_prev_per_obs_given_c.array(); //// ----
              lp_array.col(c) = rowwise_sum;
            } else {
              rowwise_sum = y1_log_prob[c].rowwise().sum();
              lp_array.col(0) =     rowwise_sum;
            }
            //// -----------------------------------------------

          }   //// end of c loop

        }  //// end of local block

        if (n_class > 1) {  /// if latent class

          //// -----------------------------------------------
          log_sum_exp_general(   lp_array,
                                 vect_type_exp,
                                 vect_type_log,
                                 log_sum_result,
                                 container_max_logs);
          const int index_start = 1 + n_params + chunk_size_orig * chunk_counter;
          out_mat.segment(index_start, chunk_size) = log_sum_result;
          //// -----------------------------------------------

        } else {

          const int index_start = 1 + n_params + chunk_size_orig * chunk_counter;
          out_mat.segment(index_start, chunk_size) = lp_array.col(0);

        }

        Eigen::Matrix<double, -1, 1> log_prob_n_recip = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
        Eigen::Matrix<double, -1, 1> prob_n_recip     = Eigen::Matrix<double, -1, 1>::Zero(chunk_size); //// ----
        { //// -----------------------------------------------
          Eigen::Matrix<double, -1, 1> log_lik = out_mat.tail(N);
          Eigen::Matrix<double, -1, 1> log_lik_segment = log_lik.segment(chunk_size_orig * chunk_counter, chunk_size);
          Eigen::Matrix<double, -1, 1> prob_n  =  fn_EIGEN_double( log_lik_segment, "exp",  vect_type_exp);
          prob_n_recip = stan::math::inv(prob_n);
          log_prob_n_recip = fn_EIGEN_double( prob_n_recip, "log", vect_type_log);
        } //// -----------------------------------------------

        const bool compute_final_scalar_grad = false;

        /////////////////////////////////////////////////
        ///////////////// ------------------------- compute grad  -----------------------------------------------------------------------------------------------------------------
        for (int c = 0; c < n_class; c++) {

          if (n_class > 1) { //// ----
            for (int n = 0; n < chunk_size; ++n) {
              int n_global = chunk_size_orig * chunk_counter + n;
              int g = pop_ind(n_global);
              log_prev_per_obs_given_c(n) = log_prev_mat_small(g, c);
              // prev_per_obs_given_c not needed in log-scale grad path
            }
          }

          ////-----------------------------------------------
          for (int i = 0; i < beta_grad_array_for_each_n.size(); i++) {
            beta_grad_array_for_each_n[i].setZero();
            sign_beta_grad_array_for_each_n[i].setOnes();
            log_abs_beta_grad_array_for_each_n[i].setConstant(-700.0);
          }
          for (int i = 0; i < Omega_grad_array_for_each_n.size(); i++) {
            Omega_grad_array_for_each_n[i].setZero();
            sign_Omega_grad_array_for_each_n[i].setOnes();
            log_abs_Omega_grad_array_for_each_n[i].setConstant(-700.0);
          }

          Eigen::Matrix<double, -1, -1> y1_log_prob_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> sign_Z_std_norm =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> prob_recip =        Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> common_grad_term_1 =                                                         Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> prob_rowwise_prod_temp =                                                     Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip =                        Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          ////-----------------------------------------------

          { ////-----------------------------------------------

            Eigen::Matrix<double, -1, -1> temp_mat =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
            Eigen::Matrix<double, -1, -1> temp_mat_2 = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);

            {
              temp_mat = y1_log_prob[c];
              temp_mat_2 = -1.0*temp_mat;
              y1_log_prob_recip = temp_mat_2;
            }
            {
              temp_mat = Z_std_norm[c];
              temp_mat_2 = stan::math::sign(temp_mat);
              sign_Z_std_norm = temp_mat_2;
            }
            {
              temp_mat = prob[c];
              temp_mat_2 = stan::math::inv(temp_mat);
              prob_recip = temp_mat_2;
            }

          } ////-----------------------------------------------

          if ( (grad_option != "none") || (grad_option == "test") ) {

            ////-----------------------------------------------
            Eigen::Matrix<double, -1, -1> abs_L_Omega_recip_double =     Eigen::Matrix<double, -1, -1>::Zero(n_tests, n_tests);
            Eigen::Matrix<double, -1, -1> log_abs_L_Omega_recip_double = Eigen::Matrix<double, -1, -1>::Zero(n_tests, n_tests);
            Eigen::Matrix<double, -1, -1> sign_L_Omega_recip_double =    Eigen::Matrix<double, -1, -1>::Ones(n_tests, n_tests);
            ////-----------------------------------------------

            ////-----------------------------------------------
            abs_L_Omega_recip_double =  stan::math::abs(L_Omega_recip_double[c]);    //  std::cout << "After abs" << std::endl;
            sign_L_Omega_recip_double = stan::math::sign(L_Omega_recip_double[c]);   //  std::cout << "After sign" << std::endl;
            for (int t = 0; t < n_tests; t++) {
              log_abs_L_Omega_recip_double(t, t) = stan::math::log(abs_L_Omega_recip_double(t, t));
            }
            ////-----------------------------------------------

            fn_MVP_grad_prep_log_scale(      log_prob_rowwise_prod_temp,
                                             log_prob_recip_rowwise_prod_temp,
                                             log_prob_rowwise_prod_temp_all,
                                             log_common_grad_term_1,
                                             log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                             log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                             sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                             sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                             y1_log_prob[c],
                                             y1_log_prob_recip,
                                             log_prob_n_recip,
                                             ////
                                             log_prev_per_obs_given_c, /// if not latent class, this is a dummy variable //// ----
                                             ////
                                             log_phi_Bound_Z[c],
                                             log_phi_Z_recip[c],
                                             log_abs_L_Omega_recip_double,
                                             sign_L_Omega_recip_double,
                                             y_sign_chunk,
                                             y_m_y_sign_x_u,
                                             Model_args_for_chunk);

            ////-----------------------------------------------
            //// these should all be OK on windows  (not dangling reference)
            common_grad_term_1 = fn_EIGEN_double(log_common_grad_term_1, "exp", vect_type_exp);
            prob_rowwise_prod_temp = fn_EIGEN_double(log_prob_rowwise_prod_temp, "exp", vect_type_exp);
            y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip  =  fn_EIGEN_double(log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip, "exp", vect_type_exp).array() *
              sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.array();
            y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = fn_EIGEN_double(log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip, "exp", vect_type_exp).array() *
              sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.array();
            ////-----------------------------------------------

          }

          //// --- up to here OK  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

          ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of nuisance parameters / u's (manual)
          if ( (grad_option == "us_only") || (grad_option == "all") ) {

            Eigen::Matrix<double, -1, -1>   u_grad_array_CM_chunk_block = u_grad_array_CM_chunk;

            { /// first compute gradients on the standard (non-log) scale.

              fn_MVP_compute_nuisance_grad_v2(   u_grad_array_CM_chunk_block,
                                                 phi_Z_recip[c],
                                                 common_grad_term_1,
                                                 L_Omega_double[c],
                                                 prob[c],
                                                 prob_recip,
                                                 prob_rowwise_prod_temp,
                                                 y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                 y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                 log_abs_z_grad_term,
                                                 log_abs_grad_prob,
                                                 log_abs_prod_container_or_inc_array,
                                                 sign_prod_container_or_inc_array,
                                                 Model_args_for_chunk);

            }

            { /// then compute gradients on the LOG-scale, but ONLY where we have underflow or overflow.

              fn_MVP_compute_nuisance_grad_log_scale(    n_problem_array[c],
                                                         problem_index_array[c],
                                                         log_abs_u_grad_array_CM_chunk,
                                                         u_grad_array_CM_chunk_block,
                                                         L_Omega_double[c],
                                                         log_abs_L_Omega_double[c],
                                                         log_phi_Z_recip[c],
                                                         y1_log_prob[c],
                                                         y1_log_prob_recip,
                                                         log_prob_rowwise_prod_temp,
                                                         log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                         sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                         log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                         sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                         log_common_grad_term_1,
                                                         log_abs_z_grad_term,
                                                         sign_z_grad_term,
                                                         log_abs_grad_prob,
                                                         sign_grad_prob,
                                                         log_abs_prod_container_or_inc_array,
                                                         sign_prod_container_or_inc_array,
                                                         log_sum_result,
                                                         sign_sum_result,
                                                         log_terms,
                                                         sign_terms,
                                                         log_abs_a,
                                                         log_abs_b,
                                                         sign_a,
                                                         sign_b,
                                                         container_max_logs,
                                                         container_sum_exp_signed,
                                                         Model_args_for_chunk);

            }

            //// update u_grad_array_CM_chunk once standard-scale and log-scale grad computations are done
            u_grad_array_CM_chunk.array() += u_grad_array_CM_chunk_block.array();

            const int start_index = 1 + (chunk_size_orig * n_tests * chunk_counter);
            const int length = chunk_size * n_tests;

            if (c == n_class - 1) {

              //// update output vector once all u_grad computations are done
              out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();

              //// account for unconstrained -> constrained transformations and Jacobian adjustments
              fn_MVP_nuisance_first_deriv( du_wrt_duu_chunk,
                                           u_vec_chunk, u_unc_vec_chunk, Model_args_for_chunk);

              fn_MVP_nuisance_deriv_of_log_det_J(    d_J_wrt_duu_chunk,
                                                     u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, Model_args_for_chunk);

              out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
              out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();

            }

            sign_z_grad_term.setOnes();
            sign_grad_prob.setOnes();
            sign_prod_container_or_inc_array.setOnes();
            sign_sum_result.setOnes();
            sign_terms.setOnes();
            sign_a.setOnes();
            sign_b.setOnes();

            log_abs_z_grad_term.setConstant(-700.0);
            log_abs_grad_prob.setConstant(-700.0);
            log_abs_prod_container_or_inc_array.setConstant(-700.0);
            log_sum_result.setConstant(-700.0);
            log_terms.setConstant(-700.0);
            log_abs_a.setConstant(-700.0);
            log_abs_b.setConstant(-700.0);
            container_max_logs.setConstant(-700.0);

          }


          ///////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)
          if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only") ) {

            if (n_covariates_max > 1) {  //// log-scale grad for coefficients not (yet) implemented for > 1 covariate (e.g., standard-MVP)

              Eigen::Matrix<int, -1, 1> n_cov_vec_c = n_covariates_per_outcome_vec.row(c).transpose();

              fn_MVP_compute_coefficients_grad_v2(   c,
                                                     beta_grad_array[c],
                                                     X[c],
                                                     n_cov_vec_c,
                                                     beta_grad_array_for_each_n,
                                                     chunk_counter,
                                                     n_covariates_max,
                                                     common_grad_term_1,
                                                     L_Omega_double[c],
                                                     prob[c],
                                                     prob_recip,
                                                     prob_rowwise_prod_temp,
                                                     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                     log_abs_grad_prob,
                                                     sign_grad_prob,
                                                     log_abs_prod_container_or_inc_array,
                                                     sign_prod_container_or_inc_array,
                                                     true, /// compute_final_scalar_grad
                                                     Model_args_for_chunk);

            } else {

              {   // compute (some or all) of grads on log-scale

                { /// first compute gradients on the standard (non-log) scale.

                  log_abs_grad_prob.setZero();
                  sign_grad_prob.setZero();
                  log_abs_prod_container_or_inc_array.setZero();
                  sign_prod_container_or_inc_array.setZero();

                  Eigen::Matrix<int, -1, 1> n_cov_vec_c = n_covariates_per_outcome_vec.row(c).transpose();

                  fn_MVP_compute_coefficients_grad_v2(    c,
                                                          beta_grad_array[c],
                                                          X[c],
                                                          n_cov_vec_c,
                                                          beta_grad_array_for_each_n,
                                                          chunk_counter,
                                                          n_covariates_max,
                                                          common_grad_term_1,
                                                          L_Omega_double[c],
                                                          prob[c],
                                                          prob_recip,
                                                          prob_rowwise_prod_temp,
                                                          y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                          y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                          log_abs_grad_prob,
                                                          sign_grad_prob,
                                                          log_abs_prod_container_or_inc_array,
                                                          sign_prod_container_or_inc_array,
                                                          false, /// compute_final_scalar_grad
                                                          Model_args_for_chunk);

                }

                { // then compute gradients on the LOG-scale, but ONLY where we have underflow or overflow.

                  //// extract previous signs and log_abs values
                  for (int i = 0; i <  beta_grad_array_for_each_n.size();  i++) {
                    auto temp_array = beta_grad_array_for_each_n[i];
                    auto temp_array_sign = stan::math::sign(temp_array);
                    sign_beta_grad_array_for_each_n[i] =  temp_array_sign;
                    auto temp_array_abs = stan::math::abs(temp_array);
                    log_abs_beta_grad_array_for_each_n[i] = fn_EIGEN_double( temp_array_abs, "log", vect_type_log);
                    for (int t = 0; t < n_tests; t++) {
                      sign_beta_grad_array_for_each_n[i].col(t)(problem_index_array[c][t]).setOnes();
                      log_abs_beta_grad_array_for_each_n[i].col(t)(problem_index_array[c][t]).setConstant(-700.0);
                    }
                  }

                  fn_MVP_compute_coefficients_grad_log_scale( n_problem_array[c],
                                                              problem_index_array[c],
                                                              beta_grad_array[c],
                                                              sign_beta_grad_array_for_each_n,
                                                              log_abs_beta_grad_array_for_each_n,
                                                              L_Omega_double[c],
                                                              log_abs_L_Omega_double[c],
                                                              log_phi_Z_recip[c],
                                                              y1_log_prob[c],
                                                              log_prob_rowwise_prod_temp,
                                                              log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                              sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                              log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                              sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                              log_common_grad_term_1,
                                                              log_abs_z_grad_term,
                                                              sign_z_grad_term,
                                                              log_abs_grad_prob,
                                                              sign_grad_prob,
                                                              log_abs_prod_container_or_inc_array,
                                                              sign_prod_container_or_inc_array,
                                                              log_abs_prod_container_or_inc_array_comp,
                                                              sign_prod_container_or_inc_array_comp,
                                                              log_sum_result,
                                                              sign_sum_result,
                                                              log_terms,
                                                              sign_terms,
                                                              container_max_logs,
                                                              container_sum_exp_signed,
                                                              Model_args_for_chunk);

                  for (int t = 0; t < n_tests; t++) {
                    for (int k = 0; k <  beta_grad_array_for_each_n.size(); k++) {
                      LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1(log_abs_beta_grad_array_for_each_n[k].col(t),  // entire array as carrying fwd the non-log-scale grad previously computed
                                                                                              sign_beta_grad_array_for_each_n[k].col(t), // entire array as carrying fwd the non-log-scale grad previously computed
                                                                                              vect_type);

                      beta_grad_array[c](k, t) += stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
                    }
                  }

                }

              }

            }

            sign_z_grad_term.setOnes();
            sign_grad_prob.setOnes();
            sign_prod_container_or_inc_array.setOnes();
            sign_sum_result.setOnes();
            sign_terms.setOnes();
            sign_a.setOnes();
            sign_b.setOnes();
            sign_prod_container_or_inc_array_comp.setOnes();

            log_abs_z_grad_term.setConstant(-700.0);
            log_abs_grad_prob.setConstant(-700.0);
            log_abs_prod_container_or_inc_array.setConstant(-700.0);
            log_sum_result.setConstant(-700.0);
            log_terms.setConstant(-700.0);
            log_abs_a.setConstant(-700.0);
            log_abs_b.setConstant(-700.0);
            container_max_logs.setConstant(-700.0);
            log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);

          }

          ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of L_Omega ('s)
          if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only") ) {

            { /// first compute gradients on the standard (non-log) scale.

              log_abs_z_grad_term.setZero();
              log_abs_grad_prob.setZero();
              log_abs_prod_container_or_inc_array.setZero();
              sign_prod_container_or_inc_array.setZero();

              fn_MVP_compute_L_Omega_grad_v2(        U_Omega_grad_array[c],
                                                     Omega_grad_array_for_each_n,
                                                     common_grad_term_1,
                                                     L_Omega_double[c],
                                                     prob[c],
                                                     prob_recip,
                                                     Bound_Z[c],
                                                     Z_std_norm[c],
                                                     prob_rowwise_prod_temp,
                                                     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                     log_abs_z_grad_term,
                                                     log_abs_grad_prob,
                                                     log_abs_prod_container_or_inc_array,
                                                     sign_prod_container_or_inc_array,
                                                     false, /// compute_final_scalar_grad
                                                     Model_args_for_chunk);

            }

            { // then compute gradients on the LOG-scale, but ONLY where we have underflow or overflow.

              //// extract previous signs and log_abs values
              for (int t1 = 0; t1 < n_tests  ; t1++ ) {

                auto temp_array = Omega_grad_array_for_each_n[t1];
                auto temp_array_sign = stan::math::sign(temp_array);
                sign_Omega_grad_array_for_each_n[t1] = temp_array_sign;
                auto temp_array_abs = stan::math::abs(temp_array);
                log_abs_Omega_grad_array_for_each_n[t1] = fn_EIGEN_double(temp_array_abs, "log", vect_type_log);

                for (int t2 = 0; t2 <  t1 + 1; t2++ ) {
                  sign_Omega_grad_array_for_each_n[t1].col(t2)(problem_index_array[c][t1]).setOnes();
                  log_abs_Omega_grad_array_for_each_n[t1].col(t2)(problem_index_array[c][t1]).setConstant(-700.0);
                }

              }

              fn_MVP_compute_L_Omega_grad_log_scale(      n_problem_array[c],
                                                          problem_index_array[c],
                                                          U_Omega_grad_array[c],
                                                          sign_Omega_grad_array_for_each_n,
                                                          log_abs_Omega_grad_array_for_each_n,
                                                          log_abs_Bound_Z[c], // not in other grads
                                                          sign_Bound_Z[c], // not in other grads
                                                          log_Z_std_norm[c], // not in other grads
                                                          sign_Z_std_norm, // not in other grads
                                                          L_Omega_double[c],
                                                          log_abs_L_Omega_double[c],
                                                          log_phi_Z_recip[c],
                                                          y1_log_prob[c],
                                                          log_prob_rowwise_prod_temp,
                                                          log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                          sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                          log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                          sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                          log_common_grad_term_1,
                                                          log_abs_z_grad_term,
                                                          sign_z_grad_term,
                                                          log_abs_grad_prob,
                                                          sign_grad_prob,
                                                          log_abs_prod_container_or_inc_array,
                                                          sign_prod_container_or_inc_array,
                                                          log_abs_prod_container_or_inc_array_comp,
                                                          sign_prod_container_or_inc_array_comp,
                                                          log_abs_derivs_chain_container_vec_comp,
                                                          sign_derivs_chain_container_vec_comp,
                                                          log_sum_result,
                                                          sign_sum_result,
                                                          log_terms,
                                                          sign_terms,
                                                          log_abs_a,
                                                          log_abs_b,
                                                          sign_a,
                                                          sign_b,
                                                          container_max_logs,
                                                          container_sum_exp_signed,
                                                          Model_args_for_chunk);

              for (int t1 = 0; t1 < n_tests  ; t1++ ) {
                for (int t2 = 0; t2 <  t1 + 1; t2++ ) {
                  LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1(log_abs_Omega_grad_array_for_each_n[t1].col(t2),  // entire array as carrying fwd the non-log-scale grad previously computed
                                                                                          sign_Omega_grad_array_for_each_n[t1].col(t2), // entire array as carrying fwd the non-log-scale grad previously computed
                                                                                          vect_type);

                  U_Omega_grad_array[c](t1, t2) +=   stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
                }
              }

              sign_z_grad_term.setOnes();
              sign_grad_prob.setOnes();
              sign_prod_container_or_inc_array.setOnes();
              sign_sum_result.setOnes();
              sign_terms.setOnes();
              sign_a.setOnes();
              sign_b.setOnes();
              sign_prod_container_or_inc_array_comp.setOnes();
              sign_derivs_chain_container_vec_comp.setOnes();

              log_abs_z_grad_term.setConstant(-700.0);
              log_abs_grad_prob.setConstant(-700.0);
              log_abs_prod_container_or_inc_array.setConstant(-700.0);
              log_sum_result.setConstant(-700.0);
              log_terms.setConstant(-700.0);
              log_abs_a.setConstant(-700.0);
              log_abs_b.setConstant(-700.0);
              container_max_logs.setConstant(-700.0);
              log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);
              log_abs_derivs_chain_container_vec_comp.setConstant(-700.0);

            }

          }

          if (n_class > 1) { /// prevelance only estimated for latent class models

            if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only" ) ) {

              fn_MVP_prev_multi_pop_accumulate_grad( prob[c],
                                                     prob_n_recip,
                                                     pop_ind,
                                                     chunk_size_orig * chunk_counter,
                                                     chunk_size,
                                                     n_pops,
                                                     c,
                                                     prev_grad_mat,
                                                     rowwise_prod);

            }

          }

        }


}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// This model ccan be either the "standard" MVP model or the latent class MVP model (w/ 2 classes) for analysis of test accuracy data.
inline  void                             fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(    Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat ,
                                                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                                            const std::string grad_option,
                                                                                                            const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                                            std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                                            const int n_threads_WCP
) {

        out_mat.setZero();
        ////
        const int N = y_ref.rows();
        const int n_tests = y_ref.cols();
        const int n_us = theta_us_vec_ref.rows()  ;
        const int n_params_main =  theta_main_vec_ref.rows()  ;
        const int n_params = n_params_main + n_us;
        ////
        //////////////  access elements from struct and read
        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>  &X =  Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0];
        ////
        const bool exclude_priors = Model_args_as_cpp_struct.Model_args_bools(0);
        const bool CI =             Model_args_as_cpp_struct.Model_args_bools(1);
        const bool corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
        const bool corr_prior_beta = Model_args_as_cpp_struct.Model_args_bools(3);
        const bool corr_prior_norm = Model_args_as_cpp_struct.Model_args_bools(4);
        const bool handle_numerical_issues = Model_args_as_cpp_struct.Model_args_bools(5);
        const bool skip_checks_exp =   Model_args_as_cpp_struct.Model_args_bools(6);
        const bool skip_checks_log =   Model_args_as_cpp_struct.Model_args_bools(7);
        const bool skip_checks_lse =   Model_args_as_cpp_struct.Model_args_bools(8);
        const bool skip_checks_tanh =  Model_args_as_cpp_struct.Model_args_bools(9);
        const bool skip_checks_Phi =  Model_args_as_cpp_struct.Model_args_bools(10);
        const bool skip_checks_log_Phi = Model_args_as_cpp_struct.Model_args_bools(11);
        const bool skip_checks_inv_Phi = Model_args_as_cpp_struct.Model_args_bools(12);
        const bool skip_checks_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_bools(13);
        const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
        ////
        const int n_cores = Model_args_as_cpp_struct.Model_args_ints(0);
        const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
        const int ub_threshold_phi_approx = Model_args_as_cpp_struct.Model_args_ints(2);
        const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
        ////
        // const double prev_prior_a = Model_args_as_cpp_struct.Model_args_doubles(0);
        // const double prev_prior_b = Model_args_as_cpp_struct.Model_args_doubles(1);
        const double overflow_threshold  = Model_args_as_cpp_struct.Model_args_doubles(0);
        const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(1);
        ////
        //// NOTE: these are non-const because they get overridden for the last chunk
        //// For WCP, each thread needs its own copies (handled below)
        ////
        std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
        const std::string &Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
        const std::string &inv_Phi_type = Model_args_as_cpp_struct.Model_args_strings(2);
        std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
        std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
        std::string vect_type_lse = Model_args_as_cpp_struct.Model_args_strings(5);
        std::string vect_type_tanh = Model_args_as_cpp_struct.Model_args_strings(6);
        std::string vect_type_Phi = Model_args_as_cpp_struct.Model_args_strings(7);
        std::string vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8);
        std::string vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);
        std::string vect_type_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_strings(10);
        const std::string J_grad_option =  Model_args_as_cpp_struct.Model_args_strings(11);
        const std::string nuisance_transformation =   Model_args_as_cpp_struct.Model_args_strings(12);
        ////
        const Eigen::Matrix<double, -1, 1> &lkj_cholesky_eta = Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
        const Eigen::Matrix<double, -1, 1> &prev_prior_a = Model_args_as_cpp_struct.Model_args_col_vecs_double[1]; //// ----
        const Eigen::Matrix<double, -1, 1> &prev_prior_b = Model_args_as_cpp_struct.Model_args_col_vecs_double[2]; //// ----
        ////
        const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
        ////
        // const Eigen::Matrix<double, -1, -1> &LT_b_priors_shape  = Model_args_as_cpp_struct.Model_args_mats_double[0];
        // const Eigen::Matrix<double, -1, -1> &LT_b_priors_scale  = Model_args_as_cpp_struct.Model_args_mats_double[1];
        // const Eigen::Matrix<double, -1, -1> &LT_known_bs_indicator = Model_args_as_cpp_struct.Model_args_mats_double[2];
        // const Eigen::Matrix<double, -1, -1> &LT_known_bs_values = Model_args_as_cpp_struct.Model_args_mats_double[3];
        ////
        std::vector<Eigen::Matrix<double, -1, -1>>   prior_coeffs_mean  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
        std::vector<Eigen::Matrix<double, -1, -1>>   prior_coeffs_sd   =  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
        const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_a   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
        const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_b   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];
        std::vector<Eigen::Matrix<double, -1, -1>>   lb_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];
        const std::vector<Eigen::Matrix<double, -1, -1>>   &ub_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
        const std::vector<Eigen::Matrix<double, -1, -1>>   &known_values    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];
        ////
        const std::vector<Eigen::Matrix<int, -1, -1 >> &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];

        // const std::vector<Eigen::Matrix<int, -1, 1 >> &perm     = Model_args_as_cpp_struct.Model_args_vecs_of_col_vecs_int[0];
        // const std::vector<Eigen::Matrix<int, -1, 1 >> &inv_perm = Model_args_as_cpp_struct.Model_args_vecs_of_col_vecs_int[1];

        //// Override lb_corr if corr_force_positive is TRUE:
        if (corr_force_positive == true) {
          for (int c = 0; c < n_class; ++c) {
            for (int i = 1; i < n_tests; ++i) {
              for (int j = 0; j < i; ++j) {
                lb_corr[c](i, j) = 0.0;
              }
            }
          }
        }

        //// ---- Other prev params:
        const int n_pops = Model_args_as_cpp_struct.Model_args_ints(6);  //// length = N ---- multi-pops
        const Eigen::Matrix<int, -1, 1> &pop_ind = Model_args_as_cpp_struct.Model_args_col_vecs_int[2];  // 0-indexed, length N //// ---- multi-pop

        //////////////
        const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;

        int n_covariates_total, n_covariates_max;
        int n_covariates_total_nd, n_covariates_total_d;
        int n_covariates_max_nd, n_covariates_max_d;

        if (n_class > 1) {

              n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
              n_covariates_total_d = n_covariates_per_outcome_vec.row(1).sum();
              n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
              n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
              n_covariates_max_d = n_covariates_per_outcome_vec.row(1).maxCoeff();
              n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);

        } else {

              n_covariates_total = n_covariates_per_outcome_vec.sum();
              n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();

        }

        const double sqrt_2_pi_recip = 1.0 / sqrt(2.0 * M_PI);
        const double sqrt_2_recip = 1.0 / stan::math::sqrt(2.0);
        const double minus_sqrt_2_recip = -sqrt_2_recip;
        const double a = 0.07056;
        const double b = 1.5976;
        const double a_times_3 = 3.0 * a;
        const double s = 1.0 / 1.702;
        const double Inf = std::numeric_limits<double>::infinity();

        //// ---- determine chunk size --------------------------
        const int desired_n_chunks = n_chunks;

        int vec_size;
        if (vect_type == "AVX512")      vec_size = 8;
        else if (vect_type == "AVX2")   vec_size = 4;
        else if (vect_type == "AVX")    vec_size = 2;
        else                            vec_size = 1;

        ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, desired_n_chunks);

        const int chunk_size_orig = chunk_size_info.chunk_size_orig;
        const int normal_chunk_size = chunk_size_info.normal_chunk_size;
        const int last_chunk_size = chunk_size_info.last_chunk_size;
        const int n_total_chunks = chunk_size_info.n_total_chunks;
        const int n_full_chunks = chunk_size_info.n_full_chunks;

        //////////////  -----------------------------------------------------------------------------------------------------------------------------------------------------------
        ////
        //// ---- Corrs (doubles):
        ////
        const Eigen::Matrix<double, -1, 1>  Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs);
        ////
        //// ---- Coeffs (doubles):
        ////
        std::vector<Eigen::Matrix<double, -1, -1 > > beta_double_array = vec_of_mats_double(n_covariates_max, n_tests,  n_class);
        {
          int i = n_corrs;
          for (int c = 0; c < n_class; ++c) {
            for (int t = 0; t < n_tests; ++t) {
              for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k) {
                beta_double_array[c](k, t) = theta_main_vec_ref(i);
                i += 1;
              }
            }
          }
        }
        // //// ---- Identification constraint (label-switch guard): D+ must have the HIGHER intercept on
        // ////      the (first) binary test. Outside this half-space the posterior is -Inf. Implemented as
        // ////      a hard rejection rather than a reparameterisation, so the priors on the two intercepts
        // ////      are exactly as specified, restricted to beta_d > beta_nd (cf. Stan: target += -Inf).
        // ////      The gradient is irrelevant at a rejected point; NaN-free zeros are fine.
        // ////
        // if (n_class > 1) {
        //       const double beta_bin_nd = beta_double_array[0](0, 0);
        //       const double beta_bin_d  = beta_double_array[1](0, 0);
        //       if (!(beta_bin_d > beta_bin_nd)) {
        //             out_mat.setZero();
        //             out_mat(0) = -std::numeric_limits<double>::infinity();
        //             // stan::math::recover_memory_nested();
        //             return;
        //       }
        // }
        ////
        //// ---- Prev (doubles):
        ////
        Eigen::Matrix<double, -1, 1> u_prev_raw(n_pops);
        if (n_class > 1) {
          for (int g = 0; g < n_pops; ++g) {
            u_prev_raw(g) = theta_main_vec_ref(n_corrs + n_covariates_total + g);
          }
        }
        ////
        //// ---- Omega:
        ////
        double prior_densities_L_Omega_double = 0.0;
        double log_det_J_L_Omega_double = 0.0;
        Eigen::Matrix<double, -1, 1> grad_Omega_raw_priors_and_log_det_J(n_corrs);

        int dim_choose_2 = n_tests * (n_tests - 1) * 0.5;
        std::vector<Eigen::Matrix<double, -1, -1>> deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_unconstrained_double = fn_convert_std_vec_of_corrs_to_3d_array_double( Eigen_vec_to_std_vec(Omega_raw_vec_double),
                                                                                                                                n_tests,
                                                                                                                                n_class);
        // ============================================================================
        // Cholesky factorisation + numeric differentiation / AD for L_Omega
        // ============================================================================
        for (int c = 0; c < n_class; ++c) {   // Pinkney_cvine_bounds_opt_dbl   Pinkney_LDL_bounds_opt_dbl

              auto out = Pinkney_corr_master_dbl(  n_tests,
                                                   lb_corr[c],
                                                   ub_corr[c],
                                                   Omega_unconstrained_double[c],
                                                   known_values_indicator[c],
                                                   known_values[c]);

              L_Omega_double[c] = out.block(1, 0, n_tests, n_tests);  // use L_perm directly
              Omega_double[c] = L_Omega_double[c] * L_Omega_double[c].transpose();

              ////
              for (int i = 1; i < n_tests; ++i) {
                for (int j = 0; j < i; ++j) {  // Only lower triangular
                  if (std::abs(L_Omega_double[c](i, j)) > 1e-10) {  // Avoid near-zero
                    L_Omega_recip_double[c](i, j) = 1.0 / L_Omega_double[c](i, j);
                  } else {
                    L_Omega_recip_double[c](i, j) = 0.0;  // or handle differently
                  }
                }
              }

        }

        //// Compute L_Omega derivatives gradient using finite differences
        if (J_grad_option == "num_diff") {

              int global_idx = 0;  // Cumulative counter for all classes

              for (int c = 0; c < n_class; ++c) {

                      //// Helper lambda: unconstrained -> un-permuted L
                      auto get_L_orig = [&](const Eigen::Matrix<double, -1, -1> &Omega_unc) {

                            auto out = Pinkney_corr_master_dbl(   n_tests,
                                                                  lb_corr[c],
                                                                  ub_corr[c],
                                                                  Omega_unc,
                                                                  known_values_indicator[c],
                                                                  known_values[c]);
                            return out.block(1, 0, n_tests, n_tests).eval();

                      };

                      // // Base
                      // L_Omega_double[c] = get_L_orig( Omega_unconstrained_double[c]);

                      // Finite differences
                      int cnt_2 = 0;
                      for (int i = 1; i < n_tests; i++) {

                            for (int j = 0; j < i; j++) {

                                  double epsilon = std::max(1e-8, 1e-6 * std::abs( Omega_unconstrained_double[c](i, j)));
                                  if (std::abs( Omega_unconstrained_double[c](i, j)) > 2.0) epsilon = 1e-4;

                                  Omega_unconstrained_double[c](i, j) += epsilon;
                                  Eigen::Matrix<double, -1, -1> L_perturbed = get_L_orig(Omega_unconstrained_double[c]);
                                  Omega_unconstrained_double[c](i, j) -= epsilon;

                                  int cnt_1 = 0;
                                  for (int k = 0; k < n_tests; k++) {
                                    for (int l = 0; l <= k; l++) {
                                      double deriv = (L_perturbed(k, l) - L_Omega_double[c](k, l)) / epsilon;
                                      deriv_L_wrt_unc_full[c](cnt_1, cnt_2) = std::isfinite(deriv) ? deriv : 0.0;
                                      cnt_1++;
                                    }
                                  }

                                  cnt_2++;
                                  global_idx++;

                            }

                      }

              }

        } else if (J_grad_option == "analytical") {

              // for (int c = 0; c < n_class; ++c) {
              //   deriv_L_wrt_unc_full[c] = compute_L_Omega_derivatives_analytical( Omega_unconstrained_double[c],
              //                                                                     L_Omega_double[c],
              //                                                                     lb_corr[c],
              //                                                                     ub_corr[c],
              //                                                                     known_values_indicator[c],
              //                                                                     known_values[c],
              //                                                                     n_tests);
              // }

        }

        std::vector<Eigen::Matrix<double, -1, -1 > > log_abs_L_Omega_double = L_Omega_double;
        std::vector<Eigen::Matrix<double, -1, -1 > > sign_L_Omega_double = L_Omega_double;

        Eigen::Matrix<double, -1, 1> grad_prev_raw(n_pops);
        double prior_densities_prev_double = 0.0;
        // double log_det_J_prev_double = 0.0; //// ---- multi-pops
        // double grad_prev_raw_priors_and_log_det_J = 0.0; //// ---- multi-pops
        double log_det_J_prev_from_AD = 0.0;

        {     ///////////   -------------------  start of AD block  ------------------------------------------------------------------------------------------------------------------

          stan::math::start_nested();  ////////////////////////
          ////
          stan::math::var target_AD = 0.0;

          // L_Omega /  correlation params  ----------------------------------------------------------------------------------------------------------------------
          Eigen::Matrix<stan::math::var, -1, 1  >  Omega_raw_vec_var =  stan::math::to_var(Omega_raw_vec_double) ;
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_raw_vec_var),
                                                                                                                                      n_tests,
                                                                                                                                      n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var   = vec_of_mats_var(n_tests, n_tests, n_class);
          // Eigen::Matrix<stan::math::var, -1, 1> log_det_J_Omega(2);

          {
              stan::math::var log_det_J_L_Omega = 0.0;

              for (int c = 0; c < n_class; ++c) { // Pinkney_LDL_bounds_opt  Pinkney_cvine_bounds_opt

                    Eigen::Matrix<stan::math::var, -1, -1  >  Chol_Schur_outs =  Pinkney_corr_master(   n_tests,
                                                                                                        lb_corr[c],
                                                                                                        ub_corr[c],
                                                                                                        Omega_unconstrained_var[c],
                                                                                                        known_values_indicator[c],
                                                                                                        known_values[c]);
                    L_Omega_var[c] = Chol_Schur_outs.block(1, 0, n_tests, n_tests);
                    target_AD         += Chol_Schur_outs(0, 0);
                    log_det_J_L_Omega += Chol_Schur_outs(0, 0);
                    Omega_var[c] = L_Omega_var[c] * L_Omega_var[c].transpose();

                    for (int i = 1; i < n_tests; ++i) {    ////   for (i in 2:n_tests) {
                      for (int j = 0; j < i; ++j) {   ////   for (j in 1:(i - 1)) {
                        if (known_values_indicator[c](i, j) == 1) {
                          stan::math::var known_val_prior_ij = stan::math::normal_lpdf( Omega_var[c](i, j), 0.0, 10.0 ); //// to ensure any corr's we aren't estimating dont cause divergences
                          target_AD += known_val_prior_ij;
                          prior_densities_L_Omega_double += known_val_prior_ij.val();   //// value now reaches lp (was grad-only)
                        }
                      }
                    }

              }

              log_det_J_L_Omega_double += log_det_J_L_Omega.val();
          }

          {
              stan::math::var prior_densities_L_Omega = 0.0;

              for (int c = 0; c < n_class; ++c) {

                    if ( (corr_prior_beta == false)   &&  (corr_prior_norm == false) ) {
                      prior_densities_L_Omega +=  stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c)); // for "Pinkney_LDL_bounds_opt"
                      // prior_densities_L_Omega += (lkj_cholesky_eta(c) - 1.0) * log_det_J_Omega(c);  // (0, 1) holds log_det_C // for "Pinkney_cvine_bounds_opt"
                    } else if ( (corr_prior_beta == true)   &&  (corr_prior_norm == false) ) {
                      for (int i = 1; i < n_tests; i++) {
                        for (int j = 0; j < i; j++) {
                          prior_densities_L_Omega +=  stan::math::beta_lpdf(  (Omega_var[c](i, j) + 1)/2, prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
                        }
                      }
                      //  Jacobian for  Omega -> L_Omega transformation for prior log-densities (since both LKJ and truncated normal prior densities are in terms of Omega, not L_Omega)
                      Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
                      for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i+1) ) * stan::math::log(L_Omega_var[c](i, i));
                      prior_densities_L_Omega  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
                    } else if  ( (corr_prior_beta == false)   &&  (corr_prior_norm == true) ) {
                      for (int i = 1; i < n_tests; i++) {
                        for (int j = 0; j < i; j++) {
                          prior_densities_L_Omega +=  stan::math::normal_lpdf(  Omega_var[c](i, j), prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
                        }
                      }
                      Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
                      for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i+1) ) * stan::math::log(L_Omega_var[c](i, i));
                      prior_densities_L_Omega  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
                    }

              }

              target_AD += prior_densities_L_Omega;
              prior_densities_L_Omega_double += prior_densities_L_Omega.val();

              ////////////////////////////////////////////////////////////
              target_AD.grad();   // differentiating this (i.e. NOT wrt this!! - this is the subject)
              grad_Omega_raw_priors_and_log_det_J = Omega_raw_vec_var.adj();    // differentiating WRT this
              out_mat.segment(1 + n_us, n_corrs) = grad_Omega_raw_priors_and_log_det_J;   //// add grad constribution to output
              stan::math::set_zero_all_adjoints_nested();
              ////////////////////////////////////////////////////////////
          }

          /////////////  prev stuff  ---- vars
          {
              if (n_class > 1) {  //// if latent class

                  fn_MVP_prev_multi_pop_AD(  u_prev_raw,
                                             prev_prior_a,
                                             prev_prior_b,
                                             n_pops,
                                             prior_densities_prev_double,
                                             log_det_J_prev_from_AD,
                                             grad_prev_raw);

                  // Write to output (n_pops positions instead of 1)
                  int prev_start = 1 + n_us + n_corrs + n_covariates_total;
                  out_mat.segment(prev_start, n_pops) = grad_prev_raw;

              }
          }
          ////////////////////////////////////////////////////////////
          if (J_grad_option == "autodiff") {
            for (int c = 0; c < n_class; ++c) {
              int cnt_1 = 0;
              for (int k = 0; k < n_tests; k++) {
                for (int l = 0; l < k + 1; l++) {
                  (  L_Omega_var[c](k, l)).grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
                  int cnt_2 = 0;
                  for (int i = 1; i < n_tests; i++) {
                    for (int j = 0; j < i; j++) {
                      deriv_L_wrt_unc_full[c](cnt_1, cnt_2)  =   Omega_unconstrained_var[c](i, j).adj();     // differentiating WRT this
                      cnt_2 += 1;
                    }
                  }
                  stan::math::set_zero_all_adjoints_nested();
                  cnt_1 += 1;
                }
              }
            }
          }

          ///////////////// get cholesky factor's (lower-triangular) of corr matrices
          // convert to 3d var array
          for (int c = 0; c < n_class; ++c) {
            for (int t1 = 0; t1 < n_tests; ++t1) {
              for (int t2 = 0; t2 < n_tests; ++t2) {
                L_Omega_double[c](t1, t2) =   L_Omega_var[c](t1, t2).val()  ;
                log_abs_L_Omega_double[c](t1, t2) =   stan::math::log(stan::math::fabs( L_Omega_double[c](t1, t2) ))  ;
                sign_L_Omega_double[c](t1, t2) = stan::math::sign( L_Omega_double[c](t1, t2) );
                L_Omega_recip_double[c](t1, t2) =   1.0 / L_Omega_double[c](t1, t2) ;
              }
            }
            ////log_abs_L_Omega_double[c] =    log_abs_L_Omega_double[c].array().min(700.0).max(-700.0);
          }

          stan::math::recover_memory_nested();  //////////////////////////////////////////

        }   //////////////////////////  end of local AD block --------------------------------------------------------------------------------------------------------------------------

        /////////////  prev stuff (multi-pop)
        Eigen::Matrix<double, -1, -1> prev_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        Eigen::Matrix<double, -1, -1> log_prev_mat_small = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        Eigen::Matrix<double, -1, 1>  tanh_u_prev_vec(n_pops);
        Eigen::Matrix<double, -1, 1>  deriv_p_wrt_u_vec(n_pops);
        double log_det_J_prev_double_total = 0.0;
        ////
        if (n_class > 1) {

            for (int g = 0; g < n_pops; ++g) {
                tanh_u_prev_vec(g) = stan::math::tanh(u_prev_raw(g));
                double prev_g = 0.5 * (tanh_u_prev_vec(g) + 1.0);
                prev_mat(g, 1) = prev_g;
                prev_mat(g, 0) = 1.0 - prev_g;
                deriv_p_wrt_u_vec(g) = 0.5 * (1.0 - tanh_u_prev_vec(g) * tanh_u_prev_vec(g));
                log_det_J_prev_double_total += stan::math::log(deriv_p_wrt_u_vec(g));
            }
            ////
            log_prev_mat_small = stan::math::log(prev_mat);

        }

        ///////////////////////////////////////////////////////////////////////// prior densities
        double prior_densities = 0.0;

        if (exclude_priors == false) {

              ///////////////////// priors for coeffs
              double prior_densities_coeffs_double = 0.0;
              for (int c = 0; c < n_class; c++) {
                for (int t = 0; t < n_tests; t++) {
                  for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) {
                    prior_densities_coeffs_double  += stan::math::normal_lpdf(beta_double_array[c](k, t), prior_coeffs_mean[c](k, t), prior_coeffs_sd[c](k, t));
                  }
                }
              }

              prior_densities += prior_densities_coeffs_double;
              prior_densities += prior_densities_L_Omega_double;
              prior_densities += prior_densities_prev_double;

        }

        ////////  ------- likelihood function  ---------------------------------------------------------------------------------------------------------------------------------
        double log_prob_out = 0.0;

        // Jacobian adjustments (none needed for coeffs as unconstrained - so only for L_Omega -> Omega and u_prev -> prev, and the one for u's is computed in the likelihood)
        const double log_det_J_main = log_det_J_prev_double_total + log_det_J_L_Omega_double;

        //// define unconstrained nuisance parameter vec
        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec = theta_us_vec_ref;

        /////////////////////////////////////////////////
        std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array =  vec_of_mats<double>(n_covariates_max, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array =  vec_of_mats<double>(n_tests, n_tests, n_class);
        Eigen::Matrix<double, -1, -1> prev_grad_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        ////
        double log_jac_u = 0.0;  //// NOTE: now at function scope — accumulates across BOTH the full-chunks block AND the last-chunk block
        /////////////////////////////////////////////////

        // ============================================================================
        // CHUNK LOOP - one shared per-chunk routine; full chunks get the SIMD struct,
        // the remainder chunk gets a "Stan" (scalar) copy (replaces the previous
        // duplicated full-chunks + last-chunk blocks).
        // ============================================================================
        for (int nc = 0; nc < n_full_chunks; nc++) {

              fn_lp_grad_MVP_LC_Pinkney_PartialLog_process_chunk(   out_mat,
                                                                    u_unc_vec,
                                                                    y_ref,
                                                                    grad_option,
                                                                    nc,
                                                                    normal_chunk_size,
                                                                    chunk_size_orig,
                                                                    N,
                                                                    n_params,
                                                                    n_tests,
                                                                    n_class,
                                                                    n_covariates_max,
                                                                    n_pops,
                                                                    overflow_threshold,
                                                                    underflow_threshold,
                                                                    n_covariates_per_outcome_vec,
                                                                    pop_ind,
                                                                    X,
                                                                    beta_double_array,
                                                                    L_Omega_double,
                                                                    L_Omega_recip_double,
                                                                    log_abs_L_Omega_double,
                                                                    prev_mat,
                                                                    log_prev_mat_small,
                                                                    log_jac_u,
                                                                    beta_grad_array,
                                                                    U_Omega_grad_array,
                                                                    prev_grad_mat,
                                                                    Model_args_as_cpp_struct);

        }

        // ---- LAST CHUNK (remainder) - scalar ("Stan") fallback end-to-end: ----
        if ((n_full_chunks < n_total_chunks) && (last_chunk_size > 0)) {

              Model_fn_args_struct Model_args_last_chunk = Model_args_as_cpp_struct;
              Model_args_last_chunk.Model_args_strings(0)  = "Stan";  // vect_type
              for (int s_i = 3; s_i <= 10; ++s_i)  Model_args_last_chunk.Model_args_strings(s_i) = "Stan";

              fn_lp_grad_MVP_LC_Pinkney_PartialLog_process_chunk(   out_mat,
                                                                    u_unc_vec,
                                                                    y_ref,
                                                                    grad_option,
                                                                    n_full_chunks,
                                                                    last_chunk_size,
                                                                    chunk_size_orig,
                                                                    N,
                                                                    n_params,
                                                                    n_tests,
                                                                    n_class,
                                                                    n_covariates_max,
                                                                    n_pops,
                                                                    overflow_threshold,
                                                                    underflow_threshold,
                                                                    n_covariates_per_outcome_vec,
                                                                    pop_ind,
                                                                    X,
                                                                    beta_double_array,
                                                                    L_Omega_double,
                                                                    L_Omega_recip_double,
                                                                    log_abs_L_Omega_double,
                                                                    prev_mat,
                                                                    log_prev_mat_small,
                                                                    log_jac_u,
                                                                    beta_grad_array,
                                                                    U_Omega_grad_array,
                                                                    prev_grad_mat,
                                                                    Model_args_last_chunk);

        } // end of last chunk loop

        // ============================================================================
        // Post-loop: prevalence gradient, log_prob, output assembly
        // ============================================================================
        Eigen::Matrix<double, -1, 1> prev_unconstrained_grad_vec_out = Eigen::Matrix<double, -1, 1>::Zero(n_pops);

        //////////////////////// gradients for latent class membership probabilitie(s) (i.e. disease prevalence)
        if (n_class > 1) {
          for (int g = 0; g < n_pops; ++g) {
            double lik_grad = (prev_grad_mat(g, 1) - prev_grad_mat(g, 0)) * deriv_p_wrt_u_vec(g);
            double jac_grad = -2.0 * tanh_u_prev_vec(g);
            prev_unconstrained_grad_vec_out(g) = lik_grad + jac_grad;
          }
        }

        ////////////////////////  --------------------------------------------------------------------------
        log_prob_out += out_mat.tail(N).sum(); // log_prob_out += log_lik.sum();
        log_prob_out += log_jac_u;
        if (exclude_priors == false) log_prob_out += prior_densities;
        log_prob_out += log_det_J_main;

        //// Pack beta gradients into vector
        Eigen::Matrix<double, -1, 1> beta_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);
        {
          int i = 0;
          for (int c = 0; c < n_class; c++) {
            for (int t = 0; t < n_tests; t++) {
              for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) {
                beta_grad_vec(i) = beta_grad_array[c](k, t);
                i += 1;
              }
            }
          }
        }

        //// Pack L_Omega gradients and chain-rule through deriv_L_wrt_unc_full
        Eigen::Matrix<double, -1, 1> L_Omega_grad_vec(n_corrs + (n_class * n_tests));
        Eigen::Matrix<double, -1, 1> U_Omega_grad_vec(n_corrs);
        {
          int i = 0;
          for (int c = 0; c < n_class; c++ ) {
            for (int t1 = 0; t1 < n_tests  ; t1++ ) {
              for (int t2 = 0; t2 <  t1 + 1; t2++ ) {
                L_Omega_grad_vec(i) = U_Omega_grad_array[c](t1,t2);
                i += 1;
              }
            }
          }
        }

        Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_nd(dim_choose_2 + n_tests);
        Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_d(dim_choose_2 + n_tests);

        if (n_class > 1) {
              grad_wrt_L_Omega_nd =   L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
              grad_wrt_L_Omega_d =   L_Omega_grad_vec.segment(dim_choose_2 + n_tests, dim_choose_2 + n_tests);
              U_Omega_grad_vec.head(dim_choose_2) =  ( grad_wrt_L_Omega_nd.transpose()  *  deriv_L_wrt_unc_full[0].cast<double>() ).transpose();
              U_Omega_grad_vec.segment(dim_choose_2, dim_choose_2) =   ( grad_wrt_L_Omega_d.transpose()  *  deriv_L_wrt_unc_full[1].cast<double>() ).transpose();
        } else {
              grad_wrt_L_Omega_nd =   L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
              U_Omega_grad_vec.head(dim_choose_2) =  ( grad_wrt_L_Omega_nd.transpose()  *  deriv_L_wrt_unc_full[0].cast<double>() ).transpose() ;
        }

        //// ---- Final output assembly ----
        out_mat(0) =  log_prob_out;
        out_mat.segment(1 + n_us, n_corrs).array() += U_Omega_grad_vec.array();
        out_mat.segment(1 + n_us + n_corrs, n_covariates_total) = beta_grad_vec;
        ////
        if (n_class > 1) { //// ----
          const int prev_start_final = 1 + n_us + n_corrs + n_covariates_total;
          out_mat.segment(prev_start_final, n_pops) += prev_unconstrained_grad_vec_out;
        }

        //// ---- Prior gradient contribution to coefficients ----
        {
          int i = n_us + n_corrs + 1;
          for (int c = 0; c < n_class; c++) {
            for (int t = 0; t < n_tests; t++) {
              for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) {
                if (exclude_priors == false) {
                  out_mat(i) += -((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t)) * (1.0 / prior_coeffs_sd[c](k, t));
                  i += 1;
                }
              }
            }
          }
        }

}



//// MVP_lp_grad_log_scale_MD_AD_fns_T.hpp  — drop-in replacement for MVP_lp_grad_log_scale_MD_AD_fns.hpp
////
//// Keeps the SAME entry-point name and signature that your InPlace wrappers call:
////   fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(...)
//// so the three fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(...) overloads and
//// fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD(...) at the bottom of your file stay exactly as they are.
//// Delete your old ..._process_chunk and ..._InPlace_process, keep the wrappers, include this file.
////
//// Requires: MVP_log_scale_grad_calc_fns_T.hpp (which pulls in MVP_helpers_migrated.hpp and
//// fn_dispatch_templated.hpp), your fn_MVP_compute_lp_GHK_cols<vec>, and the unchanged
//// fn_MVP_compute_nuisance_grad_v2 / fn_MVP_compute_coefficients_grad_v2 / fn_MVP_compute_L_Omega_grad_v2.

// #pragma once
// #include <Eigen/Dense>
// #include <unsupported/Eigen/SpecialFunctions>
// #include "MVP_log_scale_grad_calc_fns_T.hpp"





// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// template <Vec vec>
// inline void fn_lp_grad_MVP_LC_Pinkney_PartialLog_process_chunk_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
//                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec,
//                                                                    const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
//                                                                    const std::string &grad_option,
//                                                                    const int chunk_counter,
//                                                                    const int chunk_size,
//                                                                    const int chunk_size_orig,
//                                                                    const int N,
//                                                                    const int n_params,
//                                                                    const int n_tests,
//                                                                    const int n_class,
//                                                                    const int n_covariates_max,
//                                                                    const int n_pops,
//                                                                    const double overflow_threshold,
//                                                                    const double underflow_threshold,
//                                                                    const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec,
//                                                                    const Eigen::Matrix<int, -1, 1> &pop_ind,
//                                                                    const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X,
//                                                                    const std::vector<Eigen::Matrix<double, -1, -1>> &beta_double_array,
//                                                                    const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_double,
//                                                                    const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_recip_double,
//                                                                    const std::vector<Eigen::Matrix<double, -1, -1>> &log_abs_L_Omega_double,
//                                                                    const Eigen::Matrix<double, -1, -1> &prev_mat,
//                                                                    const Eigen::Matrix<double, -1, -1> &log_prev_mat_small,
//                                                                    double &log_jac_u,
//                                                                    std::vector<Eigen::Matrix<double, -1, -1>> &beta_grad_array,
//                                                                    std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
//                                                                    Eigen::Matrix<double, -1, -1> &prev_grad_mat,
//                                                                    const KernelChoice &kchoice,
//                                                                    const Model_fn_args_struct &Model_args_as_cpp_struct   //// only for the unchanged v2 grad fns (n_class etc.)
// ) {
//         typedef Eigen::Matrix<double, -1, -1> M;
//         typedef Eigen::Matrix<double, -1, 1>  V;
//         const int row_start = chunk_size_orig * chunk_counter;
// 
//         //// ---- per-chunk containers ----
//         std::vector<M> Z_std_norm = vec_of_mats<double>(chunk_size, n_tests, n_class);
//         std::vector<M> Bound_Z = Z_std_norm, Bound_U_Phi_Bound_Z = Z_std_norm, Phi_Z = Z_std_norm, phi_Bound_Z = Z_std_norm,
//                        y1_log_prob = Z_std_norm, prob = Z_std_norm, phi_Z_recip = Z_std_norm;
//         std::vector<M> log_phi_Z_recip = Z_std_norm, log_phi_Bound_Z = Z_std_norm, log_Z_std_norm = Z_std_norm,
//                        log_abs_Bound_Z = Z_std_norm, sign_Bound_Z = Z_std_norm;
// 
//         M y_chunk = M::Zero(chunk_size, n_tests), u_array = M::Zero(chunk_size, n_tests),
//           y_sign_chunk = M::Zero(chunk_size, n_tests), y_m_y_sign_x_u = M::Zero(chunk_size, n_tests);
//         V inc_array = V::Zero(chunk_size);
//         M u_grad_array_CM_chunk = M::Zero(chunk_size, n_tests);
//         M log_abs_u_grad_array_CM_chunk = M::Constant(chunk_size, n_tests, -700.0);
// 
//         V log_sum_result = V::Constant(chunk_size, -700.0), sign_result = V::Ones(chunk_size),
//           container_max_logs = V::Constant(chunk_size, -700.0), container_sum_exp_signed = V::Zero(chunk_size);
//         V u_unc_vec_chunk = V::Zero(chunk_size * n_tests), u_vec_chunk = V::Zero(chunk_size * n_tests),
//           du_wrt_duu_chunk = V::Zero(chunk_size * n_tests), d_J_wrt_duu_chunk = V::Zero(chunk_size * n_tests);
// 
//         M log_common_grad_term_1 = M::Constant(chunk_size, n_tests, -700.0);
//         M log_abs_ys = M::Constant(chunk_size, n_tests, -700.0), log_abs_ym = M::Constant(chunk_size, n_tests, -700.0);
//         M sign_ys = M::Ones(chunk_size, n_tests), sign_ym = M::Ones(chunk_size, n_tests);
//         M log_prob_rowwise_prod_temp = M::Constant(chunk_size, n_tests, -700.0), log_prob_recip_rowwise_prod_temp = M::Constant(chunk_size, n_tests, -700.0);
//         M log_abs_grad_prob = M::Constant(chunk_size, n_tests, -700.0), log_abs_z_grad_term = M::Constant(chunk_size, n_tests, -700.0);
//         M sign_grad_prob = M::Ones(chunk_size, n_tests), sign_z_grad_term = M::Ones(chunk_size, n_tests);
//         M log_abs_prod_comp = M::Constant(chunk_size, n_tests, -700.0), sign_prod_comp = M::Ones(chunk_size, n_tests);
//         M log_abs_dcc = M::Constant(chunk_size, n_tests, -700.0), sign_dcc = M::Ones(chunk_size, n_tests);
//         V log_abs_prod = V::Constant(chunk_size, -700.0), sign_prod = V::Ones(chunk_size);
//         V log_prob_rowwise_prod_temp_all = V::Constant(chunk_size, -700.0);
//         V log_abs_a = V::Constant(chunk_size, -700.0), sign_a = V::Ones(chunk_size), log_abs_b = V::Constant(chunk_size, -700.0), sign_b = V::Ones(chunk_size);
//         V sign_sum_result = V::Ones(chunk_size);
//         M log_terms = M::Constant(chunk_size, n_tests, -700.0), sign_terms = M::Ones(chunk_size, n_tests);
// 
//         Eigen::VectorXi overflow_mask(chunk_size), underflow_mask(chunk_size), OK_mask(chunk_size);
//         std::vector<std::vector<std::vector<int>>> problem_index_array(n_class);
//         std::vector<std::vector<int>> n_problem_array(n_class);
//         for (int c = 0; c < n_class; c++) { problem_index_array[c].resize(n_tests); n_problem_array[c].assign(n_tests, 0); }
// 
//         std::vector<M> Omega_grad_array_for_each_n = vec_of_mats<double>(chunk_size, n_tests, n_tests);
//         std::vector<M> sign_Omega_grad_array_for_each_n = Omega_grad_array_for_each_n, log_abs_Omega_grad_array_for_each_n = Omega_grad_array_for_each_n;
//         std::vector<M> beta_grad_array_for_each_n = vec_of_mats<double>(chunk_size, n_tests, n_covariates_max);
//         std::vector<M> sign_beta_grad_array_for_each_n = beta_grad_array_for_each_n, log_abs_beta_grad_array_for_each_n = beta_grad_array_for_each_n;
// 
//         V rowwise_prod = V::Zero(chunk_size), rowwise_sum = V::Zero(chunk_size);
//         V log_prev_per_obs_given_c = V::Zero(chunk_size), prev_per_obs_given_c = V::Zero(chunk_size);
//         M lp_array = M::Zero(chunk_size, n_class);
// 
//         //// ---- nuisance ----
//         y_chunk = y_ref.middleRows(row_start, chunk_size).cast<double>();
//         u_unc_vec_chunk = u_unc_vec.segment(row_start * n_tests, chunk_size * n_tests);
//         fn_MVP_compute_nuisance_T<vec>(u_vec_chunk, u_unc_vec_chunk, kchoice);
//         log_jac_u += fn_MVP_compute_nuisance_log_jac_u_T<vec>(u_vec_chunk, u_unc_vec_chunk, d_J_wrt_duu_chunk /*scratch*/, kchoice);
//         u_array = u_vec_chunk.reshaped(chunk_size, n_tests);
//         y_sign_chunk.array()   = 2.0 * y_chunk.array() - 1.0;
//         y_m_y_sign_x_u.array() = y_chunk.array() - (y_sign_chunk.array() * u_array.array());
// 
//         //// ---- likelihood ----
//         for (int c = 0; c < n_class; c++) {
// 
//           inc_array.setZero();
//           if (n_class > 1) {
//             for (int n = 0; n < chunk_size; ++n) {
//               const int g = pop_ind(row_start + n);
//               log_prev_per_obs_given_c(n) = log_prev_mat_small(g, c);
//               prev_per_obs_given_c(n)     = prev_mat(g, c);
//             }
//           }
// 
//           for (int t = 0; t < n_tests; t++) {
// 
//             if (n_covariates_max > 1) {
//               const V Xbeta = X[c][t].block(row_start, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>()
//                               * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
//               Bound_Z[c].col(t) = L_Omega_recip_double[c](t, t) * (-1.0 * (Xbeta + inc_array));
//             } else {
//               Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0 * (beta_double_array[c](0, t) + inc_array.array()));
//             }
//             sign_Bound_Z[c].col(t)    = Bound_Z[c].col(t).array().sign();
//             log_abs_Bound_Z[c].col(t) = Bound_Z[c].col(t).cwiseAbs();
//             apply_col_inplace<vec, Fn::log>(log_abs_Bound_Z[c], t);
// 
//             //// masks
//             overflow_mask.array()  = ((Bound_Z[c].col(t).array() > overflow_threshold)  && (y_chunk.col(t).array() == 1.0)).template cast<int>();
//             underflow_mask.array() = ((Bound_Z[c].col(t).array() < underflow_threshold) && (y_chunk.col(t).array() == 0.0)).template cast<int>();
//             OK_mask.array() = ((overflow_mask.array() == 0) && (underflow_mask.array() == 0)).template cast<int>();
//             const int n_overflows = overflow_mask.sum(), n_underflows = underflow_mask.sum();
//             const int n_problem = n_overflows + n_underflows, n_OK = OK_mask.sum();
// 
//             std::vector<int> over_index; over_index.reserve(n_overflows);
//             std::vector<int> under_index; under_index.reserve(n_underflows);
//             for (int n = 0; n < chunk_size; ++n) {
//               if (overflow_mask(n) == 1) over_index.push_back(n);
//               else if (underflow_mask(n) == 1) under_index.push_back(n);
//             }
//             std::vector<int> problem_index = over_index;
//             problem_index.insert(problem_index.end(), under_index.begin(), under_index.end());
//             n_problem_array[c][t] = n_problem;
//             problem_index_array[c][t] = problem_index;
// 
//             //// standard-scale GHK
//             fn_MVP_compute_lp_GHK_cols_T<vec>(t, Bound_U_Phi_Bound_Z[c], Phi_Z[c], Z_std_norm[c], prob[c], y1_log_prob[c],
//                                             Bound_Z[c], y_chunk, u_array, kchoice);
//             fn_MVP_compute_phi_Z_recip_cols_T<vec>(t, phi_Z_recip[c], Phi_Z[c], Z_std_norm[c], kchoice);
//             fn_MVP_compute_phi_Bound_Z_cols_T<vec>(t, phi_Bound_Z[c], Bound_U_Phi_Bound_Z[c], Bound_Z[c], kchoice);
// 
//             //// log-scale quantities (grad)
//             log_phi_Bound_Z[c].col(t) = phi_Bound_Z[c].col(t).cwiseAbs();  apply_col_inplace<vec, Fn::log>(log_phi_Bound_Z[c], t);
//             log_phi_Z_recip[c].col(t) = phi_Z_recip[c].col(t).cwiseAbs();  apply_col_inplace<vec, Fn::log>(log_phi_Z_recip[c], t);
//             log_Z_std_norm[c].col(t)  = Z_std_norm[c].col(t).cwiseAbs();   apply_col_inplace<vec, Fn::log>(log_Z_std_norm[c], t);
// 
//             if (n_OK < chunk_size) {
//               if (n_underflows > 0) {
//                 fn_MVP_compute_lp_GHK_cols_log_scale_underflow_T<vec>(t, under_index, Bound_U_Phi_Bound_Z[c], Phi_Z[c], Z_std_norm[c], log_Z_std_norm[c],
//                                                                       prob[c], y1_log_prob[c], log_phi_Bound_Z[c], log_phi_Z_recip[c], Bound_Z[c], u_array);
//                 V tmp = log_phi_Bound_Z[c](under_index, t); apply_inplace<vec, Fn::exp>(tmp); phi_Bound_Z[c](under_index, t) = tmp;
//                 tmp   = log_phi_Z_recip[c](under_index, t); apply_inplace<vec, Fn::exp>(tmp); phi_Z_recip[c](under_index, t) = tmp;
//               }
//               if (n_overflows > 0) {
//                 fn_MVP_compute_lp_GHK_cols_log_scale_overflow_T<vec>(t, n_overflows, over_index, Bound_U_Phi_Bound_Z[c], Phi_Z[c], Z_std_norm[c], log_Z_std_norm[c],
//                                                                      prob[c], y1_log_prob[c], log_phi_Bound_Z[c], log_phi_Z_recip[c], Bound_Z[c], u_array);
//                 V tmp = log_phi_Bound_Z[c](over_index, t); apply_inplace<vec, Fn::exp>(tmp); phi_Bound_Z[c](over_index, t) = tmp;
//                 tmp   = log_phi_Z_recip[c](over_index, t); apply_inplace<vec, Fn::exp>(tmp); phi_Z_recip[c](over_index, t) = tmp;
//               }
//             }
// 
//             if (t < n_tests - 1) {
//               inc_array = Z_std_norm[c].leftCols(t + 1) * L_Omega_double[c].row(t + 1).head(t + 1).transpose();
//             }
//           }
// 
//           rowwise_sum = y1_log_prob[c].rowwise().sum();
//           if (n_class > 1) { rowwise_sum.array() += log_prev_per_obs_given_c.array(); lp_array.col(c) = rowwise_sum; }
//           else             { lp_array.col(0) = rowwise_sum; }
//         }
// 
//         const int index_start = 1 + n_params + row_start;
//         if (n_class > 1) {
//           log_sum_exp_general_T<vec>(lp_array, log_sum_result, container_max_logs);
//           out_mat.segment(index_start, chunk_size) = log_sum_result;
//         } else {
//           out_mat.segment(index_start, chunk_size) = lp_array.col(0);
//         }
// 
//         V log_prob_n_recip(chunk_size), prob_n_recip(chunk_size);
//         {
//           V prob_n = out_mat.segment(index_start, chunk_size);
//           apply_inplace<vec, Fn::exp>(prob_n);
//           prob_n_recip = stan::math::inv(prob_n);
//           log_prob_n_recip = -out_mat.segment(index_start, chunk_size);      //// exact log(1/prob_n)
//         }
// 
//         //// ---- gradients ----
//         for (int c = 0; c < n_class; c++) {
// 
//           if (n_class > 1) {
//             for (int n = 0; n < chunk_size; ++n) log_prev_per_obs_given_c(n) = log_prev_mat_small(pop_ind(row_start + n), c);
//           }
//           for (auto &m : beta_grad_array_for_each_n)          m.setZero();
//           for (auto &m : sign_beta_grad_array_for_each_n)     m.setOnes();
//           for (auto &m : log_abs_beta_grad_array_for_each_n)  m.setConstant(-700.0);
//           for (auto &m : Omega_grad_array_for_each_n)         m.setZero();
//           for (auto &m : sign_Omega_grad_array_for_each_n)    m.setOnes();
//           for (auto &m : log_abs_Omega_grad_array_for_each_n) m.setConstant(-700.0);
// 
//           const M y1_log_prob_recip = -y1_log_prob[c];
//           const M sign_Z_std_norm   = Z_std_norm[c].array().sign();
//           const M prob_recip        = stan::math::inv(prob[c]);
//           M common_grad_term_1 = M::Zero(chunk_size, n_tests), prob_rowwise_prod_temp = M::Zero(chunk_size, n_tests);
//           M ys = M::Zero(chunk_size, n_tests), ym = M::Zero(chunk_size, n_tests);   //// y_sign_chunk_times_... / y_m_ysign_x_u_array_times_...
// 
//           if (grad_option != "none") {
//             M log_abs_L_Omega_recip_double = M::Zero(n_tests, n_tests);
//             M sign_L_Omega_recip_double = L_Omega_recip_double[c].array().sign();
//             for (int t = 0; t < n_tests; t++) log_abs_L_Omega_recip_double(t, t) = std::log(std::abs(L_Omega_recip_double[c](t, t)));
// 
//             fn_MVP_grad_prep_log_scale_T<vec>(log_prob_rowwise_prod_temp, log_prob_recip_rowwise_prod_temp, log_prob_rowwise_prod_temp_all,
//                                               log_common_grad_term_1, log_abs_ys, log_abs_ym, sign_ys, sign_ym,
//                                               y1_log_prob[c], y1_log_prob_recip, log_prob_n_recip, log_prev_per_obs_given_c,
//                                               log_phi_Bound_Z[c], log_phi_Z_recip[c], log_abs_L_Omega_recip_double, sign_L_Omega_recip_double,
//                                               y_sign_chunk, y_m_y_sign_x_u, n_class);
// 
//             common_grad_term_1 = log_common_grad_term_1;       apply_inplace<vec, Fn::exp>(common_grad_term_1);
//             prob_rowwise_prod_temp = log_prob_rowwise_prod_temp; apply_inplace<vec, Fn::exp>(prob_rowwise_prod_temp);
//             ys = log_abs_ys; apply_inplace<vec, Fn::exp>(ys); ys.array() *= sign_ys.array();
//             ym = log_abs_ym; apply_inplace<vec, Fn::exp>(ym); ym.array() *= sign_ym.array();
//           }
// 
//           auto reset_log_scale_scratch = [&]() {
//             sign_z_grad_term.setOnes(); sign_grad_prob.setOnes(); sign_prod.setOnes(); sign_sum_result.setOnes(); sign_terms.setOnes();
//             sign_a.setOnes(); sign_b.setOnes(); sign_prod_comp.setOnes(); sign_dcc.setOnes();
//             log_abs_z_grad_term.setConstant(-700.0); log_abs_grad_prob.setConstant(-700.0); log_abs_prod.setConstant(-700.0);
//             log_sum_result.setConstant(-700.0); log_terms.setConstant(-700.0); log_abs_a.setConstant(-700.0); log_abs_b.setConstant(-700.0);
//             container_max_logs.setConstant(-700.0); log_abs_prod_comp.setConstant(-700.0); log_abs_dcc.setConstant(-700.0);
//           };
// 
//           //// ---- nuisance ----
//           if ((grad_option == "us_only") || (grad_option == "all")) {
//             M u_grad_array_CM_chunk_block = u_grad_array_CM_chunk;
//             fn_MVP_compute_nuisance_grad_v2(u_grad_array_CM_chunk_block, phi_Z_recip[c], common_grad_term_1, L_Omega_double[c], prob[c], prob_recip,
//                                             prob_rowwise_prod_temp, ys, ym, log_abs_z_grad_term, log_abs_grad_prob, log_abs_prod, sign_prod, Model_args_as_cpp_struct);
//             fn_MVP_compute_nuisance_grad_log_scale_T<vec>(n_problem_array[c], problem_index_array[c], log_abs_u_grad_array_CM_chunk, u_grad_array_CM_chunk_block,
//                                                           L_Omega_double[c], log_abs_L_Omega_double[c], log_phi_Z_recip[c], y1_log_prob[c], y1_log_prob_recip,
//                                                           log_prob_rowwise_prod_temp, log_abs_ys, sign_ys, log_abs_ym, sign_ym, log_common_grad_term_1,
//                                                           log_abs_z_grad_term, sign_z_grad_term, log_abs_grad_prob, sign_grad_prob, log_abs_prod, sign_prod,
//                                                           log_sum_result, sign_sum_result, log_terms, sign_terms, log_abs_a, log_abs_b, sign_a, sign_b,
//                                                           container_max_logs, container_sum_exp_signed);
//             u_grad_array_CM_chunk.array() += u_grad_array_CM_chunk_block.array();
// 
//             if (c == n_class - 1) {
//               const int start_index = 1 + row_start * n_tests;
//               const int length = chunk_size * n_tests;
//               out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();
//               fn_MVP_nuisance_first_deriv_T<vec>(du_wrt_duu_chunk, u_vec_chunk, u_unc_vec_chunk, kchoice);
//               fn_MVP_nuisance_deriv_of_log_det_J_T<vec>(d_J_wrt_duu_chunk, u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, kchoice);
//               out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
//               out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();
//             }
//             reset_log_scale_scratch();
//           }
// 
//           //// ---- coefficients ----
//           if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only")) {
//             Eigen::Matrix<int, -1, 1> n_cov_vec_c = n_covariates_per_outcome_vec.row(c).transpose();
//             if (n_covariates_max > 1) {
//               fn_MVP_compute_coefficients_grad_v2(c, beta_grad_array[c], X[c], n_cov_vec_c, beta_grad_array_for_each_n, chunk_counter, n_covariates_max,
//                                                   common_grad_term_1, L_Omega_double[c], prob[c], prob_recip, prob_rowwise_prod_temp, ys, ym,
//                                                   log_abs_grad_prob, sign_grad_prob, log_abs_prod, sign_prod, true, Model_args_as_cpp_struct);
//             } else {
//               log_abs_grad_prob.setZero(); sign_grad_prob.setZero(); log_abs_prod.setZero(); sign_prod.setZero();
//               fn_MVP_compute_coefficients_grad_v2(c, beta_grad_array[c], X[c], n_cov_vec_c, beta_grad_array_for_each_n, chunk_counter, n_covariates_max,
//                                                   common_grad_term_1, L_Omega_double[c], prob[c], prob_recip, prob_rowwise_prod_temp, ys, ym,
//                                                   log_abs_grad_prob, sign_grad_prob, log_abs_prod, sign_prod, false, Model_args_as_cpp_struct);
// 
//               for (size_t i = 0; i < beta_grad_array_for_each_n.size(); i++) {
//                 sign_beta_grad_array_for_each_n[i]    = beta_grad_array_for_each_n[i].array().sign();
//                 log_abs_beta_grad_array_for_each_n[i] = beta_grad_array_for_each_n[i].cwiseAbs();
//                 apply_inplace<vec, Fn::log>(log_abs_beta_grad_array_for_each_n[i]);
//                 for (int t = 0; t < n_tests; t++) {
//                   sign_beta_grad_array_for_each_n[i].col(t)(problem_index_array[c][t]).setOnes();
//                   log_abs_beta_grad_array_for_each_n[i].col(t)(problem_index_array[c][t]).setConstant(-700.0);
//                 }
//               }
//               fn_MVP_compute_coefficients_grad_log_scale_T<vec>(n_problem_array[c], problem_index_array[c], beta_grad_array[c],
//                                                                 sign_beta_grad_array_for_each_n, log_abs_beta_grad_array_for_each_n,
//                                                                 L_Omega_double[c], log_abs_L_Omega_double[c], log_phi_Z_recip[c], y1_log_prob[c],
//                                                                 log_prob_rowwise_prod_temp, log_abs_ys, sign_ys, log_abs_ym, sign_ym, log_common_grad_term_1,
//                                                                 log_abs_z_grad_term, sign_z_grad_term, log_abs_grad_prob, sign_grad_prob, log_abs_prod, sign_prod,
//                                                                 log_abs_prod_comp, sign_prod_comp, log_sum_result, sign_sum_result, log_terms, sign_terms,
//                                                                 container_max_logs, container_sum_exp_signed);
//               for (int t = 0; t < n_tests; t++) {
//                 for (size_t k = 0; k < beta_grad_array_for_each_n.size(); k++) {
//                   const LogSumSignedResult_T r = log_sum_vec_signed_T<vec>(log_abs_beta_grad_array_for_each_n[k].col(t), sign_beta_grad_array_for_each_n[k].col(t));
//                   beta_grad_array[c](k, t) += std::exp(r.log_sum) * r.sign;
//                 }
//               }
//             }
//             reset_log_scale_scratch();
//           }
// 
//           //// ---- L_Omega ----
//           if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only")) {
//             log_abs_z_grad_term.setZero(); log_abs_grad_prob.setZero(); log_abs_prod.setZero(); sign_prod.setZero();
//             fn_MVP_compute_L_Omega_grad_v2(U_Omega_grad_array[c], Omega_grad_array_for_each_n, common_grad_term_1, L_Omega_double[c], prob[c], prob_recip,
//                                            Bound_Z[c], Z_std_norm[c], prob_rowwise_prod_temp, ys, ym,
//                                            log_abs_z_grad_term, log_abs_grad_prob, log_abs_prod, sign_prod, false, Model_args_as_cpp_struct);
// 
//             for (int t1 = 0; t1 < n_tests; t1++) {
//               sign_Omega_grad_array_for_each_n[t1]    = Omega_grad_array_for_each_n[t1].array().sign();
//               log_abs_Omega_grad_array_for_each_n[t1] = Omega_grad_array_for_each_n[t1].cwiseAbs();
//               apply_inplace<vec, Fn::log>(log_abs_Omega_grad_array_for_each_n[t1]);
//               for (int t2 = 0; t2 < t1 + 1; t2++) {
//                 sign_Omega_grad_array_for_each_n[t1].col(t2)(problem_index_array[c][t1]).setOnes();
//                 log_abs_Omega_grad_array_for_each_n[t1].col(t2)(problem_index_array[c][t1]).setConstant(-700.0);
//               }
//             }
//             fn_MVP_compute_L_Omega_grad_log_scale_T<vec>(n_problem_array[c], problem_index_array[c], U_Omega_grad_array[c],
//                                                          sign_Omega_grad_array_for_each_n, log_abs_Omega_grad_array_for_each_n,
//                                                          log_abs_Bound_Z[c], sign_Bound_Z[c], log_Z_std_norm[c], sign_Z_std_norm,
//                                                          L_Omega_double[c], log_abs_L_Omega_double[c], log_phi_Z_recip[c], y1_log_prob[c],
//                                                          log_prob_rowwise_prod_temp, log_abs_ys, sign_ys, log_abs_ym, sign_ym, log_common_grad_term_1,
//                                                          log_abs_z_grad_term, sign_z_grad_term, log_abs_grad_prob, sign_grad_prob, log_abs_prod, sign_prod,
//                                                          log_abs_prod_comp, sign_prod_comp, log_abs_dcc, sign_dcc, log_sum_result, sign_sum_result,
//                                                          log_terms, sign_terms, log_abs_a, log_abs_b, sign_a, sign_b, container_max_logs, container_sum_exp_signed);
//             for (int t1 = 0; t1 < n_tests; t1++) {
//               for (int t2 = 0; t2 < t1 + 1; t2++) {
//                 const LogSumSignedResult_T r = log_sum_vec_signed_T<vec>(log_abs_Omega_grad_array_for_each_n[t1].col(t2), sign_Omega_grad_array_for_each_n[t1].col(t2));
//                 U_Omega_grad_array[c](t1, t2) += std::exp(r.log_sum) * r.sign;
//               }
//             }
//             reset_log_scale_scratch();
//           }
// 
//           //// ---- prevalence ----
//           if ((n_class > 1) && ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only"))) {
//             fn_MVP_prev_multi_pop_accumulate_grad(prob[c], prob_n_recip, pop_ind, row_start, chunk_size, n_pops, c, prev_grad_mat, rowwise_prod);
//           }
//         }
// }
// 
// 
// 
// 
// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// template <Vec vec>
// inline void fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
//                                                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
//                                                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
//                                                                                 const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
//                                                                                 const std::string &grad_option,
//                                                                                 const Model_fn_args_struct &Model_args_as_cpp_struct
// ) {
//         out_mat.setZero();
//         const int N = y_ref.rows();
//         const int n_tests = y_ref.cols();
//         const int n_us = theta_us_vec_ref.rows();
//         const int n_params_main = theta_main_vec_ref.rows();
//         const int n_params = n_params_main + n_us;
// 
//         const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X = Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0];
//         const bool exclude_priors      = Model_args_as_cpp_struct.Model_args_bools(0);
//         const bool corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
//         const bool corr_prior_beta     = Model_args_as_cpp_struct.Model_args_bools(3);
//         const bool corr_prior_norm     = Model_args_as_cpp_struct.Model_args_bools(4);
//         const int n_class  = Model_args_as_cpp_struct.Model_args_ints(1);
//         const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
//         const double overflow_threshold  = Model_args_as_cpp_struct.Model_args_doubles(0);
//         const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(1);
//         const std::string &vect_type     = Model_args_as_cpp_struct.Model_args_strings(0);
//         const std::string &J_grad_option = Model_args_as_cpp_struct.Model_args_strings(11);
//         const KernelChoice kchoice = kernel_choice_from_args(Model_args_as_cpp_struct);
// 
//         const Eigen::Matrix<double, -1, 1> &lkj_cholesky_eta = Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
//         const Eigen::Matrix<double, -1, 1> &prev_prior_a     = Model_args_as_cpp_struct.Model_args_col_vecs_double[1];
//         const Eigen::Matrix<double, -1, 1> &prev_prior_b     = Model_args_as_cpp_struct.Model_args_col_vecs_double[2];
//         const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
//         const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_mean = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
//         const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_sd   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
//         const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_a  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
//         const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_b  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];
//         std::vector<Eigen::Matrix<double, -1, -1>>        lb_corr           = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];
//         const std::vector<Eigen::Matrix<double, -1, -1>> &ub_corr           = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
//         const std::vector<Eigen::Matrix<double, -1, -1>> &known_values      = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];
//         const std::vector<Eigen::Matrix<int, -1, -1>>    &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];
// 
//         if (corr_force_positive) {
//           for (int c = 0; c < n_class; ++c) for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j) lb_corr[c](i, j) = 0.0;
//         }
// 
//         const int n_pops = Model_args_as_cpp_struct.Model_args_ints(6);
//         const Eigen::Matrix<int, -1, 1> &pop_ind = Model_args_as_cpp_struct.Model_args_col_vecs_int[2];
//         const int n_corrs = n_class * n_tests * (n_tests - 1) / 2;
// 
//         int n_covariates_total, n_covariates_max;
//         if (n_class > 1) {
//           n_covariates_total = n_covariates_per_outcome_vec.row(0).sum() + n_covariates_per_outcome_vec.row(1).sum();
//           n_covariates_max = std::max(n_covariates_per_outcome_vec.row(0).maxCoeff(), n_covariates_per_outcome_vec.row(1).maxCoeff());
//         } else {
//           n_covariates_total = n_covariates_per_outcome_vec.sum();
//           n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
//         }
// 
//         int vec_size;
//         if (vect_type == "AVX512")      vec_size = 8;
//         else if (vect_type == "AVX2")   vec_size = 4;
//         else if (vect_type == "AVX")    vec_size = 2;
//         else                            vec_size = 1;
//         ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, n_chunks);
//         const int chunk_size_orig   = chunk_size_info.chunk_size_orig;
//         const int normal_chunk_size = chunk_size_info.normal_chunk_size;
//         const int last_chunk_size   = chunk_size_info.last_chunk_size;
//         const int n_total_chunks    = chunk_size_info.n_total_chunks;
//         const int n_full_chunks     = chunk_size_info.n_full_chunks;
// 
//         //// ---- corrs / coeffs / prev unpack ----
//         const Eigen::Matrix<double, -1, 1> Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs);
//         std::vector<Eigen::Matrix<double, -1, -1>> beta_double_array = vec_of_mats_double(n_covariates_max, n_tests, n_class);
//         {
//           int i = n_corrs;
//           for (int c = 0; c < n_class; ++c) for (int t = 0; t < n_tests; ++t) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k)
//             beta_double_array[c](k, t) = theta_main_vec_ref(i++);
//         }
//         Eigen::Matrix<double, -1, 1> u_prev_raw(n_pops);
//         if (n_class > 1) for (int g = 0; g < n_pops; ++g) u_prev_raw(g) = theta_main_vec_ref(n_corrs + n_covariates_total + g);
// 
//         //// ---- Omega ----
//         double prior_densities_L_Omega_double = 0.0, log_det_J_L_Omega_double = 0.0;
//         Eigen::Matrix<double, -1, 1> grad_Omega_raw_priors_and_log_det_J(n_corrs);
//         const int dim_choose_2 = n_tests * (n_tests - 1) / 2;
//         std::vector<Eigen::Matrix<double, -1, -1>> deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double       = vec_of_mats_double(n_tests, n_tests, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double = vec_of_mats_double(n_tests, n_tests, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> Omega_double         = vec_of_mats_double(n_tests, n_tests, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> Omega_unconstrained_double =
//             fn_convert_std_vec_of_corrs_to_3d_array_double(Eigen_vec_to_std_vec(Omega_raw_vec_double), n_tests, n_class);
// 
//         for (int c = 0; c < n_class; ++c) {
//           auto out = Pinkney_corr_master_dbl(n_tests, lb_corr[c], ub_corr[c], Omega_unconstrained_double[c], known_values_indicator[c], known_values[c]);
//           L_Omega_double[c] = out.block(1, 0, n_tests, n_tests);
//           Omega_double[c] = L_Omega_double[c] * L_Omega_double[c].transpose();
//           for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j)
//             L_Omega_recip_double[c](i, j) = (std::abs(L_Omega_double[c](i, j)) > 1e-10) ? 1.0 / L_Omega_double[c](i, j) : 0.0;
//         }
// 
//         if (J_grad_option == "num_diff") {
//           for (int c = 0; c < n_class; ++c) {
//             auto get_L_orig = [&](const Eigen::Matrix<double, -1, -1> &Omega_unc) {
//               auto out = Pinkney_corr_master_dbl(n_tests, lb_corr[c], ub_corr[c], Omega_unc, known_values_indicator[c], known_values[c]);
//               return out.block(1, 0, n_tests, n_tests).eval();
//             };
//             int cnt_2 = 0;
//             for (int i = 1; i < n_tests; i++) {
//               for (int j = 0; j < i; j++) {
//                 double epsilon = std::max(1e-8, 1e-6 * std::abs(Omega_unconstrained_double[c](i, j)));
//                 if (std::abs(Omega_unconstrained_double[c](i, j)) > 2.0) epsilon = 1e-4;
//                 Omega_unconstrained_double[c](i, j) += epsilon;
//                 Eigen::Matrix<double, -1, -1> L_perturbed = get_L_orig(Omega_unconstrained_double[c]);
//                 Omega_unconstrained_double[c](i, j) -= epsilon;
//                 int cnt_1 = 0;
//                 for (int k = 0; k < n_tests; k++) for (int l = 0; l <= k; l++) {
//                   const double deriv = (L_perturbed(k, l) - L_Omega_double[c](k, l)) / epsilon;
//                   deriv_L_wrt_unc_full[c](cnt_1++, cnt_2) = std::isfinite(deriv) ? deriv : 0.0;
//                 }
//                 cnt_2++;
//               }
//             }
//           }
//         }
// 
//         std::vector<Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double = L_Omega_double;
//         Eigen::Matrix<double, -1, 1> grad_prev_raw(n_pops);
//         double prior_densities_prev_double = 0.0, log_det_J_prev_from_AD = 0.0;
// 
//         {   //// ---- AD block ----
//           stan::math::start_nested();
//           stan::math::var target_AD = 0.0;
//           Eigen::Matrix<stan::math::var, -1, 1> Omega_raw_vec_var = stan::math::to_var(Omega_raw_vec_double);
//           std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Omega_unconstrained_var =
//               fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_raw_vec_var), n_tests, n_class);
//           std::vector<Eigen::Matrix<stan::math::var, -1, -1>> L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
//           std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Omega_var   = vec_of_mats_var(n_tests, n_tests, n_class);
//           {
//             stan::math::var log_det_J_L_Omega = 0.0;
//             for (int c = 0; c < n_class; ++c) {
//               Eigen::Matrix<stan::math::var, -1, -1> Chol_Schur_outs = Pinkney_corr_master(n_tests, lb_corr[c], ub_corr[c], Omega_unconstrained_var[c], known_values_indicator[c], known_values[c]);
//               L_Omega_var[c] = Chol_Schur_outs.block(1, 0, n_tests, n_tests);
//               target_AD += Chol_Schur_outs(0, 0);
//               log_det_J_L_Omega += Chol_Schur_outs(0, 0);
//               Omega_var[c] = L_Omega_var[c] * L_Omega_var[c].transpose();
//               for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j) {
//                 if (known_values_indicator[c](i, j) == 1) {
//                   stan::math::var kv = stan::math::normal_lpdf(Omega_var[c](i, j), 0.0, 10.0);
//                   target_AD += kv;
//                   prior_densities_L_Omega_double += kv.val();
//                 }
//               }
//             }
//             log_det_J_L_Omega_double += log_det_J_L_Omega.val();
//           }
//           {
//             stan::math::var prior_densities_L_Omega = 0.0;
//             for (int c = 0; c < n_class; ++c) {
//               if (!corr_prior_beta && !corr_prior_norm) {
//                 prior_densities_L_Omega += stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c));
//               } else if (corr_prior_beta && !corr_prior_norm) {
//                 for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++)
//                   prior_densities_L_Omega += stan::math::beta_lpdf((Omega_var[c](i, j) + 1) / 2, prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
//                 Eigen::Matrix<stan::math::var, -1, 1> jd(n_tests);
//                 for (int i = 0; i < n_tests; ++i) jd(i) = (n_tests + 1 - (i + 1)) * stan::math::log(L_Omega_var[c](i, i));
//                 prior_densities_L_Omega += (n_tests * stan::math::log(2) + jd.sum());
//               } else if (!corr_prior_beta && corr_prior_norm) {
//                 for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++)
//                   prior_densities_L_Omega += stan::math::normal_lpdf(Omega_var[c](i, j), prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
//                 Eigen::Matrix<stan::math::var, -1, 1> jd(n_tests);
//                 for (int i = 0; i < n_tests; ++i) jd(i) = (n_tests + 1 - (i + 1)) * stan::math::log(L_Omega_var[c](i, i));
//                 prior_densities_L_Omega += (n_tests * stan::math::log(2) + jd.sum());
//               }
//             }
//             target_AD += prior_densities_L_Omega;
//             prior_densities_L_Omega_double += prior_densities_L_Omega.val();
//             target_AD.grad();
//             grad_Omega_raw_priors_and_log_det_J = Omega_raw_vec_var.adj();
//             out_mat.segment(1 + n_us, n_corrs) = grad_Omega_raw_priors_and_log_det_J;
//             stan::math::set_zero_all_adjoints_nested();
//           }
//           if (n_class > 1) {
//             fn_MVP_prev_multi_pop_AD(u_prev_raw, prev_prior_a, prev_prior_b, n_pops, prior_densities_prev_double, log_det_J_prev_from_AD, grad_prev_raw);
//             out_mat.segment(1 + n_us + n_corrs + n_covariates_total, n_pops) = grad_prev_raw;
//           }
//           if (J_grad_option == "autodiff") {
//             for (int c = 0; c < n_class; ++c) {
//               int cnt_1 = 0;
//               for (int k = 0; k < n_tests; k++) for (int l = 0; l < k + 1; l++) {
//                 (L_Omega_var[c](k, l)).grad();
//                 int cnt_2 = 0;
//                 for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++) deriv_L_wrt_unc_full[c](cnt_1, cnt_2++) = Omega_unconstrained_var[c](i, j).adj();
//                 stan::math::set_zero_all_adjoints_nested();
//                 cnt_1++;
//               }
//             }
//           }
//           for (int c = 0; c < n_class; ++c) for (int t1 = 0; t1 < n_tests; ++t1) for (int t2 = 0; t2 < n_tests; ++t2) {
//             L_Omega_double[c](t1, t2)         = L_Omega_var[c](t1, t2).val();
//             log_abs_L_Omega_double[c](t1, t2) = std::log(std::fabs(L_Omega_double[c](t1, t2)));
//             L_Omega_recip_double[c](t1, t2)   = 1.0 / L_Omega_double[c](t1, t2);
//           }
//           stan::math::recover_memory_nested();
//         }
// 
//         //// ---- prev ----
//         Eigen::Matrix<double, -1, -1> prev_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
//         Eigen::Matrix<double, -1, -1> log_prev_mat_small = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
//         Eigen::Matrix<double, -1, 1>  tanh_u_prev_vec(n_pops), deriv_p_wrt_u_vec(n_pops);
//         double log_det_J_prev_double_total = 0.0;
//         if (n_class > 1) {
//           for (int g = 0; g < n_pops; ++g) {
//             tanh_u_prev_vec(g) = stan::math::tanh(u_prev_raw(g));
//             const double prev_g = 0.5 * (tanh_u_prev_vec(g) + 1.0);
//             prev_mat(g, 1) = prev_g; prev_mat(g, 0) = 1.0 - prev_g;
//             deriv_p_wrt_u_vec(g) = 0.5 * (1.0 - tanh_u_prev_vec(g) * tanh_u_prev_vec(g));
//             log_det_J_prev_double_total += stan::math::log(deriv_p_wrt_u_vec(g));
//           }
//           log_prev_mat_small = stan::math::log(prev_mat);
//         }
// 
//         //// ---- priors ----
//         double prior_densities = 0.0;
//         if (!exclude_priors) {
//           for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++)
//             prior_densities += stan::math::normal_lpdf(beta_double_array[c](k, t), prior_coeffs_mean[c](k, t), prior_coeffs_sd[c](k, t));
//           prior_densities += prior_densities_L_Omega_double + prior_densities_prev_double;
//         }
// 
//         //// ---- likelihood: chunk loop (remainder is just the last iteration) ----
//         double log_prob_out = 0.0;
//         const double log_det_J_main = log_det_J_prev_double_total + log_det_J_L_Omega_double;
//         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec = theta_us_vec_ref;
//         std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array    = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(n_tests, n_tests, n_class);
//         Eigen::Matrix<double, -1, -1> prev_grad_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
//         double log_jac_u = 0.0;
// 
//         for (int nc = 0; nc < n_total_chunks; nc++) {
//           const int chunk_size = (nc == n_full_chunks) ? last_chunk_size : normal_chunk_size;
//           fn_lp_grad_MVP_LC_Pinkney_PartialLog_process_chunk_T<vec>(out_mat, u_unc_vec, y_ref, grad_option, nc, chunk_size, chunk_size_orig,
//                                                                     N, n_params, n_tests, n_class, n_covariates_max, n_pops,
//                                                                     overflow_threshold, underflow_threshold, n_covariates_per_outcome_vec, pop_ind, X,
//                                                                     beta_double_array, L_Omega_double, L_Omega_recip_double, log_abs_L_Omega_double,
//                                                                     prev_mat, log_prev_mat_small, log_jac_u, beta_grad_array, U_Omega_grad_array, prev_grad_mat,
//                                                                     kchoice, Model_args_as_cpp_struct);
//         }
// 
//         //// ---- post-loop ----
//         Eigen::Matrix<double, -1, 1> prev_unconstrained_grad_vec_out = Eigen::Matrix<double, -1, 1>::Zero(n_pops);
//         if (n_class > 1) {
//           for (int g = 0; g < n_pops; ++g)
//             prev_unconstrained_grad_vec_out(g) = (prev_grad_mat(g, 1) - prev_grad_mat(g, 0)) * deriv_p_wrt_u_vec(g) - 2.0 * tanh_u_prev_vec(g);
//         }
//         log_prob_out += out_mat.tail(N).sum() + log_jac_u + log_det_J_main;
//         if (!exclude_priors) log_prob_out += prior_densities;
// 
//         Eigen::Matrix<double, -1, 1> beta_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);
//         { int i = 0; for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) beta_grad_vec(i++) = beta_grad_array[c](k, t); }
// 
//         Eigen::Matrix<double, -1, 1> L_Omega_grad_vec(n_corrs + (n_class * n_tests));
//         Eigen::Matrix<double, -1, 1> U_Omega_grad_vec(n_corrs);
//         { int i = 0; for (int c = 0; c < n_class; c++) for (int t1 = 0; t1 < n_tests; t1++) for (int t2 = 0; t2 < t1 + 1; t2++) L_Omega_grad_vec(i++) = U_Omega_grad_array[c](t1, t2); }
//         if (n_class > 1) {
//           const Eigen::Matrix<double, -1, 1> g_nd = L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
//           const Eigen::Matrix<double, -1, 1> g_d  = L_Omega_grad_vec.segment(dim_choose_2 + n_tests, dim_choose_2 + n_tests);
//           U_Omega_grad_vec.head(dim_choose_2)                  = (g_nd.transpose() * deriv_L_wrt_unc_full[0]).transpose();
//           U_Omega_grad_vec.segment(dim_choose_2, dim_choose_2) = (g_d.transpose()  * deriv_L_wrt_unc_full[1]).transpose();
//         } else {
//           const Eigen::Matrix<double, -1, 1> g_nd = L_Omega_grad_vec.head(dim_choose_2 + n_tests);
//           U_Omega_grad_vec.head(dim_choose_2) = (g_nd.transpose() * deriv_L_wrt_unc_full[0]).transpose();
//         }
// 
//         out_mat(0) = log_prob_out;
//         out_mat.segment(1 + n_us, n_corrs).array() += U_Omega_grad_vec.array();
//         out_mat.segment(1 + n_us + n_corrs, n_covariates_total) = beta_grad_vec;
//         if (n_class > 1) out_mat.segment(1 + n_us + n_corrs + n_covariates_total, n_pops) += prev_unconstrained_grad_vec_out;
// 
//         if (!exclude_priors) {
//           int i = n_us + n_corrs + 1;
//           for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++)
//             out_mat(i++) += -((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t)) * (1.0 / prior_coeffs_sd[c](k, t));
//         }
// }
// 
// 
// 
// 
// //// -------------------------------------------------------------------------------------
// //// Entry point with the ORIGINAL name/signature: your InPlace wrappers call this unchanged.
// //// -------------------------------------------------------------------------------------
// inline void fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(   Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
//                                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
//                                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
//                                                                               const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
//                                                                               const std::string grad_option,
//                                                                               const Model_fn_args_struct &Model_args_as_cpp_struct,
//                                                                               std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
//                                                                               const int n_threads_WCP
// ) {
//   
//         (void)LC_MVP_ws_structs; (void)n_threads_WCP;
//         
//         const Vec vec = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));
//         
//         if (vec == Vec::AVX512) {
//           
//               fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process_T<Vec::AVX512>( out_mat, 
//                                                                                              theta_main_vec_ref, 
//                                                                                              theta_us_vec_ref, 
//                                                                                              y_ref, grad_option, 
//                                                                                              Model_args_as_cpp_struct);
//           
//         } else {
//           
//               fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process_T<Vec::Scalar>( out_mat, 
//                                                                                              theta_main_vec_ref, 
//                                                                                              theta_us_vec_ref, 
//                                                                                              y_ref, 
//                                                                                              grad_option, 
//                                                                                              Model_args_as_cpp_struct);
//         }
// }
// 
// 
// 
// 







// Internal function using Eigen::Ref as inputs for matrices
inline void     fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Matrix<double, -1, 1> &&out_mat_R_val,
                                                                   const Eigen::Matrix<double, -1, 1> &&theta_main_vec_R_val,
                                                                   const Eigen::Matrix<double, -1, 1> &&theta_us_vec_R_val,
                                                                   const Eigen::Matrix<int, -1, -1> &&y_R_val,
                                                                   const std::string &grad_option,
                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                   std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                   const int n_threads_WCP
) {
  
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat_ref(out_mat_R_val);
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref(theta_main_vec_R_val);  // create Eigen::Ref from R-value
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref(theta_us_vec_R_val);  // create Eigen::Ref from R-value
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref(y_R_val);  // create Eigen::Ref from R-value
      
      
      fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process( out_mat_ref,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      Model_args_as_cpp_struct, 
                                                                      LC_MVP_ws_structs,
                                                                      n_threads_WCP);
      
}




// Internal function using Eigen::Ref as inputs for matrices
inline void     fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Matrix<double, -1, 1> &out_mat_ref,
                                                                   const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                                   const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                                   const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                   const std::string &grad_option,
                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                   std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                   const int n_threads_WCP
) {
  
      fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process( out_mat_ref,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      Model_args_as_cpp_struct,
                                                                      LC_MVP_ws_structs,
                                                                      n_threads_WCP);
  
}




// Internal function using Eigen::Ref as inputs for matrices
template <typename MatrixType>
inline void     fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  &out_mat_ref,
                                                                   const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_main_vec_ref,
                                                                   const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_us_vec_ref,
                                                                   const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                   const std::string &grad_option,
                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                   std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                   const int n_threads_WCP
) {
      
      fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process( out_mat_ref,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      Model_args_as_cpp_struct, 
                                                                      LC_MVP_ws_structs,
                                                                      n_threads_WCP);
  
}




// Internal function using Eigen::Ref as inputs for matrices
inline Eigen::Matrix<double, -1, 1>    fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD(  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                 const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                 const std::string &grad_option,
                                                                                 const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                 std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                 const int n_threads_WCP
) {
  
      int n_params_main = theta_main_vec_ref.rows();
      int n_us = theta_us_vec_ref.rows();
      int n_params = n_us + n_params_main;
      int N = y_ref.rows();
      
      Eigen::Matrix<double, -1, 1> out_mat = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
      
      fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace( out_mat,
                                                              theta_main_vec_ref,
                                                              theta_us_vec_ref,
                                                              y_ref,
                                                              grad_option,
                                                              Model_args_as_cpp_struct, 
                                                              LC_MVP_ws_structs,
                                                              n_threads_WCP);
      
      return out_mat;
  
}












