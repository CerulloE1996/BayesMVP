#pragma once

struct LC_MVP_workspace_struct {
  
  // Constructors
  LC_MVP_workspace_struct() = default;
  
  LC_MVP_workspace_struct(int chunk_size, 
                          int n_tests, 
                          int n_class,
                          int n_covariates_max,
                          int n_corrs, 
                          int n_covariates_total) {
    
    allocate(chunk_size, 
             n_tests, 
             n_class, 
             n_covariates_max, 
             n_corrs, 
             n_covariates_total);
    
  } 
  
  void reset() {
    
    // // First restore sizes if they were shrunk by last-chunk resize
    if (is_allocated && y1_log_prob.rows() != allocated_chunk_size) {
      
      allocate(allocated_chunk_size, 
               stored_n_tests, 
               stored_n_class,
               stored_n_covariates_max, 
               stored_n_corrs, 
               stored_n_covariates_total);
       
      return;  // allocate already zeros everything
      
    }
    
  }
  
  void reset_sizes() {
    if (is_allocated && y1_log_prob.rows() != allocated_chunk_size) {
      restore_sizes();
    } 
  }
  
  //// Always size 2 — 1-class model just uses [0]
  ///////////////////////////////////////////////
  std::vector<Eigen::Matrix<double, -1, -1>> Z_std_norm;
  std::vector<Eigen::Matrix<double, -1, -1>> Bound_Z;
  std::vector<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z;
  std::vector<Eigen::Matrix<double, -1, -1>> prob;
  std::vector<Eigen::Matrix<double, -1, -1>> Phi_Z;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> y1_log_prob;
  Eigen::Matrix<double, -1, -1> phi_Z_recip;
  Eigen::Matrix<double, -1, -1> phi_Bound_Z;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> common_grad_term_1;
  Eigen::Matrix<double, -1, -1> y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip;
  Eigen::Matrix<double, -1, -1> y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip;
  Eigen::Matrix<double, -1, -1> prob_rowwise_prod_temp;
  Eigen::Matrix<double, -1, -1> prob_recip_rowwise_prod_temp;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> prod_container_or_inc_array;
  Eigen::Matrix<double, -1, 1> derivs_chain_container_vec;
  Eigen::Matrix<double, -1, 1> prob_rowwise_prod_temp_all;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> dphi_direct_Cj; // ordinal-only
  Eigen::Matrix<double, -1, 1> dZ_direct_Cj; // ordinal-only
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> grad_prob;
  Eigen::Matrix<double, -1, -1> z_grad_term;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> y_chunk;
  Eigen::Matrix<double, -1, -1> u_array;
  Eigen::Matrix<double, -1, -1> y_sign;
  Eigen::Matrix<double, -1, -1> y_m_y_sign_x_u;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk_block;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> u_unc_vec_chunk;
  Eigen::Matrix<double, -1, 1> u_vec_chunk;
  Eigen::Matrix<double, -1, 1> du_wrt_duu_chunk;
  Eigen::Matrix<double, -1, 1> d_J_wrt_duu_chunk;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> lp_array;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> prob_n;
  Eigen::Matrix<double, -1, 1> prob_n_recip;
  Eigen::Matrix<double, -1, 1> log_sum_result;
  Eigen::Matrix<double, -1, 1> container_max_logs;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> rowwise_log_sum;
  Eigen::Matrix<double, -1, 1> rowwise_prod;
  Eigen::Matrix<double, -1, 1> rowwise_sum;
  Eigen::Matrix<double, -1, 1> log_lik_chunk;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> prob_recip;
  ///////////////////////////////////////////////
  std::vector<Eigen::Matrix<double, -1, -1>> Upper_Bound_Z;        // for ordinal (MVOP) - (chunk_size × n_tests)
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> phi_Upper_Bound_Z;    // for ordinal (MVOP) - (chunk_size × n_tests)
  Eigen::Matrix<double, -1, -1> dphi_over_L;          // for ordinal (MVOP) - (chunk_size × n_tests)
  Eigen::Matrix<double, -1, -1> dZ_dmu_neg;           // for ordinal (MVOP) - (chunk_size × n_tests)
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> dphi_times_bz;        // for ordinal (MVOP) - (chunk_size × n_tests)
  Eigen::Matrix<double, -1, -1> dZ_times_bz;          // for ordinal (MVOP) - (chunk_size × n_tests)
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> prev_per_obs_given_c;
  Eigen::Matrix<double, -1, 1> log_prev_per_obs_given_c;
  ///////////////////////////////////////////////
  
  // ---- tracking ----
  bool is_allocated = false;
  int allocated_chunk_size = 0;
  
  int stored_n_tests = 0;
  int stored_n_class = 0;
  int stored_n_covariates_max = 0;
  int stored_n_corrs = 0;
  int stored_n_covariates_total = 0;
  
  void allocate(int chunk_size,
                int n_tests, 
                int n_class, 
                int n_covariates_max,
                int n_corrs, 
                int n_covariates_total) {
    
    stored_n_tests = n_tests;
    stored_n_class = n_class;
    stored_n_covariates_max = n_covariates_max;
    stored_n_corrs = n_corrs;
    stored_n_covariates_total = n_covariates_total;
    
    const int dim_choose_2 = n_tests * (n_tests - 1) / 2;
    
    ////////////////////////////////////////////////
    Z_std_norm =          vec_of_mats<double>(chunk_size, n_tests, 2);
    Bound_Z =             vec_of_mats<double>(chunk_size, n_tests, 2);
    Bound_U_Phi_Bound_Z = vec_of_mats<double>(chunk_size, n_tests, 2);
    prob =                vec_of_mats<double>(chunk_size, n_tests, 2);
    Phi_Z =               vec_of_mats<double>(chunk_size, n_tests, 2);
    ////////////////////////////////////////////////
    y1_log_prob =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    phi_Z_recip =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    phi_Bound_Z =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    u_grad_array_CM_chunk =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    common_grad_term_1 =     Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    prob_rowwise_prod_temp =         Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    prob_recip_rowwise_prod_temp =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    prod_container_or_inc_array =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    derivs_chain_container_vec =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    prob_rowwise_prod_temp_all =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ////////////////////////////////////////////////
    dphi_direct_Cj = Eigen::Matrix<double, -1, 1>::Zero(chunk_size); // ordinal-only
    dZ_direct_Cj =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size); // ordinal-only
    ////////////////////////////////////////////////
    grad_prob =              Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    z_grad_term =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    y_chunk =                Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    u_array =                Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    y_sign =                 Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    y_m_y_sign_x_u =         Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    u_grad_array_CM_chunk_block = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    u_unc_vec_chunk =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
    u_vec_chunk =        Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
    du_wrt_duu_chunk =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
    d_J_wrt_duu_chunk =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
    ////////////////////////////////////////////////
    lp_array =               Eigen::Matrix<double, -1, -1>::Zero(chunk_size, 2);
    ////////////////////////////////////////////////
    prob_n =                       Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    prob_n_recip =                 Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    log_sum_result =               Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    container_max_logs =           Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ////////////////////////////////////////////////
    rowwise_log_sum =              Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    rowwise_prod =                 Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    rowwise_sum =                  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    log_lik_chunk =                Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ////////////////////////////////////////////////
    prob_recip =             Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    Upper_Bound_Z     = vec_of_mats<double>(chunk_size, n_tests, 2); // ordinal-only
    ////////////////////////////////////////////////
    phi_Upper_Bound_Z = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
    dphi_over_L       = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
    dZ_dmu_neg        = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
    //////////////////////////////////////////////// 
    dphi_times_bz     = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
    dZ_times_bz       = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
    //////////////////////////////////////////////// 
    log_prev_per_obs_given_c  = Eigen::Matrix<double, -1, 1>::Zero(chunk_size); 
    prev_per_obs_given_c      = Eigen::Matrix<double, -1, 1>::Zero(chunk_size); 
    //////////////////////////////////////////////// 
    
    is_allocated = true;
    allocated_chunk_size = chunk_size;
    
  } 
  
  void restore_sizes() {
    
    ////////////////////////////////////////////////
    for (int c = 0; c < 2; c++) { 
      Z_std_norm[c].resize(allocated_chunk_size, stored_n_tests);
      Bound_Z[c].resize(allocated_chunk_size, stored_n_tests);
      Bound_U_Phi_Bound_Z[c].resize(allocated_chunk_size, stored_n_tests);
      prob[c].resize(allocated_chunk_size, stored_n_tests);
      Phi_Z[c].resize(allocated_chunk_size, stored_n_tests);
    }
    ////////////////////////////////////////////////
    y1_log_prob.resize(allocated_chunk_size, stored_n_tests);
    phi_Z_recip.resize(allocated_chunk_size, stored_n_tests);
    phi_Bound_Z.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    u_grad_array_CM_chunk.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    common_grad_term_1.resize(allocated_chunk_size, stored_n_tests);
    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(allocated_chunk_size, stored_n_tests);
    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(allocated_chunk_size, stored_n_tests);
    prob_rowwise_prod_temp.resize(allocated_chunk_size, stored_n_tests);
    prob_recip_rowwise_prod_temp.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    prod_container_or_inc_array.resize(allocated_chunk_size);
    derivs_chain_container_vec.resize(allocated_chunk_size);
    prob_rowwise_prod_temp_all.resize(allocated_chunk_size);
    ////////////////////////////////////////////////
    dphi_direct_Cj.resize(allocated_chunk_size); // ordinal-only
    dZ_direct_Cj.resize(allocated_chunk_size); // ordinal-only
    ////////////////////////////////////////////////
    grad_prob.resize(allocated_chunk_size, stored_n_tests);
    z_grad_term.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    y_chunk.resize(allocated_chunk_size, stored_n_tests);
    u_array.resize(allocated_chunk_size, stored_n_tests);
    y_sign.resize(allocated_chunk_size, stored_n_tests);
    y_m_y_sign_x_u.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    u_grad_array_CM_chunk_block.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    u_unc_vec_chunk.resize(allocated_chunk_size * stored_n_tests);
    u_vec_chunk.resize(allocated_chunk_size * stored_n_tests);
    du_wrt_duu_chunk.resize(allocated_chunk_size * stored_n_tests);
    d_J_wrt_duu_chunk.resize(allocated_chunk_size * stored_n_tests);
    ////////////////////////////////////////////////
    lp_array.resize(allocated_chunk_size, 2);
    ////////////////////////////////////////////////
    prob_n.resize(allocated_chunk_size);
    prob_n_recip.resize(allocated_chunk_size);
    log_sum_result.resize(allocated_chunk_size);
    container_max_logs.resize(allocated_chunk_size);
    ////////////////////////////////////////////////
    rowwise_log_sum.resize(allocated_chunk_size);
    rowwise_prod.resize(allocated_chunk_size);
    rowwise_sum.resize(allocated_chunk_size);
    log_lik_chunk.resize(allocated_chunk_size);
    ////////////////////////////////////////////////
    prob_recip.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    for (int c = 0; c < 2; c++) {
      Upper_Bound_Z[c].resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    }
    ////////////////////////////////////////////////
    phi_Upper_Bound_Z.resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    dphi_over_L.resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    dZ_dmu_neg.resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    ////////////////////////////////////////////////
    dphi_times_bz.resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    dZ_times_bz.resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    //////////////////////////////////////////////// 
    log_prev_per_obs_given_c.resize(allocated_chunk_size);
    prev_per_obs_given_c.resize(allocated_chunk_size);
    //////////////////////////////////////////////
    
  }
  
};






























  
