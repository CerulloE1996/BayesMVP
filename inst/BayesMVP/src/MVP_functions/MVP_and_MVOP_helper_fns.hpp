
//// MVP_and_MVOP_helper_fns.hpp

#pragma once




#include <Eigen/Dense>
 
#include <unsupported/Eigen/SpecialFunctions>

#pragma once
 
 
 // ============================================================================
 // Multi-population prevalence helpers for LC-MVP / LC-MVOP
 //
 // Replaces the single-prevalence code with n_pops independent prevalences.
 // Each population g has its own unconstrained parameter u_raw[g], transformed
 // to prev[g] = 0.5 * (tanh(u_raw[g]) + 1) ∈ (0, 1).
 //
 // n_class is always 2 (non-diseased / diseased).
 // n_pops >= 1 (number of populations, e.g. 3 for Afrikaans/Xhosa/Zulu).
 //f
 // Parameter layout in theta_main_vec:
 //   [corrs | coeffs | u_prev_raw[0..n_pops-1] | cutpoints]
 //   i.e. n_pops params starting at index (n_corrs + n_covariates_total)
 //
 // Data required: pop_ind (length N, 0-indexed) mapping each observation
 //   to its population.
 // ============================================================================
 
 
// ============================================================================
// 1. Forward transform (doubles): u_raw → prev, log_prev, Jacobian
// ============================================================================
//
// Inputs:
//   u_prev_raw:   (n_pops × 1) unconstrained prevalence params
//   pop_ind:      (N × 1) 0-indexed population index per observation
//   n_pops, N:    dimensions
//
// Outputs (pre-allocated by caller):
//   prev:         (n_pops × 2) matrix. Col 0 = 1-prev[g], Col 1 = prev[g]
//   log_prev_mat: (N × 2) matrix. log_prev_mat(n, c) = log(prev(pop[n], c))
//   tanh_u:       (n_pops × 1) cached tanh values
//   deriv_p_wrt_u:(n_pops × 1) dp/du = 0.5*(1 - tanh²(u))
//   log_det_J:    scalar log|det(J)| = Σ_g log(dp[g]/du[g])
//
ALWAYS_INLINE void fn_MVP_prev_multi_pop_transform_double( const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_prev_raw,
                                                           const Eigen::Ref<const Eigen::Matrix<int, -1, 1>> pop_ind,
                                                           const int n_pops,
                                                           const int N,
                                                           // outputs:
                                                           Eigen::Matrix<double, -1, -1> &prev,           // n_pops × 2
                                                           Eigen::Matrix<double, -1, -1> &log_prev_mat,   // N × 2
                                                           Eigen::Matrix<double, -1, 1>  &tanh_u,         // n_pops
                                                           Eigen::Matrix<double, -1, 1>  &deriv_p_wrt_u,  // n_pops
                                                           double &log_det_J
) {
  
       log_det_J = 0.0;
       
       for (int g = 0; g < n_pops; ++g) {
             
             tanh_u(g) = stan::math::tanh(u_prev_raw(g));
             double prev_g = 0.5 * (tanh_u(g) + 1.0);
             
             prev(g, 1) = prev_g;           // P(diseased) for population g
             prev(g, 0) = 1.0 - prev_g;     // P(non-diseased) for population g
             
             deriv_p_wrt_u(g) = 0.5 * (1.0 - tanh_u(g) * tanh_u(g));
             log_det_J += stan::math::log(deriv_p_wrt_u(g));
         
       }
       
       Eigen::Matrix<double, -1, -1> log_prev = stan::math::log(prev);
       
       // Build per-observation log_prev matrix
       for (int n = 0; n < N; ++n) {
         
             int g = pop_ind(n);  // 0-indexed
             log_prev_mat(n, 0) = log_prev(g, 0);   // log(1 - prev[g])
             log_prev_mat(n, 1) = log_prev(g, 1);   // log(prev[g])
         
       }

}
 
 
// ============================================================================
// 2. AD block: priors + Jacobian for prevalence parameters
// ============================================================================
//
// Computes grad of [Beta priors + log_det_J] w.r.t. u_prev_raw
// via Stan autodiff. 
//
// This replaces the existing AD block that handles a single prevalence.
//
// Inputs:
//   u_prev_raw:    (n_pops × 1) unconstrained params (doubles, will be converted to var)
//   prior_prev_a:   (n_pops × 1) Beta prior alpha per population
//   prior_prev_b:   (n_pops × 1) Beta prior beta per population
//   n_pops:        number of populations
//
// Outputs:
//   prior_densities_prev_double: scalar prior density value
//   log_det_J_prev_double:       scalar Jacobian value
//   grad_prev_raw:               (n_pops × 1) gradient of (prior + Jacobian) w.r.t. u_raw
//
ALWAYS_INLINE void fn_MVP_prev_multi_pop_AD( const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_prev_raw,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> prior_prev_a,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> prior_prev_b,
                                             const int n_pops,
                                             //// outputs:
                                             double &prior_densities_prev_double,
                                             double &log_det_J_prev_double,
                                             Eigen::Matrix<double, -1, 1> &grad_prev_raw
) {
  
         stan::math::start_nested();
         
         stan::math::var target_AD_prev = 0.0;
         
         // Create var versions of unconstrained params
         Eigen::Matrix<stan::math::var, -1, 1> u_var(n_pops);
         for (int g = 0; g < n_pops; ++g) {
           u_var(g) = stan::math::to_var(u_prev_raw(g));
         }
         
         // Transform and compute Jacobian + priors
         stan::math::var log_det_J_var = 0.0;
         stan::math::var prior_var = 0.0;
         
         for (int g = 0; g < n_pops; ++g) {
             
             stan::math::var tanh_g = stan::math::tanh(u_var(g));
             stan::math::var prev_g = 0.5 * (tanh_g + 1.0);
             stan::math::var deriv_g = 0.5 * (1.0 - tanh_g * tanh_g);
             
             // Jacobian: log|dp/du|
             log_det_J_var += stan::math::log(deriv_g);
             
             // Beta prior on prev[g]
             prior_var += stan::math::beta_lpdf(prev_g, prior_prev_a(g), prior_prev_b(g));
           
         }
         
         log_det_J_prev_double = log_det_J_var.val();
         prior_densities_prev_double = prior_var.val();
         
         // We differentiate (log_det_J + prior) w.r.t. u_raw
         // But log_det_J gradient is done manually in the likelihood gradient,
         // so here we only AD the prior (matching existing pattern).
         // The log_det_J_prev_double is added to log_prob separately.
         target_AD_prev = prior_var;  // prior only (Jacobian handled manually)
         
         target_AD_prev.grad();
         
         grad_prev_raw.resize(n_pops);
         for (int g = 0; g < n_pops; ++g) {
           grad_prev_raw(g) = u_var(g).adj();
         }
         
         stan::math::set_zero_all_adjoints_nested();
         stan::math::recover_memory_nested();
   
}
 
 
// // ============================================================================
// // 3. Build log_prev for a chunk
// // ============================================================================
// //
// // For chunk-based processing: extracts the per-observation log_prev values
// // for the current chunk from the full N×2 log_prev_mat.
// //
// // This replaces `rowwise_sum.array() += log_prev(0, c);` (which was scalar)
// // with per-observation addition.
// //
// ALWAYS_INLINE void fn_MVP_prev_add_log_prev_to_lp( Eigen::Ref<Eigen::Matrix<double, -1, 1>> rowwise_sum,  // chunk_size × 1 (modified in place)
//                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prev_mat,  // N × 2
//                                                    const int c,                  // class index (0 or 1)
//                                                    const int chunk_offset,       // row offset into full data
//                                                    const int chunk_size
// ) {
//    
//          rowwise_sum.array() += log_prev_mat.col(c).segment(chunk_offset, chunk_size).array();
//    
// }
 
 
// ============================================================================
// 4. Prevalence gradient accumulation (per-population, in chunk loop)
// ============================================================================
//
// In the chunk loop, instead of accumulating a single scalar per class,
// we accumulate per-population. This replaces:
//
//   rowwise_prod = prob[c].rowwise().prod();
//   rowwise_prod.array() = prob_n_recip.array() * rowwise_prod.array();
//   double prev_grad = rowwise_prod.sum();
//   prev_grad_vec(c) += prev_grad;
//
// With per-population accumulation.
//
// Inputs:
//   prob_c:        chunk_size × n_tests probability matrix for class c
//   prob_n_recip:  chunk_size × 1 reciprocal of total probability
//   pop_ind:       N × 1 (full), 0-indexed
//   chunk_offset:  starting row in full data
//   chunk_size:    current chunk size
//   n_pops:        number of populations
//   c:             class index
//
// Output (accumulated):
//   prev_grad_mat: n_pops × n_class matrix of accumulated gradients
//
ALWAYS_INLINE void fn_MVP_prev_multi_pop_accumulate_grad(  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob_c,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> prob_n_recip,
                                                           const Eigen::Ref<const Eigen::Matrix<int, -1, 1>> pop_ind,
                                                           const int chunk_offset,
                                                           const int chunk_size,
                                                           const int n_pops,
                                                           const int c,
                                                           // output (accumulated, NOT reset here):
                                                           Eigen::Matrix<double, -1, -1> &prev_grad_mat,  // n_pops × n_class
                                                           // workspace:
                                                           Eigen::Matrix<double, -1, 1> &rowwise_prod_ws   // chunk_size
) {
   
       rowwise_prod_ws = prob_c.rowwise().prod();
       rowwise_prod_ws.array() *= prob_n_recip.array();
       
       for (int n = 0; n < chunk_size; ++n) {
         int g = pop_ind(chunk_offset + n);
         prev_grad_mat(g, c) += rowwise_prod_ws(n);
       }
   
}
 
 
 // ============================================================================
 // 5. Final prevalence gradient: chain rule u_raw → prev → log_lik
 // ============================================================================
 //
 // After all chunks, converts accumulated per-population gradients to
 // gradients w.r.t. the unconstrained parameters u_raw[g].
 //
 // For each population g:
 //   d(log_lik)/d(u[g]) = [prev_grad_mat(g,1) - prev_grad_mat(g,0)] * dp[g]/du[g]
 //                         + d(log_det_J[g])/du[g]
 //
 // Where d(log_det_J[g])/du[g] = -2 * tanh(u[g])
 //
 // This replaces the existing post-loop code:
 //   prev_unconstrained_grad_vec(c) = prev_grad_vec(c) * deriv_p_wrt_pu_double;
 //   prev_unconstrained_grad_vec(0) = prev_unconstrained_grad_vec(1) - prev_unconstrained_grad_vec(0) - 2*tanh_u_prev[1];
 //
ALWAYS_INLINE void fn_MVP_prev_multi_pop_final_grad( const Eigen::Matrix<double, -1, -1> &prev_grad_mat,  // n_pops × n_class
                                                     const Eigen::Matrix<double, -1, 1>  &tanh_u,         // n_pops
                                                     const Eigen::Matrix<double, -1, 1>  &deriv_p_wrt_u,  // n_pops
                                                     const int n_pops,
                                                     // output:
                                                     Eigen::Matrix<double, -1, 1> &grad_u_raw_out         // n_pops
) {
  
       for (int g = 0; g < n_pops; ++g) {
         
             // Likelihood contribution: (weight_diseased - weight_nondiseased) * dp/du
             double lik_grad = (prev_grad_mat(g, 1) - prev_grad_mat(g, 0)) * deriv_p_wrt_u(g);
             
             // Jacobian contribution: d/du log(dp/du) = -2*tanh(u)
             double jac_grad = -2.0 * tanh_u(g);
             
             grad_u_raw_out(g) = lik_grad + jac_grad;
         
       }
   
}
 
 
 // ============================================================================
 // 6. Convenience: single-population backward-compatible wrapper
 // ============================================================================
 //
 // If n_pops == 1 and you want the old interface, this wraps the multi-pop
 // functions. But you should just use n_pops = 1 with the multi-pop code.
 //
 
 
 // ============================================================================
 // INTEGRATION GUIDE
 // ============================================================================
 //
 // In fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial:
 //
 // 1. NEW DATA REQUIRED:
 //    - pop_ind: Eigen::Matrix<int, -1, 1> of length N, 0-indexed
 //      Add to Model_args_as_cpp_struct (e.g. col_vecs_int[2])
 //    - n_pops: integer (add to Model_args_ints, e.g. index 6)
 //    - prior_p_alpha, prior_p_beta: vectors of length n_pops
 //      Add to Model_args_col_vecs_double
 //
 // 2. PARAMETER LAYOUT CHANGE:
 //    Old: 1 prev param at index (n_corrs + n_covariates_total)
 //    New: n_pops prev params starting at index (n_corrs + n_covariates_total)
 //    Cutpoints now start at: (n_corrs + n_covariates_total + n_pops)
 //    
 //    Replace: `(n_class - 1)` offsets with `n_pops`
 //
 // 3. REPLACE the "prev stuff" double block with:
 //    ```
 //    Eigen::Matrix<double, -1, -1> prev(n_pops, 2);
 //    Eigen::Matrix<double, -1, -1> log_prev_mat(N, 2);
 //    Eigen::Matrix<double, -1, 1>  tanh_u_prev(n_pops);
 //    Eigen::Matrix<double, -1, 1>  deriv_p_wrt_u(n_pops);
 //    double log_det_J_prev_double = 0.0;
 //    
 //    Eigen::Matrix<double, -1, 1> u_prev_raw(n_pops);
 //    for (int g = 0; g < n_pops; ++g) {
 //        u_prev_raw(g) = theta_main_vec_ref(n_corrs + n_covariates_total + g);
 //    }
 //    
 //    fn_prev_multi_pop_transform_double(
 //        u_prev_raw, pop_ind, n_pops, N,
 //        prev, log_prev_mat, tanh_u_prev, deriv_p_wrt_u, log_det_J_prev_double);
 //    ```
 //
 // 4. REPLACE the AD prev block with:
 //    ```
 //    double prior_densities_prev_double = 0.0;
 //    double log_det_J_prev_from_AD = 0.0;
 //    Eigen::Matrix<double, -1, 1> grad_prev_raw(n_pops);
 //    
 //    fn_prev_multi_pop_AD(
 //        u_prev_raw, prior_p_alpha, prior_p_beta, n_pops,
 //        prior_densities_prev_double, log_det_J_prev_from_AD, grad_prev_raw);
 //    
 //    // Write to output (n_pops positions instead of 1)
 //    int prev_start = 1 + n_us + n_corrs + n_covariates_total;
 //    out_mat.segment(prev_start, n_pops) = grad_prev_raw;
 //    ```
 //
 // 5. REPLACE `log_prev(0, c)` usage in chunk loop with:
 //    ```
 //    // Old: rowwise_sum.array() += log_prev(0, c);
 //    // New:
 //    fn_prev_add_log_prev_to_lp(rowwise_sum, log_prev_mat, c,
 //                                chunk_size_orig * chunk_counter, chunk_size);
 //    ```
 //
 // 6. REPLACE prev gradient accumulation in chunk loop:
 //    ```
 //    // Old:
 //    //   rowwise_prod = prob[c].rowwise().prod();
 //    //   rowwise_prod.array() = prob_n_recip.array() * rowwise_prod.array();
 //    //   prev_grad_vec(c) += rowwise_prod.sum();
 //    
 //    // New: declare before chunk loop:
 //    Eigen::Matrix<double, -1, -1> prev_grad_mat = 
 //        Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
 //    
 //    // In the chunk loop (per class c):
 //    fn_prev_multi_pop_accumulate_grad(
 //        prob[c], prob_n_recip, pop_ind,
 //        chunk_size_orig * chunk_counter, chunk_size,
 //        n_pops, c, prev_grad_mat, rowwise_prod);
 //    ```
 //
 // 7. REPLACE post-loop prevalence gradient with:
 //    ```
 //    Eigen::Matrix<double, -1, 1> prev_unc_grad(n_pops);
 //    fn_prev_multi_pop_final_grad(
 //        prev_grad_mat, tanh_u_prev, deriv_p_wrt_u, n_pops, prev_unc_grad);
 //    
 //    int prev_start = 1 + n_us + n_corrs + n_covariates_total;
 //    out_mat.segment(prev_start, n_pops) += prev_unc_grad;
 //    ```
 //
 // 8. CUTPOINT OFFSET: everywhere you see `(n_class - 1)` as the offset
 //    past the coefficients, replace with `n_pops`:
 //    ```
 //    // Old: int i = n_corrs + n_covariates_total + (n_class - 1);
 //    // New: int i = n_corrs + n_covariates_total + n_pops;
 //    ```
 //
 // 9. STAN MODEL: the data block needs:
 //    ```
 //    int<lower=1> n_pops;
 //    array[N] int<lower=1, upper=n_pops> pop;  // 1-indexed in Stan
 //    ```
 //    And the parameters block: `vector[n_pops] p_raw;`
 //    With: `vector<lower=0,upper=1>[n_pops] prev;`
 //    And priors: `for (g in 1:n_pops) prev[g] ~ beta(alpha[g], beta[g]);`
 //
 // ============================================================================
 
 