#pragma once


 

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <tbb/concurrent_vector.h>

 
 
#include <chrono> 
#include <unordered_map>
#include <memory>
#include <thread>
#include <functional>
 
#include <iomanip>
 
#include <numeric>
#include <algorithm>
 
 
//// ANSI codes for different colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

 
using namespace Rcpp;
using namespace Eigen;
 
 



static std::mutex print_mutex;  //// global mutex 
static std::mutex result_mutex_1; //// global mutex
static std::mutex result_mutex_2; //// global mutex


 

  
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE  void                    fn_sample_HMC_multi_iter_single_thread(      HMC_output_single_chain &HMC_output_single_chain_i,
                                                                                    HMCResult &result_input,
                                                                                    const bool burnin_indicator,
                                                                                    const int chain_id,
                                                                                    const int current_iter,
                                                                                    const int seed_main_chain_i,
                                                                                    const int seed_nuisance_chain_i,
                                                                                    T &rng_main_i,
                                                                                    T &rng_nuisance_i,
                                                                                    const int n_iter,
                                                                                    const bool partitioned_HMC,
                                                                                    const bool diffusion_HMC,
                                                                                    const std::string &Model_type,
                                                                                    const bool sample_nuisance,
                                                                                    const bool force_autodiff,
                                                                                    const bool force_PartialLog,
                                                                                    const bool multi_attempts,
                                                                                    const int n_nuisance_to_track,
                                                                                    const Eigen::Matrix<int, -1, -1> &y_Eigen_i,
                                                                                    const Model_fn_args_struct &Model_args_as_cpp_struct,  ///// ALWAYS read-only
                                                                                    EHMC_fn_args_struct  &EHMC_args_as_cpp_struct,
                                                                                    const EHMC_Metric_struct   &EHMC_Metric_as_cpp_struct, 
                                                                                    const Stan_model_struct    &Stan_model_as_cpp_struct,
                                                                                    std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                    const int n_threads_wcp,
                                                                                    bool *persistent_lp_grad_valid = nullptr
)  {
  
         const int N =  Model_args_as_cpp_struct.N;
         const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
         const int n_params_main = Model_args_as_cpp_struct.n_params_main;
         const int n_params = n_params_main + n_nuisance;
 
          Eigen::Matrix<double, -1, 1> theta_prev = result_input.main_theta_vec();
          int n_frozen = 0;
          bool did_healthy_probe = false;
          // Scoped to this call: new inputs / model / gradient settings must
          // first be evaluated. Consecutive joint FKF iterations can reuse lp.
          bool local_lp_grad_valid = false;
          bool &current_lp_grad_valid = persistent_lp_grad_valid == nullptr
                                      ? local_lp_grad_valid : *persistent_lp_grad_valid;
          if (partitioned_HMC || !diffusion_HMC || n_nuisance == 0) {
              current_lp_grad_valid = false;
          }
 
         ///////////////////////////////////////// perform iterations for adaptation interval
         ////// main iteration loop:
         for (int ii = 0; ii < n_iter; ++ii) {
                     
                     if (burnin_indicator) {
                         fn_seed_burnin_rng(rng_main_i, seed_main_chain_i, chain_id, current_iter + ii, 0);
                         fn_seed_burnin_rng(rng_nuisance_i, seed_nuisance_chain_i, chain_id, current_iter + ii, 1);
                     } else {
                     // Preserve the existing post-burn-in streams; only the overlapping burn-in seeds change.
                     #if RNG_TYPE_CPP_STD == 1
                         rng_main_i.seed(seed_main_chain_i + (ii + 1));
                         rng_nuisance_i.seed(1e6 + seed_nuisance_chain_i + (ii + 1));
                     #elif RNG_TYPE_dqrng_xoshiro256plusplus == 1
                         rng_main_i.seed(seed_main_chain_i + (ii + 1));
                         rng_nuisance_i.seed(1e6 + seed_nuisance_chain_i + (ii + 1));
                     #endif 
                     }
                     
                     if (partitioned_HMC == true) {
                   
                               //////////////////////////////////////// sample nuisance (GIVEN main)
                               if ((sample_nuisance == true) && (n_nuisance > 0))   {
                                             
                                            //stan::math::start_nested();
                                            fn_Diffusion_HMC_nuisance_only_single_iter_InPlace_process(    result_input,    
                                                                                                            rng_nuisance_i,
                                                                                                            Model_type, 
                                                                                                            force_autodiff, force_PartialLog,  multi_attempts, 
                                                                                                            y_Eigen_i,
                                                                                                            Model_args_as_cpp_struct,  
                                                                                                            EHMC_args_as_cpp_struct,
                                                                                                            EHMC_Metric_as_cpp_struct, 
                                                                                                            Stan_model_as_cpp_struct, 
                                                                                                   LC_MVP_ws_structs,
                                                                                                             n_threads_wcp);
                                             //stan::math::recover_memory_nested(); 
                                            
                                             HMC_output_single_chain_i.diagnostics_p_jump_us()(ii) =  result_input.us_p_jump();
                                             HMC_output_single_chain_i.diagnostics_div_us()(ii) =  result_input.us_div();
                                   
                                 } /// end of nuisance-part of iteration
                               
                                 { /// sample main GIVEN u's
                                   
                                             //stan::math::start_nested();
                                             fn_standard_HMC_main_only_single_iter_InPlace_process(      result_input,   
                                                                                                         rng_main_i,
                                                                                                         Model_type,  
                                                                                                         force_autodiff, force_PartialLog,  multi_attempts,
                                                                                                         y_Eigen_i,
                                                                                                         Model_args_as_cpp_struct, 
                                                                                                         EHMC_args_as_cpp_struct,
                                                                                                         EHMC_Metric_as_cpp_struct, 
                                                                                                         Stan_model_as_cpp_struct,
                                                                                                         LC_MVP_ws_structs,
                                                                                                         n_threads_wcp);
                                             //stan::math::recover_memory_nested(); 
                                             
                                             HMC_output_single_chain_i.diagnostics_p_jump_main()(ii) =  result_input.main_p_jump();
                                             HMC_output_single_chain_i.diagnostics_div_main()(ii) =  result_input.main_div();
                                   
                                 } /// end of main_params part of iteration
                     
                     } else {  //// sample all params at once 
                       
                                 if ((diffusion_HMC == true) && (n_nuisance > 0)) {
                                   
                                             //stan::math::start_nested();
                                              fn_diffusion_HMC_dual_single_iter_InPlace_process(   result_input,
                                                                                                  rng_main_i,
                                                                                                  rng_nuisance_i,
                                                                                                  Model_type, 
                                                                                                  force_autodiff, force_PartialLog,  multi_attempts, 
                                                                                                  y_Eigen_i,
                                                                                                  Model_args_as_cpp_struct,   
                                                                                                  EHMC_args_as_cpp_struct, 
                                                                                                  EHMC_Metric_as_cpp_struct, 
                                                                                                  Stan_model_as_cpp_struct, 
                                                                                                  LC_MVP_ws_structs,
                                                                                                   n_threads_wcp,
                                                                                                   &current_lp_grad_valid);
                                             //stan::math::recover_memory_nested(); 
                                             
                                             HMC_output_single_chain_i.diagnostics_p_jump_us()(ii) =  result_input.us_p_jump();
                                             HMC_output_single_chain_i.diagnostics_div_us()(ii) =  result_input.us_div();
                                             HMC_output_single_chain_i.diagnostics_p_jump_main()(ii) =  result_input.main_p_jump();
                                             HMC_output_single_chain_i.diagnostics_div_main()(ii) =  result_input.main_div();
                         
                                 } else if (n_nuisance == 0) {
                           
                                             //// ---- NO nuisance block (n_nuisance == 0): the dual samplers have no
                                             //// nuisance coordinates to pair with the main block, so sample the
                                             //// COMPLETE main vector with the standard main-only process (the
                                             //// main-block metric / tau / eps settings are unchanged).
                                             fn_standard_HMC_main_only_single_iter_InPlace_process(    result_input,    
                                                                                                      rng_main_i,
                                                                                                      Model_type, 
                                                                                                      force_autodiff, force_PartialLog,  multi_attempts, 
                                                                                                      y_Eigen_i,
                                                                                                      Model_args_as_cpp_struct,   
                                                                                                      EHMC_args_as_cpp_struct,
                                                                                                      EHMC_Metric_as_cpp_struct, 
                                                                                                      Stan_model_as_cpp_struct,
                                                                                                      LC_MVP_ws_structs,
                                                                                                      n_threads_wcp);
                                             
                                             HMC_output_single_chain_i.diagnostics_p_jump_main()(ii) =  result_input.main_p_jump();
                                             HMC_output_single_chain_i.diagnostics_div_main()(ii) =  result_input.main_div();
                                             
                                 } else {
                           
                                             //stan::math::start_nested();
                                             fn_standard_HMC_dual_single_iter_InPlace_process(    result_input,    
                                                                                                  rng_main_i,
                                                                                                  rng_nuisance_i,
                                                                                                  Model_type, 
                                                                                                  force_autodiff, force_PartialLog,  multi_attempts, 
                                                                                                  y_Eigen_i,
                                                                                                  Model_args_as_cpp_struct,   
                                                                                                  EHMC_args_as_cpp_struct,
                                                                                                  EHMC_Metric_as_cpp_struct, 
                                                                                                  Stan_model_as_cpp_struct,
                                                                                                  LC_MVP_ws_structs,
                                                                                                  n_threads_wcp);
                                             //stan::math::recover_memory_nested(); 
                                             
                                             HMC_output_single_chain_i.diagnostics_p_jump_us()(ii) =  result_input.us_p_jump();
                                             HMC_output_single_chain_i.diagnostics_div_us()(ii) =  result_input.us_div();
                                             HMC_output_single_chain_i.diagnostics_p_jump_main()(ii) =  result_input.main_p_jump();
                                             HMC_output_single_chain_i.diagnostics_div_main()(ii) =  result_input.main_div();
                                             
                                 }
                                         
                     }
                     
                     //// store iteration ii 
                     {
                       
                              // end of each iteration:
                              if ((result_input.main_theta_vec() - theta_prev).cwiseAbs().maxCoeff() == 0.0) ++n_frozen; else n_frozen = 0;
                              theta_prev = result_input.main_theta_vec();
                              
                              // if (n_frozen == 20) {
                              //   const auto &g = result_input.lp_and_grad_outs();
                              //   Eigen::Matrix<double, -1, 1> lp_NoLog = g, lp_AD = g;
                              //   // re-evaluate the CURRENT state two ways, no multi_attempts
                              //   fn_lp_grad_InPlace(lp_NoLog, Model_type, false, false, false,
                              //                      result_input.main_theta_vec(), result_input.us_theta_vec(), y_Eigen_i, "none",
                              //                      Model_args_as_cpp_struct, Stan_model_as_cpp_struct, LC_MVP_ws_structs, 1);
                              //   fn_lp_grad_InPlace(lp_AD,    Model_type, true,  true,  false,
                              //                      result_input.main_theta_vec(), result_input.us_theta_vec(), y_Eigen_i, "all",
                              //                      Model_args_as_cpp_struct, Stan_model_as_cpp_struct, LC_MVP_ws_structs, 1);
                              //   std::cerr << "FROZEN chain=" << chain_id << " iter=" << ii
                              //             << " p_jump_main=" << result_input.main_p_jump()
                              //             << " p_jump_us="   << result_input.us_p_jump()
                              //             << " lp_NoLog=" << lp_NoLog(0) << " lp_AD=" << lp_AD(0)
                              //             << " max|grad_us_AD|="   << lp_AD.segment(1, n_nuisance).cwiseAbs().maxCoeff()
                              //             << " max|grad_main_AD|=" << lp_AD.segment(1 + n_nuisance, n_params_main).cwiseAbs().maxCoeff()
                              //             << " max|u_unc|=" << result_input.us_theta_vec().cwiseAbs().maxCoeff()
                              //             << std::endl << std::flush;
                              // }
                              
                             //  if (n_frozen == 20 || (chain_id == 0 && ii == 30 && !burnin_indicator)) {   // sticky state + one healthy reference
                             //    
                             //        const bool sticky = (n_frozen == 20);
                             //        const long long n_tail = g_n_tail_obs.exchange(0);
                             //        const auto &g = result_input.lp_and_grad_outs();
                             //        Eigen::Matrix<double, -1, 1> g_man = g, g_AD = g;
                             //        fn_lp_grad_InPlace(g_man, Model_type, false, false, false,
                             //                           result_input.main_theta_vec(), result_input.us_theta_vec(), y_Eigen_i, "all",
                             //                           Model_args_as_cpp_struct, Stan_model_as_cpp_struct, LC_MVP_ws_structs, 1);
                             //        fn_lp_grad_InPlace(g_AD,  Model_type, true,  true,  false,
                             //                           result_input.main_theta_vec(), result_input.us_theta_vec(), y_Eigen_i, "all",
                             //                           Model_args_as_cpp_struct, Stan_model_as_cpp_struct, LC_MVP_ws_structs, 1);
                             //  
                             //        //// (a) manual vs AD gradient: worst coordinate in each block
                             //        double worst_us = 0.0, worst_main = 0.0; int k_us = -1, j_main = -1;
                             //        for (int k = 0; k < n_nuisance; ++k) {
                             //            const double r = std::abs(g_man(1 + k) - g_AD(1 + k)) / (std::abs(g_AD(1 + k)) + 1e-6);
                             //            if (r > worst_us) { worst_us = r; k_us = k; }
                             //        }
                             //        for (int j = 0; j < n_params_main; ++j) {
                             //            const double r = std::abs(g_man(1 + n_nuisance + j) - g_AD(1 + n_nuisance + j)) / (std::abs(g_AD(1 + n_nuisance + j)) + 1e-6);
                             //            if (r > worst_main) { worst_main = r; j_main = j; }
                             //        }
                             //  
                             //        //// (b) finite differences of the MANUAL lp (self-consistency): all main params + two u's
                             //        Eigen::Matrix<double, -1, 1> th_us = result_input.us_theta_vec(), th_main = result_input.main_theta_vec(), tmp = g;
                             //        auto lp_man = [&]() {
                             //            tmp = g;
                             //            fn_lp_grad_InPlace(tmp, Model_type, false, false, false, th_main, th_us, y_Eigen_i, "none",
                             //                               Model_args_as_cpp_struct, Stan_model_as_cpp_struct, LC_MVP_ws_structs, 1);
                             //            return tmp(0);
                             //        };
                             //        const double h = 1e-5;
                             //        auto fd_us   = [&](int k) { th_us(k)   += h; const double a = lp_man(); th_us(k)   -= 2*h; const double b = lp_man(); th_us(k)   += h; return (a - b) / (2*h); };
                             //        auto fd_main = [&](int j) { th_main(j) += h; const double a = lp_man(); th_main(j) -= 2*h; const double b = lp_man(); th_main(j) += h; return (a - b) / (2*h); };
                             //        double worst_fd_main = 0.0; int j_fd = -1;
                             //        for (int j = 0; j < n_params_main; ++j) {
                             //            const double gfd = fd_main(j);
                             //            const double r = std::abs(g_man(1 + n_nuisance + j) - gfd) / (std::abs(gfd) + 1e-6);
                             //            if (r > worst_fd_main) { worst_fd_main = r; j_fd = j; }
                             //        }
                             //        Eigen::Index k_big = 0; th_us.cwiseAbs().maxCoeff(&k_big);
                             //        const double fd_us_worstAD = (k_us >= 0) ? fd_us(k_us) : 0.0;
                             //        const double fd_us_maxabs  = fd_us(static_cast<int>(k_big));
                             //  
                             //        std::cerr << (sticky ? "GRADCHECK_STICKY " : "GRADCHECK_HEALTHY ")
                             //                  << "chain=" << chain_id << " iter=" << ii << " p_jump=" << result_input.main_p_jump()
                             //                  << " tail_obs=" << n_tail
                             //                  << " | man-vs-AD  us: rel=" << worst_us << " k=" << k_us << " (n=" << (k_us % N) << ",t=" << (k_us / N) << ")"
                             //                  << "  main: rel=" << worst_main << " j=" << j_main
                             //                  << " | man-vs-FD  main: rel=" << worst_fd_main << " j=" << j_fd
                             //                  << "  us[k_worstAD]: man=" << (k_us >= 0 ? g_man(1 + k_us) : 0.0) << " fd=" << fd_us_worstAD
                             //                  << "  us[k_max|u|]: man=" << g_man(1 + static_cast<int>(k_big)) << " fd=" << fd_us_maxabs
                             //                  << std::endl;
                             //        
                             // }
                             
                             // if ( (n_frozen == 20)
                             //    || (!burnin_indicator && ii == 0 && chain_id < 4)                                                  // the 4 parent states
                             //    || (!burnin_indicator && !did_healthy_probe && ii >= 30 && result_input.main_p_jump() > 0.7) ) {  // a genuinely healthy state
                             //        const char* tag = (n_frozen == 20) ? "PROBE_STICKY" : ((ii == 0) ? "PROBE_PARENT" : "PROBE_HEALTHY");
                             //        if (n_frozen != 20 && ii != 0) did_healthy_probe = true;
                             //  
                             //        const int    n_all = n_nuisance + n_params_main;
                             //        const double eps   = EHMC_args_as_cpp_struct.eps_main;
                             //        Eigen::Matrix<double,-1,1> s(n_all);                                             // sqrt(M_inv), diag metric
                             //        // s.head(n_nuisance)    = EHMC_Metric_as_cpp_struct.M_inv_us_vec.array().sqrt();
                             //        // s.tail(n_params_main) = EHMC_Metric_as_cpp_struct.M_inv_main_vec.array().sqrt();
                             //        const Eigen::Matrix<double,-1,1>  s_us = EHMC_Metric_as_cpp_struct.M_inv_us_vec.array().sqrt();   // diag u-metric
                             //        const Eigen::Matrix<double,-1,-1> &L   = EHMC_Metric_as_cpp_struct.M_inv_dense_main_chol;         // M_inv_main = L L^T
                             //  
                             //        const Eigen::Matrix<double,-1,1> th_us0 = result_input.us_theta_vec(), th_main0 = result_input.main_theta_vec();
                             //        Eigen::Matrix<double,-1,1> gbuf = result_input.lp_and_grad_outs();
                             //        auto grad_at = [&](const Eigen::Matrix<double,-1,1>& d) {                        // grad(lp) at theta + d
                             //            Eigen::Matrix<double,-1,1> th_us = th_us0 + d.head(n_nuisance), th_main = th_main0 + d.tail(n_params_main);
                             //            gbuf = result_input.lp_and_grad_outs();
                             //            fn_lp_grad_InPlace(gbuf, Model_type, false, false, false, th_main, th_us, y_Eigen_i, "all",
                             //                               Model_args_as_cpp_struct, Stan_model_as_cpp_struct, LC_MVP_ws_structs, 1);
                             //            return Eigen::Matrix<double,-1,1>(gbuf.segment(1, n_all));
                             //        };
                             //  
                             //        const double h = 1e-4;
                             //        Eigen::Matrix<double,-1,1> v = Eigen::Matrix<double,-1,1>::Random(n_all);  v.normalize();
                             //        double lam = 0.0;
                             //        for (int it = 0; it < 20; ++it) {
                             //            // // const Eigen::Matrix<double,-1,1> w  = (s.array() * v.array()).matrix();                       // M^{-1/2} v
                             //            // // const Eigen::Matrix<double,-1,1> Hw = -(grad_at(h * w) - grad_at(-h * w)) / (2.0 * h);         // Hess(U) w,  U = -lp
                             //            // // const Eigen::Matrix<double,-1,1> Hv = (s.array() * Hw.array()).matrix();                      // M^{-1/2} H M^{-1/2} v
                             //            // lam = v.dot(Hv);
                             //            // const double nrm = Hv.norm();  if (nrm == 0.0) break;
                             //            // v = Hv / nrm;
                             //            Eigen::Matrix<double,-1,1> w(n_all);                                            // M^{-1/2} v
                             //            w.head(n_nuisance)    = (s_us.array() * v.head(n_nuisance).array()).matrix();
                             //            w.tail(n_params_main) = L * v.tail(n_params_main);
                             //            const Eigen::Matrix<double,-1,1> Hw = -(grad_at(h * w) - grad_at(-h * w)) / (2.0 * h);
                             //            Eigen::Matrix<double,-1,1> Hv(n_all);                                           // M^{-1/2} H M^{-1/2} v
                             //            Hv.head(n_nuisance)    = (s_us.array() * Hw.head(n_nuisance).array()).matrix();
                             //            Hv.tail(n_params_main) = L.transpose() * Hw.tail(n_params_main);
                             //            lam = v.dot(Hv);
                             //            const double nrm = Hv.norm();  if (nrm == 0.0) break;
                             //            v = Hv / nrm;
                             //        }
                             //  
                             //        std::vector<int> idx(n_all);  std::iota(idx.begin(), idx.end(), 0);
                             //        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b){ return std::abs(v(a)) > std::abs(v(b)); });
                             //        std::cerr << tag << " chain=" << chain_id << " iter=" << ii << " p_jump=" << result_input.main_p_jump()
                             //                  << " lam_max=" << lam << " eps2_lam=" << eps * eps * lam << " (unstable>4)"
                             //                  << " frac_us=" << v.head(n_nuisance).squaredNorm() << " top:";
                             //        for (int q = 0; q < 5; ++q) {
                             //            const int k = idx[q];
                             //            if (k < n_nuisance) std::cerr << " u[n=" << (k % N) << ",t=" << (k / N) << "]=" << v(k);
                             //            else                std::cerr << " main[j=" << (k - n_nuisance) << "]=" << v(k);
                             //        }
                             //        std::cerr << std::endl;
                             // }
                        
                             // const bool rejected = (result_input.main_p_jump() < 1e-8) &&
                             // (!sample_nuisance || result_input.us_p_jump() < 1e-8);
                             // n_consec_reject = rejected ? n_consec_reject + 1 : 0;
                             // 
                             // if (n_consec_reject == 20) {
                             //    const auto &g = result_input.lp_and_grad_outs();
                             //    std::cerr << "STUCK chain=" << chain_id << " iter=" << ii
                             //              << " lp=" << g(0)
                             //              << " max|u_unc|=" << result_input.us_theta_vec().cwiseAbs().maxCoeff()
                             //              << " max|grad_us|=" << g.segment(1, n_nuisance).cwiseAbs().maxCoeff()
                             //              << " max|grad_main|=" << g.segment(1 + n_nuisance, n_params_main).cwiseAbs().maxCoeff()
                             //              << " max|v_us|=" << result_input.us_velocity_vec().cwiseAbs().maxCoeff()
                             //              << std::endl;
                             // }
                             //// this handles both RAM and HDD storage modes:
                             HMC_output_single_chain_i.store_iteration( ii, 
                                                                        sample_nuisance, 
                                                                        result_input); 
                       
                             // ////std::lock_guard<std::mutex> lock(result_mutex_1);  
                             // 
                             // // HMC_output_single_chain_i.store_iteration(ii, sample_nuisance);
                             // HMC_output_single_chain_i.trace_main().col(ii) = result_input.main_theta_vec(); 
                             // 
                             // if (sample_nuisance == true) {
                             //      HMC_output_single_chain_i.trace_div()(0, ii) =  (0.50 * (result_input.main_div() + result_input.us_div()));
                             //      HMC_output_single_chain_i.trace_nuisance().col(ii) = result_input.us_theta_vec();  
                             // } else {
                             //      HMC_output_single_chain_i.trace_div()(0, ii) = result_input.main_div();
                             // }
                             // 
                             // HMC_output_single_chain_i.trace_log_lik().col(ii) = result_input.log_lik();
                     }
                    
                     // if ((burnin_indicator && ii == n_iter - 1) || (!burnin_indicator && ii == 0)) {
                     //      std::cerr << (burnin_indicator ? "BURNIN_END " : "SAMPLE_START ")
                     //                << "chain=" << chain_id
                     //                << " sum_main=" << std::setprecision(17) << result_input.main_theta_vec().sum()
                     //                << " sum_us="   << result_input.us_theta_vec().sum()
                     //                << " lp0="      << result_input.lp_and_grad_outs_0()(0)
                     //                << " eps="      << EHMC_args_as_cpp_struct.eps_main
                     //                << std::endl;
                     // }
                     
                     // if ((burnin_indicator && ii == n_iter - 1) || (!burnin_indicator && ii == 0)) {
                     //      Eigen::Matrix<double, -1, 1> g_fresh = result_input.lp_and_grad_outs();   // just for sizing
                     //      fn_lp_grad_InPlace(g_fresh, Model_type, force_autodiff, force_PartialLog, multi_attempts,
                     //                         result_input.main_theta_vec(), result_input.us_theta_vec(), y_Eigen_i, "all",
                     //                         Model_args_as_cpp_struct, Stan_model_as_cpp_struct, LC_MVP_ws_structs, n_threads_wcp);
                     //      std::cerr << (burnin_indicator ? "BURNIN_END " : "SAMPLE_START ")
                     //                << "chain=" << chain_id
                     //                << std::setprecision(17)
                     //                << " sum_main=" << result_input.main_theta_vec().sum()
                     //                << " sum_us="   << result_input.us_theta_vec().sum()
                     //                << " lp="       << g_fresh(0)
                     //                << " max|g_us|="   << g_fresh.segment(1, n_nuisance).cwiseAbs().maxCoeff()
                     //                << " max|g_main|=" << g_fresh.segment(1 + n_nuisance, n_params_main).cwiseAbs().maxCoeff()
                     //                << " eps=" << EHMC_args_as_cpp_struct.eps_main
                     //                << " wcp=" << n_threads_wcp
                     //                << std::endl;
                     // }
                           
                     if (burnin_indicator == false) {
                       if ((chain_id == 0) || (chain_id == 1)) {
                         if (ii %  static_cast<int>(std::round(static_cast<double>(n_iter)/4.0)) == 0) {
                           ////std::lock_guard<std::mutex> lock(print_mutex);
                           double pct_complete = 100.0 * (static_cast<double>(ii) / static_cast<double>(n_iter));
                           std::cout << "Chain #" << chain_id << " - Sampling is around " << pct_complete << " % complete" << "\n";
                         }
                       }
                     }
                 
         } ////////////////////// end of iteration(s)
         
    {
           // std::lock_guard<std::mutex> lock(result_mutex_2);
           HMC_output_single_chain_i.result_input() = result_input;
    }
     
}
     
     
     
     
     
     
     
     
     
     

     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
