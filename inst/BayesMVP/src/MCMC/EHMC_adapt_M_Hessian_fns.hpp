
#pragma once

 

#include <Eigen/Dense>
 
 
 
 
using namespace Eigen;

 

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
 


 
 
// Helper to check positive-definiteness using Eigen's LDLT decomposition
bool is_positive_definite(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  mat) {
  
        Eigen::LLT<Eigen::Matrix<double, -1, -1> > llt((mat + mat.transpose()) * 0.50);  /// make symmetric first  and then attempt Cholesky factorisation
        
        return llt.info() == Eigen::Success;
  
}


// Helper function to make a matrix positive-definite
Eigen::Matrix<double, -1, -1> near_PD(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  mat) {
  
         // Eigen::Matrix<double, -1, -1>  symMat = (mat + mat.transpose()) / 2.0; 
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, -1, -1> > es((mat + mat.transpose()) * 0.50); // Make symmetric and apply es
        Eigen::Matrix<double, -1, 1>   eigenValues = es.eigenvalues();
        Eigen::Matrix<double, -1, -1>  eigenVectors = es.eigenvectors();
        
        
        double max_eigenvalue = eigenValues.maxCoeff();
        double epsilon = std::max(1e-8, 1e-6 * max_eigenvalue);
        for (int i = 0; i < eigenValues.size(); ++i) {
          eigenValues(i) = std::max(eigenValues(i), epsilon);
        }
        
        // // Shift eigenvalues to ensure all are positive
        // for (int i = 0; i < eigenValues.size(); ++i) {
        //   if (eigenValues(i) < 0.0) {
        //     eigenValues(i) = 1e-6; // Set negative eigenvalues to a small positive value
        //   }
        // } 
        
        // Recompose the matrix with positive eigenvalues
        return eigenVectors * eigenValues.asDiagonal() * eigenVectors.transpose();
  
}


Eigen::Matrix<double, -1, -1> shrink_hessian( const Eigen::Matrix<double, -1, -1> &hessian, 
                                              double shrinkage_factor) {
  
        Eigen::Matrix<double, -1, -1> diagonal = hessian.diagonal().asDiagonal();
        return (1 - shrinkage_factor) * hessian + shrinkage_factor * diagonal;
  
}


Eigen::Matrix<double, -1, -1> num_diff_Hessian_main_given_nuisance(   const double num_diff_e,
                                                                      const double shrinkage_factor,
                                                                      const std::string &Model_type,
                                                                      const bool force_autodiff,
                                                                      const bool force_PartialLog,
                                                                      const bool multi_attemps,
                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                      const Model_fn_args_struct  &Model_args_as_cpp_struct,
                                                                      const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                      LC_MVP_workspace_struct &LC_MVP_ws_struct
) {
          
          const int n_params_main = theta_main_vec_ref.rows();
          const int n_us = theta_us_vec_ref.rows();
          const int n_params = n_params_main + n_us;
          const int N = y_ref.rows();
          const std::string grad_option = "main_only";
          
          Eigen::Matrix<double, -1, 1> lp_and_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
          Eigen::Matrix<double, -1, -1> Hessian(n_params_main, n_params_main);
          Eigen::Matrix<double, -1, 1> theta_main_vec_perturbed = theta_main_vec_ref;
          
          double num_diff_e_inv = 1.0 / (2 * num_diff_e);
          
          Eigen::Matrix<double, -1, 1> grad_plus(n_params_main);
          Eigen::Matrix<double, -1, 1> grad_minus(n_params_main);
          
          std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs;
          LC_MVP_ws_structs.resize(1);
          LC_MVP_ws_structs[0] = LC_MVP_ws_struct;
          
          for (int j = 0; j < n_params_main; ++j) {
            
                // Compute g(x + hⱼeⱼ)
                theta_main_vec_perturbed(j) = theta_main_vec_ref(j) + num_diff_e;
            
                fn_lp_grad_InPlace(  lp_and_grad_outs, 
                                     Model_type,
                                     force_autodiff, force_PartialLog, multi_attemps,
                                     theta_main_vec_perturbed, theta_us_vec_ref, y_ref, grad_option, 
                                     Model_args_as_cpp_struct,
                                     Stan_model_as_cpp_struct,
                                     LC_MVP_ws_structs,
                                     1);
                
                grad_plus.array() = - lp_and_grad_outs.segment(1 + n_us, n_params_main).array();
                
                // Compute g(x - hⱼeⱼ)
                theta_main_vec_perturbed(j) = theta_main_vec_ref(j) - num_diff_e;
                
                fn_lp_grad_InPlace(  lp_and_grad_outs, 
                                     Model_type, 
                                     force_autodiff, force_PartialLog, multi_attemps,
                                     theta_main_vec_perturbed, theta_us_vec_ref, y_ref, grad_option, 
                                     Model_args_as_cpp_struct,
                                     Stan_model_as_cpp_struct,
                                     LC_MVP_ws_structs,
                                     1);
                
                grad_minus.array() = - lp_and_grad_outs.segment(1 + n_us, n_params_main).array();
                
                // Compute j-th column of Hessian
                Hessian.col(j) = (grad_plus - grad_minus) * num_diff_e_inv;
                
                // Reset perturbed vector
                theta_main_vec_perturbed(j) = theta_main_vec_ref(j);
            
          }
          
          // Symmetrize the Hessian
          Hessian = (Hessian + Hessian.transpose()) * 0.5;
          
          // Apply shrinkage
          Hessian = shrink_hessian(Hessian, shrinkage_factor);
          
          // Symmetrize the Hessian
          Hessian = (Hessian + Hessian.transpose()) * 0.5;
          
          return Hessian;
  
}



Eigen::Matrix<double, -1, -1> num_diff_Hessian_main_given_nuisance_parallel(   const double num_diff_e,
                                                                               const double shrinkage_factor,
                                                                               const std::string &Model_type,
                                                                               const bool force_autodiff,
                                                                               const bool force_PartialLog,
                                                                               const bool multi_attemps,
                                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                               const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                               const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                               const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                               std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs_per_thread, // one per thread
                                                                               const int n_threads
) {
  
          const int n_params_main = theta_main_vec_ref.rows();
          const int n_us = theta_us_vec_ref.rows();
          const int n_params = n_params_main + n_us;
          const int N = y_ref.rows();
          const std::string grad_option = "main_only";
          const double num_diff_e_inv = 1.0 / (2.0 * num_diff_e);
          
          Eigen::Matrix<double, -1, -1> Hessian(n_params_main, n_params_main);
          
          #pragma omp parallel num_threads(n_threads)
          {
                stan::math::ChainableStack ad_tape;
                
                const int tid = omp_get_thread_num();
                
                // Thread-local buffers
                Eigen::Matrix<double, -1, 1> lp_and_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
                Eigen::Matrix<double, -1, 1> theta_perturbed = theta_main_vec_ref;
                Eigen::Matrix<double, -1, 1> grad_plus(n_params_main);
                Eigen::Matrix<double, -1, 1> grad_minus(n_params_main);
                
                // Thread-local workspace (wrapped in vector for fn_lp_grad_InPlace signature)
                std::vector<LC_MVP_workspace_struct> ws_vec(1);
                ws_vec[0] = LC_MVP_ws_structs_per_thread[tid];
                
                #pragma omp for schedule(dynamic)
                for (int j = 0; j < n_params_main; ++j) {
                  
                      // g(x + h*e_j)
                      theta_perturbed(j) = theta_main_vec_ref(j) + num_diff_e;
                      
                      fn_lp_grad_InPlace(lp_and_grad_outs,
                                         Model_type,
                                         force_autodiff, force_PartialLog, multi_attemps,
                                         theta_perturbed, theta_us_vec_ref, y_ref, grad_option,
                                         Model_args_as_cpp_struct,
                                         Stan_model_as_cpp_struct,
                                         ws_vec,
                                         1);
                      
                      grad_plus = -lp_and_grad_outs.segment(1 + n_us, n_params_main);
                      
                      // g(x - h*e_j)
                      theta_perturbed(j) = theta_main_vec_ref(j) - num_diff_e;
                      
                      fn_lp_grad_InPlace(lp_and_grad_outs,
                                         Model_type,
                                         force_autodiff, force_PartialLog, multi_attemps,
                                         theta_perturbed, theta_us_vec_ref, y_ref, grad_option,
                                         Model_args_as_cpp_struct,
                                         Stan_model_as_cpp_struct,
                                         ws_vec,
                                         1);
                      
                      grad_minus = -lp_and_grad_outs.segment(1 + n_us, n_params_main);
                      
                      // Write to column j — no sync needed, columns are independent
                      Hessian.col(j) = (grad_plus - grad_minus) * num_diff_e_inv;
                      
                      // Reset
                      theta_perturbed(j) = theta_main_vec_ref(j);
                  
                }
          }
        
        Hessian = (Hessian + Hessian.transpose()) * 0.5;
        Hessian = shrink_hessian(Hessian, shrinkage_factor); 
        Hessian = (Hessian + Hessian.transpose()) * 0.5;
        
        return Hessian;

} 



 


Eigen::Matrix<double, -1, -1>  compute_PD_Hessian_main(     const double shrinkage_factor,
                                                            const double num_diff_e,
                                                            const std::string  &Model_type,
                                                            const bool force_autodiff,
                                                            const bool force_PartialLog,
                                                            const bool multi_attemps, 
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                            const Model_fn_args_struct  &Model_args_as_cpp_struct
) {
  
  
          const int n_params_main = theta_main_vec_ref.rows();
          Eigen::Matrix<double, -1, -1>    Hessian(n_params_main, n_params_main);
          
        
          
          if (Model_type == "Stan") {
                  
                  Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct.model_so_file, 
                                                                                             Model_args_as_cpp_struct.json_file_path, 
                                                                                             123);
                  // LC_MVP_workspace_struct LC_MVP_ws_struct; // dummy struct
                  
                  const int n_threads_hessian = std::min(n_params_main, omp_get_max_threads());
                  std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs(n_threads_hessian); // dummy structs
                  
                  //// compute proposed Hessian
                  Hessian  = num_diff_Hessian_main_given_nuisance_parallel(    num_diff_e,
                                                                      shrinkage_factor,
                                                                      Model_type,
                                                                      force_autodiff,
                                                                      force_PartialLog,
                                                                      multi_attemps,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref, 
                                                                      y_ref,
                                                                      Model_args_as_cpp_struct, 
                                                                      Stan_model_as_cpp_struct,
                                                                      LC_MVP_ws_structs,
                                                                      n_threads_hessian);
                  
                  
                  
                  //// destroy Stan model object
                  fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
            
          } else {
            
                  Stan_model_struct Stan_model_as_cpp_struct; /// dummy struct 
            
                  // LC_MVP_workspace_struct LC_MVP_ws_struct;
                  
                  const int n_threads_hessian = std::min(n_params_main, omp_get_max_threads());
                  std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs(n_threads_hessian);
                  
                  if ((Model_type == "LC_MVP") || (Model_type == "LC_MVOP")) {
                          
                          const int n_tests = y_ref.cols();
                          const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
                          const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
                          const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
                          const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
                          
                          const int n_corrs = n_class * n_tests * (n_tests - 1) / 2;
                          
                          int n_covariates_total, n_covariates_max;
                          if (n_class > 1) {
                            const int n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
                            const int n_covariates_total_d  = n_covariates_per_outcome_vec.row(1).sum();
                            n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
                            const int n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
                            const int n_covariates_max_d  = n_covariates_per_outcome_vec.row(1).maxCoeff();
                            n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);
                          } else {
                            n_covariates_total = n_covariates_per_outcome_vec.sum();
                            n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
                          }
                          
                          int vec_size;
                          if      (vect_type == "AVX512") vec_size = 8;
                          else if (vect_type == "AVX2")   vec_size = 4;
                          else if (vect_type == "AVX")    vec_size = 2;
                          else                            vec_size = 1;
                          
                          const int N = y_ref.rows();
                          
                          ChunkSizeInfo chunk_info = calculate_chunk_sizes( N, 
                                                                            vec_size, 
                                                                            n_chunks);

                          for (int t = 0; t < n_threads_hessian; ++t) {
                            LC_MVP_ws_structs[t].allocate(chunk_info.normal_chunk_size,
                                                          n_tests,
                                                          n_class,
                                                          n_covariates_max,
                                                          n_corrs,
                                                          n_covariates_total);
                            // int padded_chunk = ((chunk_info.normal_chunk_size + vec_size - 1) / vec_size) * vec_size;
                            // LC_MVP_ws_structs[t].allocate(padded_chunk,
                            //                               n_tests,
                            //                               n_class,
                            //                               n_covariates_max,
                            //                               n_corrs,
                            //                               n_covariates_total);
                          }
                    
                  } // else: LC_MVP_ws_struct[i].is_allocated stays false (dummy)
                  
                  // std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs;
                  // LC_MVP_ws_structs.resize(1);
                  // LC_MVP_ws_structs[0] = LC_MVP_ws_struct;
                  
                  //// compute proposed Hessian
                  Hessian  = num_diff_Hessian_main_given_nuisance_parallel(    num_diff_e,
                                                                      shrinkage_factor,
                                                                      Model_type,
                                                                      force_autodiff,
                                                                      force_PartialLog,
                                                                      multi_attemps,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref, 
                                                                      y_ref,
                                                                      Model_args_as_cpp_struct, 
                                                                      Stan_model_as_cpp_struct, 
                                                                      LC_MVP_ws_structs,
                                                                      n_threads_hessian);
            
          }
  
          // force-symmetric positive-definiteness check
          if (!is_positive_definite(Hessian)) {
            Hessian = near_PD(Hessian); // Make the Hessian positive-definite
          }
          
          return Hessian;
          
}


void update_M_dense_main_Hessian_InPlace( Eigen::Ref<Eigen::Matrix<double, -1, -1>> M_dense_main,  /// to be updated 
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> M_inv_dense_main, /// to be updated 
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> M_inv_dense_main_chol, /// to be updated 
                                          const double shrinkage_factor,
                                          const double ratio,
                                          const int interval_width,
                                          const double num_diff_e,
                                          const std::string  &Model_type,
                                          const bool force_autodiff,
                                          const bool force_PartialLog,
                                          const bool multi_attemps, 
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                          const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                          const Model_fn_args_struct  &Model_args_as_cpp_struct,
                                          const double   &ii, 
                                          const double   &n_burnin, 
                                          const std::string &metric_type
) {

          const int n_params_main = theta_main_vec_ref.rows();
          Eigen::Matrix<double, -1, -1>    Hessian(n_params_main, n_params_main);
          
          if (Model_type == "Stan") {
          
                        Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct.model_so_file, 
                                                                                                   Model_args_as_cpp_struct.json_file_path, 
                                                                                                   123);
                        // LC_MVP_workspace_struct LC_MVP_ws_struct; // dummy
                        
                        const int n_threads_hessian = std::min(n_params_main, omp_get_max_threads());
                        std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs(n_threads_hessian);
                        
                        // std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs;
                        // LC_MVP_ws_structs.resize(1);
                        // LC_MVP_ws_structs[0] = LC_MVP_ws_struct;
                        
                        //// compute proposed Hessian
                        Hessian  = num_diff_Hessian_main_given_nuisance_parallel(    num_diff_e,
                                                                            shrinkage_factor,
                                                                            Model_type,
                                                                            force_autodiff,
                                                                            force_PartialLog,
                                                                            multi_attemps,
                                                                            theta_main_vec_ref,
                                                                            theta_us_vec_ref, 
                                                                            y_ref,
                                                                            Model_args_as_cpp_struct, 
                                                                            Stan_model_as_cpp_struct,
                                                                            LC_MVP_ws_structs,
                                                                            n_threads_hessian);
        
        
      
                        //// destroy Stan model object
                        fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
                          
          } else {
            
                        Stan_model_struct Stan_model_as_cpp_struct; /// dummy struct 
                        
                        const int n_threads_hessian = std::min(n_params_main, omp_get_max_threads());
                        std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs(n_threads_hessian);
                        
                        if ((Model_type == "LC_MVP") || (Model_type == "LC_MVOP")) {
                          
                              const int n_tests = y_ref.cols();
                              const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
                              const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
                              const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
                              const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
                              
                              const int n_corrs = n_class * n_tests * (n_tests - 1) / 2;
                              
                              int n_covariates_total, n_covariates_max;
                              if (n_class > 1) {
                                const int n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
                                const int n_covariates_total_d  = n_covariates_per_outcome_vec.row(1).sum();
                                n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
                                const int n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
                                const int n_covariates_max_d  = n_covariates_per_outcome_vec.row(1).maxCoeff();
                                n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);
                              } else { 
                                n_covariates_total = n_covariates_per_outcome_vec.sum();
                                n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
                              } 
                              
                              int vec_size;
                              if      (vect_type == "AVX512") vec_size = 8;
                              else if (vect_type == "AVX2")   vec_size = 4;
                              else if (vect_type == "AVX")    vec_size = 2;
                              else                            vec_size = 1;
                              
                              const int N = y_ref.rows();
                              
                              ChunkSizeInfo chunk_info = calculate_chunk_sizes( N, 
                                                                                vec_size, 
                                                                                n_chunks); 
                              
                              for (int t = 0; t < n_threads_hessian; ++t) {
                                LC_MVP_ws_structs[t].allocate(chunk_info.normal_chunk_size,
                                                              n_tests,
                                                              n_class,
                                                              n_covariates_max,
                                                              n_corrs,
                                                              n_covariates_total);
                                // int padded_chunk = ((chunk_info.normal_chunk_size + vec_size - 1) / vec_size) * vec_size;
                                // LC_MVP_ws_structs[t].allocate(padded_chunk,
                                //                               n_tests,
                                //                               n_class,
                                //                               n_covariates_max,
                                //                               n_corrs,
                                //                               n_covariates_total);
                              }
                          
                        } // else: LC_MVP_ws_struct[i].is_allocated stays false (dummy) 
            
                        //// compute proposed Hessian
                        Hessian  = num_diff_Hessian_main_given_nuisance_parallel(    num_diff_e,
                                                                            shrinkage_factor,
                                                                            Model_type,
                                                                            force_autodiff,
                                                                            force_PartialLog,
                                                                            multi_attemps,
                                                                            theta_main_vec_ref,
                                                                            theta_us_vec_ref, 
                                                                            y_ref,
                                                                            Model_args_as_cpp_struct, 
                                                                            Stan_model_as_cpp_struct,
                                                                            LC_MVP_ws_structs,
                                                                            n_threads_hessian);
            
          }
                
          // force-symmetric positive-definiteness check
          if (!is_positive_definite(Hessian)) {
            Hessian = near_PD(Hessian); // Make the Hessian positive-definite
          }
          
          // update M_dense_main
          M_dense_main = (1.0 - ratio) * M_dense_main + ratio * Hessian;  
          
         
          if (!is_positive_definite(M_dense_main)) {
            M_dense_main = near_PD(M_dense_main);
          }
          
          // update M_inv_dense_main
          M_inv_dense_main = M_dense_main.inverse();
          
          // update M_inv_dense_main_chol
          Eigen::LLT<Eigen::MatrixXd> llt(M_inv_dense_main);
          if (llt.info() == Eigen::Success) {
            
             M_inv_dense_main_chol = llt.matrixL();
            
          } else {
            
             throw std::runtime_error("Cholesky decomposition failed");
            
          }
 
}









// Compute only diagonal of Hessian using numerical differentiation
Eigen::Matrix<double, -1, 1> num_diff_Hessian_diag_main_given_nuisance( const double num_diff_e,
                                                                        const double shrinkage_factor,
                                                                        const std::string &Model_type,
                                                                        const bool force_autodiff,
                                                                        const bool force_PartialLog,
                                                                        const bool multi_attemps, 
                                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                        const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                        const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                        const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                        LC_MVP_workspace_struct &LC_MVP_ws_struct
) {
  
                    const int n_params_main = theta_main_vec_ref.rows();
                    const int n_us = theta_us_vec_ref.rows();
                    const int n_params = n_params_main + n_us;
                    const int N = y_ref.rows();
                    const std::string grad_option = "main_only";
                    
                    Eigen::Matrix<double, -1, 1> lp_and_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
                    Eigen::Matrix<double, -1, 1> Hessian_diag(n_params_main);
                    Eigen::Matrix<double, -1, 1> theta_main_vec_perturbed = theta_main_vec_ref;
                    
                    double num_diff_e_inv = 1.0 / (2 * num_diff_e);
                    
                    std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs;
                    LC_MVP_ws_structs.resize(1);
                    LC_MVP_ws_structs[0] = LC_MVP_ws_struct;
                    
                    for (int j = 0; j < n_params_main; ++j) {
                      
                          // Compute g(x + hⱼeⱼ) - only need j-th component
                          theta_main_vec_perturbed(j) = theta_main_vec_ref(j) + num_diff_e;
                      
                          fn_lp_grad_InPlace(lp_and_grad_outs, 
                                             Model_type,
                                             force_autodiff, force_PartialLog, multi_attemps,
                                             theta_main_vec_perturbed, theta_us_vec_ref, y_ref, grad_option, 
                                             Model_args_as_cpp_struct, 
                                             Stan_model_as_cpp_struct, 
                                             LC_MVP_ws_structs,
                                             1);
                          
                          double grad_j_plus = -lp_and_grad_outs(1 + n_us + j);
                          
                          // Compute g(x - hⱼeⱼ) - only need j-th component
                          theta_main_vec_perturbed(j) = theta_main_vec_ref(j) - num_diff_e;
                          
                          fn_lp_grad_InPlace(lp_and_grad_outs, 
                                             Model_type, 
                                             force_autodiff, force_PartialLog, multi_attemps,
                                             theta_main_vec_perturbed, theta_us_vec_ref, y_ref, grad_option, 
                                             Model_args_as_cpp_struct, 
                                             Stan_model_as_cpp_struct,
                                             LC_MVP_ws_structs,
                                             1);
                          
                          double grad_j_minus = -lp_and_grad_outs(1 + n_us + j);
                          
                          // Compute j-th diagonal element of Hessian
                          Hessian_diag(j) = (grad_j_plus - grad_j_minus) * num_diff_e_inv;
                          
                          // Reset perturbed vector
                          theta_main_vec_perturbed(j) = theta_main_vec_ref(j);
                      
                    }
                    
                    // Apply shrinkage (for diagonal, this is just mixing with 1's)
                    Hessian_diag = (1.0 - shrinkage_factor) * Hessian_diag.array() + shrinkage_factor;
                    
                    return Hessian_diag;
  
} 





// Compute positive-definite diagonal Hessian
Eigen::Matrix<double, -1, 1> compute_PD_Hessian_diag_main(  const double shrinkage_factor,
                                                            const double num_diff_e,
                                                            const std::string &Model_type,
                                                            const bool force_autodiff,
                                                            const bool force_PartialLog,
                                                            const bool multi_attemps, 
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                            const Model_fn_args_struct &Model_args_as_cpp_struct
) {
  
                    const int n_params_main = theta_main_vec_ref.rows();
                    Eigen::Matrix<double, -1, 1> Hessian_diag(n_params_main);
                    
                    if (Model_type == "Stan") {
                      
                          Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data( Model_args_as_cpp_struct.model_so_file, 
                                                                                                    Model_args_as_cpp_struct.json_file_path,  
                                                                                                    123);
                          LC_MVP_workspace_struct LC_MVP_ws_struct; // dummy
                          
                          std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs;
                          LC_MVP_ws_structs.resize(1);
                          LC_MVP_ws_structs[0] = LC_MVP_ws_struct;
                          
                          // Compute diagonal Hessian
                          Hessian_diag = num_diff_Hessian_diag_main_given_nuisance( num_diff_e, shrinkage_factor, Model_type,
                                                                                    force_autodiff, force_PartialLog, multi_attemps,
                                                                                    theta_main_vec_ref, theta_us_vec_ref, y_ref,
                                                                                    Model_args_as_cpp_struct, 
                                                                                    Stan_model_as_cpp_struct, 
                                                                                    LC_MVP_ws_struct);
                           
                          // Destroy Stan model object
                          fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
                      
                    } else {  
                      
                          Stan_model_struct Stan_model_as_cpp_struct; // dummy struct 
                          
                          LC_MVP_workspace_struct LC_MVP_ws_struct;
                        
                          if ((Model_type == "LC_MVP") || (Model_type == "LC_MVOP")) {
                            
                                    const int n_tests = y_ref.cols();
                                    const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
                                    const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
                                    const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
                                    const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
                                    
                                    const int n_corrs = n_class * n_tests * (n_tests - 1) / 2;
                                    
                                    int n_covariates_total, n_covariates_max;
                                    if (n_class > 1) {
                                      const int n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
                                      const int n_covariates_total_d  = n_covariates_per_outcome_vec.row(1).sum();
                                      n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
                                      const int n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
                                      const int n_covariates_max_d  = n_covariates_per_outcome_vec.row(1).maxCoeff();
                                      n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);
                                    } else {
                                      n_covariates_total = n_covariates_per_outcome_vec.sum();
                                      n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
                                    }
                                    
                                    int vec_size;
                                    if      (vect_type == "AVX512") vec_size = 8;
                                    else if (vect_type == "AVX2")   vec_size = 4;
                                    else if (vect_type == "AVX")    vec_size = 2;
                                    else                            vec_size = 1;
                                    
                                    const int N = y_ref.rows();
                                    
                                    ChunkSizeInfo chunk_info = calculate_chunk_sizes( N, 
                                                                                      vec_size, 
                                                                                      n_chunks);
                                    
                                    LC_MVP_ws_struct.allocate( chunk_info.normal_chunk_size,
                                                               n_tests,
                                                               n_class,
                                                               n_covariates_max,
                                                               n_corrs,
                                                               n_covariates_total);
                                    // int padded_chunk = ((chunk_info.normal_chunk_size + vec_size - 1) / vec_size) * vec_size;
                                    // LC_MVP_ws_struct.allocate(  padded_chunk,
                                    //                             n_tests,
                                    //                             n_class,
                                    //                             n_covariates_max,
                                    //                             n_corrs,
                                    //                             n_covariates_total);
                            
                          } // else: LC_MVP_ws_struct[i].is_allocated stays false (dummy)
                          
                          std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs;
                          LC_MVP_ws_structs.resize(1);
                          LC_MVP_ws_structs[0] = LC_MVP_ws_struct;
                          
                          // Compute diagonal Hessian
                          Hessian_diag = num_diff_Hessian_diag_main_given_nuisance( num_diff_e, shrinkage_factor, Model_type,
                                                                                    force_autodiff, force_PartialLog, multi_attemps,
                                                                                    theta_main_vec_ref, theta_us_vec_ref, y_ref,
                                                                                    Model_args_as_cpp_struct,
                                                                                    Stan_model_as_cpp_struct, 
                                                                                    LC_MVP_ws_struct); 
                      
                    }
                    
                    // Ensure all diagonal elements are positive
                    double min_eigenvalue = Hessian_diag.minCoeff();
                    double epsilon = std::max(1e-8, 1e-6 * Hessian_diag.maxCoeff());
                    
                    for (int i = 0; i < n_params_main; ++i) {
                      if (Hessian_diag(i) < epsilon) {
                        Hessian_diag(i) = epsilon;
                      } 
                    }
                    
                    return Hessian_diag;
  
}





// Update diagonal metric using Hessian
void update_M_diag_main_Hessian_InPlace(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> M_main_vec,  // to be updated 
                                          Eigen::Ref<Eigen::Matrix<double, -1, 1>> M_inv_main_vec, // to be updated 
                                          const double shrinkage_factor,
                                          const double ratio,
                                          const int interval_width,
                                          const double num_diff_e,
                                          const std::string &Model_type,
                                          const bool force_autodiff,
                                          const bool force_PartialLog,
                                          const bool multi_attemps, 
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                          const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                          const Model_fn_args_struct &Model_args_as_cpp_struct,
                                          const double &ii, 
                                          const double &n_burnin, 
                                          const std::string &metric_type
) {
  
                    const int n_params_main = theta_main_vec_ref.rows();
                    Eigen::Matrix<double, -1, 1> Hessian_diag(n_params_main);
                    
                    if (Model_type == "Stan") {
                      
                          Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data( Model_args_as_cpp_struct.model_so_file, 
                                                                                                    Model_args_as_cpp_struct.json_file_path,  
                                                                                                    123);
                          
                          LC_MVP_workspace_struct LC_MVP_ws_struct; // dummy
                          
                          std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs;
                          LC_MVP_ws_structs.resize(1);
                          LC_MVP_ws_structs[0] = LC_MVP_ws_struct;
                          
                          // Compute diagonal Hessian
                          Hessian_diag = num_diff_Hessian_diag_main_given_nuisance( num_diff_e, shrinkage_factor, Model_type,
                                                                                    force_autodiff, force_PartialLog, multi_attemps,
                                                                                    theta_main_vec_ref, theta_us_vec_ref, y_ref, 
                                                                                    Model_args_as_cpp_struct, 
                                                                                    Stan_model_as_cpp_struct, 
                                                                                    LC_MVP_ws_struct);
                          
                          // Destroy Stan model object
                          fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
                      
                    } else {  
                      
                          Stan_model_struct Stan_model_as_cpp_struct; // dummy struct 
                          
                          LC_MVP_workspace_struct LC_MVP_ws_struct;
                          
                          if ((Model_type == "LC_MVP") || (Model_type == "LC_MVOP")) {
                            
                                    const int n_tests = y_ref.cols();
                                    const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
                                    const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
                                    const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
                                    const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
                                    
                                    const int n_corrs = n_class * n_tests * (n_tests - 1) / 2;
                                    
                                    int n_covariates_total, n_covariates_max;
                                    if (n_class > 1) {
                                      const int n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
                                      const int n_covariates_total_d  = n_covariates_per_outcome_vec.row(1).sum();
                                      n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
                                      const int n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
                                      const int n_covariates_max_d  = n_covariates_per_outcome_vec.row(1).maxCoeff();
                                      n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);
                                    } else {
                                      n_covariates_total = n_covariates_per_outcome_vec.sum();
                                      n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
                                    }
                                    
                                    int vec_size;
                                    if      (vect_type == "AVX512") vec_size = 8;
                                    else if (vect_type == "AVX2")   vec_size = 4;
                                    else if (vect_type == "AVX")    vec_size = 2;
                                    else                            vec_size = 1;
                                    
                                    const int N = y_ref.rows();
                                    
                                    ChunkSizeInfo chunk_info = calculate_chunk_sizes( N, 
                                                                                      vec_size, 
                                                                                      n_chunks);
                                    
                                    LC_MVP_ws_struct.allocate( chunk_info.normal_chunk_size,
                                                               n_tests,
                                                               n_class,
                                                               n_covariates_max,
                                                               n_corrs,
                                                               n_covariates_total);
                                    // int padded_chunk = ((chunk_info.normal_chunk_size + vec_size - 1) / vec_size) * vec_size;
                                    // LC_MVP_ws_struct.allocate(  padded_chunk,
                                    //                             n_tests,
                                    //                             n_class,
                                    //                             n_covariates_max,
                                    //                             n_corrs,
                                    //                             n_covariates_total);
                                    
                            
                          } // else: LC_MVP_ws_struct[i].is_allocated stays false (dummy)
                          
                          std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs;
                          LC_MVP_ws_structs.resize(1);
                          LC_MVP_ws_structs[0] = LC_MVP_ws_struct;
                          
                          // Compute diagonal Hessian
                          Hessian_diag = num_diff_Hessian_diag_main_given_nuisance( num_diff_e, shrinkage_factor, Model_type,
                                                                                    force_autodiff, force_PartialLog, multi_attemps,
                                                                                    theta_main_vec_ref, theta_us_vec_ref, y_ref, 
                                                                                    Model_args_as_cpp_struct,
                                                                                    Stan_model_as_cpp_struct, 
                                                                                    LC_MVP_ws_struct);
                      
                    }
                    
                    // Ensure all diagonal elements are positive
                    double epsilon = std::max(1e-8, 1e-6 * Hessian_diag.maxCoeff());
                    for (int i = 0; i < n_params_main; ++i) {
                      if (Hessian_diag(i) < epsilon) {
                        Hessian_diag(i) = epsilon;
                      }
                    } 
                    
                    // Update M_main_vec (diagonal of M)
                    M_main_vec = (1.0 - ratio) * M_main_vec + ratio * Hessian_diag;
                    
                    // Ensure positive
                    for (int i = 0; i < n_params_main; ++i) {
                      if (M_main_vec(i) < epsilon) {
                        M_main_vec(i) = epsilon;
                      } 
                    }
                    
                    // Update M_inv_main_vec (diagonal of M^{-1})
                    M_inv_main_vec = 1.0 / M_main_vec.array();
  
} 






 







