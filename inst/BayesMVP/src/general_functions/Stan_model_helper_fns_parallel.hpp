
#pragma once

 

#include <sstream>
#include <stdexcept>  
#include <complex>

#include <map>
#include <vector>  
#include <string> 
#include <stdexcept>
#include <stdio.h>
#include <iostream>
 
#include <stan/model/model_base.hpp>  
 
#include <stan/io/array_var_context.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/dump.hpp> 

 
#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/rapidjson_parser.hpp>   
 
 




#include <Eigen/Dense>
#include <RcppParallel.h>
 

 
 
 
  
  
static std::mutex copy_mutex;  //// global mutex 
 
  
  
struct ParamConstrainWorker : public RcppParallel::Worker {
    
    //// Inputs
    const int n_threads;
    const std::vector<Eigen::Matrix<double, -1, -1>> unc_params_trace_input_main;
    const std::vector<Eigen::Matrix<double, -1, -1>> unc_params_trace_input_nuisance;
    const std::vector<int> pars_indicies_to_track;
    const int n_params_full;
    const int n_nuisance;
    const int n_params_main;
    const bool include_nuisance;
    const std::string model_so_file;
    const std::string json_file_path;
    
    //// local storage output using tbb container
    tbb::concurrent_vector<RcppParallel::RMatrix<double>> all_param_outs_trace_concurrent;  
    
    //// output
    std::vector<Rcpp::NumericMatrix> &all_param_outs_trace;
    
    // Constructor
    ParamConstrainWorker( const int &n_threads_R_,
                          const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_main_,
                          const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_nuisance_,
                          const std::vector<int>  &pars_indicies_to_track_,
                          const int &n_params_full_,
                          const int &n_nuisance_,
                          const int &n_params_main_,
                          const bool &include_nuisance_,
                          const std::string &model_so_file_,
                          const std::string &json_file_path_,
                          std::vector<Rcpp::NumericMatrix>   &all_param_outs_trace_)
      :
      n_threads(n_threads_R_),
      unc_params_trace_input_main(unc_params_trace_input_main_),
      unc_params_trace_input_nuisance(unc_params_trace_input_nuisance_),
      pars_indicies_to_track(pars_indicies_to_track_),
      n_params_full(n_params_full_),
      n_nuisance(n_nuisance_),
      n_params_main(n_params_main_), 
      include_nuisance(include_nuisance_),
      model_so_file(model_so_file_),
      json_file_path(json_file_path_),
      all_param_outs_trace(all_param_outs_trace_)
    {
          
          const int n_iter = unc_params_trace_input_main[0].cols();
          const int n_params_to_track = pars_indicies_to_track.size();
          
          all_param_outs_trace_concurrent.reserve(n_threads_R_);
          for (int i = 0; i < n_threads_R_; ++i) {
             Rcpp::NumericMatrix Rcpp_mat(n_iter, n_params_to_track);
             RcppParallel::RMatrix<double> mat(Rcpp_mat);
             all_param_outs_trace_concurrent.emplace_back(mat);
          }
      
    }
    
    // Thread-local processing
    void operator()(std::size_t begin, std::size_t end) {
      
      std::cout << "Starting worker for chains " << begin << " to " << end << std::endl;
      
      // Process assigned chains
      std::size_t kk = begin;
      {
        
              char* error_msg = nullptr;
              const int n_iter = unc_params_trace_input_main[0].cols();
              const int n_params = n_nuisance + n_params_main;
              const int n_params_to_track = pars_indicies_to_track.size();
              
              thread_local stan::math::ChainableStack ad_tape;
              thread_local stan::math::nested_rev_autodiff nested;
              
              // Initialize matrix for this chain
              Eigen::Matrix<double, -1, -1> chain_output(n_params_to_track, n_iter);
              Eigen::Matrix<double, -1, 1>  theta_unc_full_input(n_params);
              Eigen::Matrix<double, -1, 1>  theta_constrain_full_output(n_params_full);
              
              Stan_model_struct Stan_model_as_cpp_struct;
              bs_rng* bs_rng_object = nullptr;
              
              // Initialize model and RNG once per thread
              if (model_so_file != "none" && Stan_model_as_cpp_struct.bs_model_ptr == nullptr) {
                Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(model_so_file, json_file_path, 123 + kk);
                bs_rng_object = Stan_model_as_cpp_struct.bs_rng_construct(123 + kk, &error_msg);
              }
              
              if (!bs_rng_object) {
                std::cout << "Failed to create RNG object: " << (error_msg ? error_msg : "Unknown error") << std::endl;
              }
              
              if (!Stan_model_as_cpp_struct.bs_model_ptr) {
                std::cout << "Model pointer is null!" << std::endl;
              } else {
                std::cout << "Model pointer is valid" << std::endl;
              }
              
              for (int ii = 0; ii < n_iter; ii++) {
                
                              theta_unc_full_input.setZero();
                              theta_constrain_full_output.setZero();
                
                              if (ii % 100 == 0) { //// Print every 100th iteration
                                    std::cout << "Processing iteration " << ii << " for chain " << kk << std::endl;
                                 
                                  //// Print values before combining
                                  std::cout << "Main params for iter " << ii << ": "  << unc_params_trace_input_main[kk].col(ii).transpose().head(5) << std::endl; /// this is fine 
                                  
                                  if (include_nuisance) {
                                    std::cout << "Nuisance params for iter " << ii << ": "   << unc_params_trace_input_nuisance[kk].col(ii).transpose().head(5) << std::endl;  /// this is fine 
                                  }
                              
                              }
                                  
                             if (ii % 100 == 0) { //// Print every 100th iteration
                                  std::cout << "Full input size: " << theta_unc_full_input.size() << std::endl;
                                  std::cout << "Full output size: " << theta_constrain_full_output.size() << std::endl;
                                  std::cout << "Params to track size: " << pars_indicies_to_track.size() << std::endl;
                             }
                                  
                                  theta_unc_full_input.tail(n_params_main) = unc_params_trace_input_main[kk].col(ii);
                             
                             if (ii % 100 == 0) { //// Print every 100th iteration
                                  std::cout << "After copying main params: " << theta_unc_full_input.tail(5).transpose() << std::endl;
                             }
                            
                            if (include_nuisance == true) {
                                theta_unc_full_input.head(n_nuisance) = unc_params_trace_input_nuisance[kk].col(ii);
                            } else {
                                theta_unc_full_input.head(n_nuisance).array() = 0.0;
                            }
                            
                            if (ii % 100 == 0) { // Print every 100th iteration
                                // After combining into full input
                                std::cout << "Combined input for iter " << ii << ": " << theta_unc_full_input.transpose().tail(5) << std::endl;  //// NOT OK
                            }
                            
                            int result = Stan_model_as_cpp_struct.bs_param_constrain( Stan_model_as_cpp_struct.bs_model_ptr,
                                                                                      true,
                                                                                      true,
                                                                                      theta_unc_full_input.data(),
                                                                                      theta_constrain_full_output.data(),
                                                                                      bs_rng_object,
                                                                                      &error_msg);
                            
                            if (error_msg) {
                              std::cout << "Error message: " << error_msg << std::endl;
                            }
                            
                            if (result != 0) {
                              throw std::runtime_error("Constraint computation failed: " + std::string(error_msg ? error_msg : "Unknown error"));
                            }
                            
                            if (ii % 100 == 0) { // Print every 100th iteration
                                
                                // After constraining, check if values were actually written
                                std::cout << "Output after constraint op - head: " << theta_constrain_full_output.head(5).transpose() << std::endl;
                                std::cout << "Output after constraint op - tail: " << theta_constrain_full_output.tail(5).transpose() << std::endl;
                              
                                // Check the indices we're using to extract values
                              //  std::cout << "Indices to track: " << pars_indicies_to_track.head(5).transpose() << std::endl;
                              
                              // Show first few indices to track
                              std::cout << "First few indices to track: ";
                              for(int i = 0; i < std::min(5, (int)pars_indicies_to_track.size()); i++) {
                                  std::cout << pars_indicies_to_track[i] << " ";
                              }
                              
                                std::cout << "Constrain operation result: " << result << std::endl;
                                std::cout << "Input values - head: " << theta_unc_full_input.head(5).transpose() << std::endl;
                                std::cout << "Input values - tail: " << theta_unc_full_input.tail(5).transpose() << std::endl;
                                
                            }
                            
                            // chain_output.col(ii) = theta_constrain_full_output(pars_indicies_to_track);
                               chain_output.col(ii) = theta_constrain_full_output;
                            
                            if (ii % 100 == 0) { // Print every 100th iteration
                                std::cout << "Chain output after copy - head: " << chain_output.col(ii).head(5).transpose() << std::endl;
                                std::cout << "Chain output after copy - tail: " << chain_output.col(ii).tail(5).transpose() << std::endl;
                            }
                            
                            if (ii % 100 == 0) { // Print every 100th iteration
                                std::cout << "Sample values for iteration - head " << ii << ": "  << theta_constrain_full_output.head(5).transpose() << std::endl;
                                std::cout << "Sample values for iteration - tail " << ii << ": "  << theta_constrain_full_output.tail(5).transpose() << std::endl;
                            }
                            
                
              }
              
              // Store completed chain
              std::cout << "chain_output size: " << chain_output.rows() << "x" << chain_output.cols() << std::endl;
              std::cout << "all_param_outs_trace_concurrent[kk] size: " << all_param_outs_trace_concurrent[kk].nrow() << "x" << all_param_outs_trace_concurrent[kk].ncol() << std::endl;
              
              {
                 //  std::lock_guard<std::mutex> lock(copy_mutex);
                
                  // all_param_outs_trace[kk].setZero();   
                  for (int ii = 0; ii < n_iter; ii++) {
                    for (int param = 0; param < all_param_outs_trace.nrow(); ++param) {
                        all_param_outs_trace_concurrent[kk](param, ii) = chain_output(param, ii);
                    }
                  }
                 // std::cout << "After final copy, first col: " << all_param_outs_trace_concurrent[kk].column(0)  << std::endl;
                 // std::cout << "After final copy, last col: "  << all_param_outs_trace_concurrent[kk].column(n_iter - 1)  << std::endl;
              }
              
              // Clean up thread-local resources if needed
              if (bs_rng_object != nullptr) {
                 Stan_model_as_cpp_struct.bs_rng_destruct(bs_rng_object); 
              }
              
              // Clean up thread-local resources 
              fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
              
        
      }
      
      
      
    } //// end of parallel operator()
    
    
    // Copy results directly to R matrices
    void copy_results_to_output() {
      
          for (int kk = 0; kk < n_threads; ++kk) {
              for (int ii = 0; ii < all_param_outs_trace[0].ncol(); ++ii) {
                for (int param = 0; param < all_param_outs_trace[0].nrow(); ++param) {
                  all_param_outs_trace[kk](param, ii) = all_param_outs_trace_concurrent[kk](param, ii);
                }
              }
          }
          
    }
    
    
    
  };
  
  
 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  