
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

  
// static std::mutex copy_mutex;  //// global mutex 


struct ParamConstrainWorker : public RcppParallel::Worker {
    
    //// Inputs
    const int n_threads;
    const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_main;
    const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_nuisance;
    const int n_params_full;
    const int n_params_main;
    const int n_nuisance;
    const std::string model_so_file;
    const std::string json_file_path;
    
    //// Disk mode
    const bool use_disk;
    const std::string trace_dir;
    // REMOVED: std::vector<std::ofstream> output_files;  // Can't copy ofstream!
    
    //// SHARED Stan model (loaded once in constructor)
    Stan_model_struct shared_Stan_model;
    
    //// Output (only used if use_disk = false)
    // std::vector<Eigen::Matrix<double, -1, -1>> &all_param_outs_trace;
    std::vector<Eigen::Matrix<double, -1, -1>> *all_param_outs_trace_ptr;
    
    //// Constructor
    ParamConstrainWorker( const int &n_threads_R_,
                          const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_main_,
                          const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_nuisance_,
                          const int &n_params_full_,
                          const int &n_params_main_,
                          const int &n_nuisance_,
                          const std::string &model_so_file_,
                          const std::string &json_file_path_,
                          std::vector<Eigen::Matrix<double, -1, -1>> &all_param_outs_trace_,
                          const bool use_disk_,
                          const std::string& trace_dir_)
      :
      n_threads(n_threads_R_),
      unc_params_trace_input_main(unc_params_trace_input_main_),
      unc_params_trace_input_nuisance(unc_params_trace_input_nuisance_),
      n_params_full(n_params_full_),
      n_params_main(n_params_main_),
      n_nuisance(n_nuisance_),
      model_so_file(model_so_file_),
      json_file_path(json_file_path_),
      all_param_outs_trace_ptr(&all_param_outs_trace_),
      use_disk(use_disk_),
      trace_dir(trace_dir_)
    {
          // Load model ONCE
          if (model_so_file != "none") {
            shared_Stan_model = fn_load_Stan_model_and_data(model_so_file, json_file_path, 123);
          }
    }
    
    // // Destructor - only clean up model (files are closed in operator())
    // ~ParamConstrainWorker() {
    //   if (shared_Stan_model.bs_model_ptr != nullptr) {
    //     fn_bs_destroy_Stan_model(shared_Stan_model);
    //   }
    // }
    
    // NO destructor - model cleanup happens in calling function!
    // Default destructor is fine
    
    // Thread-local processing
    void operator()(std::size_t begin, std::size_t end) {
      
      // std::cout << "A: entered operator" << std::flush;
      
      // Process assigned chains
      for (std::size_t kk = begin; kk < end; ++kk) {
        
              // std::cout << "Chain " <<  kk + 1  << " / " << n_threads << std::endl;
              
              char* error_msg = nullptr;
        
              // Thread-local model instance
              bs_model* local_model = shared_Stan_model.bs_model_construct(json_file_path.c_str(), 123, &error_msg);
              
              const int n_iter = unc_params_trace_input_main[0].cols();
              
              // std::cout << "B: chain=" << kk << " n_iter=" << n_iter 
              //           << " n_params_full=" << n_params_full 
              //           << " n_params_main=" << n_params_main << std::flush;
              
              const int n_params_output = n_params_full;
            
              //// ---- COMPLETE unconstrained draw = [nuisance (n_nuisance) | main (n_params_main)].
              //// BridgeStan's constrain expects the full vector; a main-only vector would
              //// be silently re-interpreted as the first n_params_main coordinates.
              Eigen::Matrix<double, -1, 1> theta_unc_input(n_nuisance + n_params_main);
              Eigen::Matrix<double, -1, 1> theta_constrain_full_output(n_params_full);
              Eigen::Matrix<double, -1, 1> tracked_params(n_params_full);  // temp buffer for disk write
              
              //// validate against the model's OWN unconstrained dimension:
              int model_unc_num = -1;
              if (shared_Stan_model.bs_model_ptr && shared_Stan_model.bs_param_unc_num) {
                model_unc_num = shared_Stan_model.bs_param_unc_num(shared_Stan_model.bs_model_ptr);
              }
              if (model_unc_num >= 0 && model_unc_num != (n_nuisance + n_params_main)) {
                throw std::runtime_error("fn_compute_param_constrain_from_trace_parallel: n_nuisance ("
                                         + std::to_string(n_nuisance) + ") + n_params_main ("
                                         + std::to_string(n_params_main) + ") = "
                                         + std::to_string(n_nuisance + n_params_main)
                                         + " does not match the model's unconstrained dimension ("
                                         + std::to_string(model_unc_num) + ").");
              }
              
              // OPEN FILE HERE - inside operator(), per chain
              std::ofstream output_file;
              if (use_disk) {
                std::string filepath = trace_dir + "/chain_" + std::to_string(kk) + "_constrained.bin";
                output_file.open(filepath, std::ios::binary);
                if (!output_file.is_open()) {
                  throw std::runtime_error("Failed to open file: " + filepath);
                }
              } 
              
              // Create thread-local RNG using the SHARED model
              bs_rng* bs_rng_object = nullptr;
              if (local_model != nullptr) {
                bs_rng_object = shared_Stan_model.bs_rng_construct(123 + kk, &error_msg);
              } 
              
              for (int ii = 0; ii < n_iter; ii++) {
                
                        // if (ii % 500 == 0) Rcpp::Rcout << "C: iter=" << ii << std::endl;
                
                        // std::cout << "C: iter=" << ii << std::flush;
                
                        theta_constrain_full_output.setZero();
                
                        // // Get the column for this iteration from this chain's matrix
                        // theta_unc_input = unc_params_trace_input_main[kk].col(ii);
                        
                        // Force a copy with .eval() or explicit loop - COMPLETE vector:
                        // nuisance block first, main block second. An EMPTY nuisance trace
                        // (n_nuisance_to_track = 0) means the block was not stored: zero-fill it
                        // (main.cpp validated the shape and printed the caveat).
                        if (unc_params_trace_input_nuisance.size() > 0) {
                          for (int p = 0; p < n_nuisance; ++p) {
                            theta_unc_input(p) = unc_params_trace_input_nuisance[kk](p, ii);
                          }
                        } else {
                          theta_unc_input.head(n_nuisance).setZero();
                        }
                        for (int p = 0; p < n_params_main; ++p) {
                          theta_unc_input(n_nuisance + p) = unc_params_trace_input_main[kk](p, ii);
                        }
                        
                        // Use SHARED model pointer
                        int result = shared_Stan_model.bs_param_constrain(  local_model,
                                                                            true,
                                                                            true,
                                                                            theta_unc_input.data(),
                                                                            theta_constrain_full_output.data(), 
                                                                            bs_rng_object,
                                                                            &error_msg);
                        
                        // if (ii % 500 == 0) std::cout <<  "D: constrain done" << std::flush;
                        
                        if (result != 0) {
                          throw std::runtime_error("Constraint computation failed: " + 
                                                   std::string(error_msg ? error_msg : "Unknown error"));
                        }
                        
                        if (use_disk) {
                            for (int p = 0; p < n_params_full; ++p) {
                              tracked_params(p) = theta_constrain_full_output(p);
                            }
                            output_file.write(
                              reinterpret_cast<const char*>(tracked_params.data()),
                              n_params_full * sizeof(double));
                        } else {
                            // for (int p = 0; p < n_params_full; ++p) {
                            //   all_param_outs_trace[kk](p, ii) = theta_constrain_full_output(p);
                            // }
                            // Use pointer dereference
                            for (int p = 0; p < n_params_full; ++p) {
                              (*all_param_outs_trace_ptr)[kk](p, ii) = theta_constrain_full_output(p);
                            }
                        }
                        
                        // if (ii % 500 == 0) std::cout << "E: write done"<< std::flush;
                        
              }
              
              // std::cout << "F: ii loop done" << std::flush;
              
              // Close file for this chain
              if (use_disk && output_file.is_open()) {
                output_file.close();
              }
              
              // Clean up RNG
              if (bs_rng_object != nullptr) {
                shared_Stan_model.bs_rng_destruct(bs_rng_object);
              } 
              
              // Clean up thread-local model
              if (local_model != nullptr) {
                shared_Stan_model.bs_model_destruct(local_model);
              }
              
              // std::cout << "G: chain done" << std::flush;
              
      }
      

      
      // std::cout << "H: cleaned up thread-local model" << std::flush;
       
    } //// end of parallel operator()
    
    // void copy_results_to_output( std::vector<Rcpp::NumericMatrix> &output) {
    //       
    //       if (use_disk) {
    //          return; // Data is on disk, nothing to copy
    //       }
    //       
    //       const int n_iter = all_param_outs_trace[0].cols();
    //       
    //       for (int i = 0; i < n_threads; ++i) {
    //         for (int ii = 0; ii < n_iter; ++ii) {
    //           for (int p = 0; p < n_params_to_track; ++p) {
    //             output[i](p, ii) = all_param_outs_trace[i](p, ii);
    //           }
    //         }
    //       }
    //   
    // }
    
    void copy_results_to_output(std::vector<Rcpp::NumericMatrix> &output) {
      
          // std::cout << "I: Calling copy_results_to_output" << std::flush;
      
          if (use_disk) return;
          
          const int n_iter = (*all_param_outs_trace_ptr)[0].cols();
          for (int i = 0; i < n_threads; ++i) {
            for (int ii = 0; ii < n_iter; ++ii) {
              for (int p = 0; p < n_params_full; ++p) {
                output[i](p, ii) = (*all_param_outs_trace_ptr)[i](p, ii);
              }
            }
          }
          
          // std::cout << "J: Finished running copy_results_to_output" << std::flush;
      
    }
    
    // Method to clean up model - call AFTER parallelFor!
    void cleanup_model() {
      
          if (shared_Stan_model.bs_model_ptr != nullptr) {
                fn_bs_destroy_Stan_model(shared_Stan_model);
                shared_Stan_model.bs_model_ptr = nullptr;
          }
      
    }
    
    bool is_using_disk() const { return use_disk; }
    std::string get_trace_dir() const { return trace_dir; }
    
};
  
 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  