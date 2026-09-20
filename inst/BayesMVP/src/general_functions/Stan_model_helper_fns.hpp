
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
//// #include <RcppParallel.h>
 

 
using namespace Eigen;
 
 
#ifdef _WIN32
#include <windows.h>
#define RTLD_LAZY 0  // Windows doesn't need this flag but define for compatibility
#define dlopen(x,y) LoadLibrary(x) //#define dlopen(x,y) LoadLibraryA(x)
#define dlclose(x) FreeLibrary((HMODULE)x)
#define dlsym(x,y) GetProcAddress((HMODULE)x,y)
#else
#include <dlfcn.h>
#endif

#ifdef _WIN32
inline std::string windows_error_str() {
   
   char error_msg[256];
   DWORD error = GetLastError();
   DWORD flags = FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
   
   FormatMessageA(   flags,
                     NULL,
                     error,
                     MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                     error_msg,
                     sizeof(error_msg),
                     NULL);
   
   return std::string(error_msg);
   
 }
 
#define dlerror() windows_error_str()
#endif



 
 
 
 
 
 
 
std::string normalize_windows_path(const std::string &path) {
   
   std::string normalized = path;
   
   // Remove any leading forward slashes
   while (normalized.size() >= 2 && normalized[0] == '/' && normalized[1] == '/') {
     normalized = normalized.substr(1);
   }
   while (normalized.size() >= 1 && normalized[0] == '/') {
     normalized = normalized.substr(1);
   }
   
   // Replace forward slashes with double backslashes
   std::string result;
   for (char c : normalized) {
     if (c == '/') {
       result += "\\\\";  // Will become \\ in the string
     } else {
       result += c;
     }
   }
   
   //// std::cout << "Original path: " << path << std::endl;
   //// std::cout << "Normalized path: " << result << std::endl;
   
   return result;
   
}
 
 
 
 
 
////  Struct to hold the model handle and function pointers
struct ModelHandle_struct {

   void* bs_handle = nullptr;
   bs_model* (*bs_model_construct)(const char*, unsigned int, char**) = nullptr;
   int (*bs_log_density_gradient)(bs_model*, bool, bool, const double*, double*, double*, char**) = nullptr;
   int (*bs_param_constrain)(bs_model*, bool, bool, const double*, double*, bs_rng*, char**) = nullptr;
   int (*bs_param_unc_num)(const bs_model*) = nullptr;
   bs_rng* (*bs_rng_construct)(unsigned int, char**) = nullptr;
   void (*bs_model_destruct)(bs_model*) = nullptr;
   void (*bs_rng_destruct)(bs_rng*) = nullptr;
   
};
 
 
 
 
 
struct Stan_model_struct {
   
   void* bs_handle = nullptr; // has no arguments
   bs_model* bs_model_ptr = nullptr; // has no arguments
   bs_model* (*bs_model_construct)(const char*, unsigned int, char**) = nullptr;
   int (*bs_log_density_gradient)(bs_model*, bool, bool, const double*, double*, double*, char**) = nullptr;
   int (*bs_param_constrain)(bs_model*, bool, bool, const double*, double*, bs_rng*, char**) = nullptr;
   int (*bs_param_unc_num)(const bs_model*) = nullptr;
   bs_rng* (*bs_rng_construct)(unsigned int, char**) = nullptr;
   void (*bs_model_destruct)(bs_model*) = nullptr;
   void (*bs_rng_destruct)(bs_rng*) = nullptr;
   
};
 
 
 
  
 
 


//// fn to handle JSON via file input and compute the log-prob and gradient
bs_model* fn_convert_JSON_data_to_BridgeStan(ModelHandle_struct &model_handle,
                                             const std::string  &json_file, 
                                             unsigned int seed) {
  
         //// std::cout << "Converting JSON data from: " << json_file << std::endl;
     
         // Load the Stan model from the .so or .dll file using BridgeStan
         char* error_msg = nullptr;
         //  unsigned int seed = seed;
         
         // Use the user-provided JSON file path and construct the bs_model_ptr
         #ifdef _WIN32
             const std::string json_file_normalised = normalize_windows_path(json_file);
             bs_model* bs_model_ptr = model_handle.bs_model_construct(json_file_normalised.c_str(), seed, &error_msg);
         #else
             bs_model* bs_model_ptr = model_handle.bs_model_construct(json_file.c_str(), seed, &error_msg);
         #endif
         
         
         if (bs_model_ptr) {
            //// std::cout << "Model constructed successfully. Pointer: " << bs_model_ptr << std::endl;
           // Add any model info/dimension queries that bridgestan provides
         } else {
            std::cout << "Model construction failed: " << (error_msg ? error_msg : "Unknown error") << std::endl;
         }
         
         if (!bs_model_ptr) {
           throw std::runtime_error("Error constructing the model: " + std::string(error_msg ? error_msg : "Unknown error"));
         } 
         
         
         
         if (!bs_model_ptr) {
           std::cout << "Failed to construct model. Error: " << 
             (error_msg ? error_msg : "Unknown error") << std::endl;
         } else {
            //// std::cout << "Successfully constructed model" << std::endl;
         }
         
         if (!bs_model_ptr) {
           throw std::runtime_error("Error constructing the model: " + std::string(error_msg ? error_msg : "Unknown error"));
         } 
         
         return bs_model_ptr; 
 
}



 
 
// fn to dynamically load the user-provided .so file and resolve symbols
Stan_model_struct fn_load_Stan_model_and_data( const std::string &model_so_file, 
                                               const std::string &json_file,
                                               unsigned int seed) {
  
           //// std::cout << "Loading model from: " << model_so_file << std::endl;
   
           // Load the .so or .dll file
           #ifdef _WIN32
               std::string normalized_path = normalize_windows_path(model_so_file);
               //// std::cout << "Attempting to load from normalized path: " << normalized_path << std::endl;
               
               // // Add the DLL's directory to the search path
               // std::string dll_dir = normalized_path.substr(0, normalized_path.find_last_of('\\'));
               // SetDllDirectoryA(dll_dir.c_str());
               // std::cout << "Adding DLL directory to search path: " << dll_dir << std::endl;
               
               void* bs_handle = LoadLibraryA(normalized_path.c_str());
               
               if (!bs_handle) {
                 DWORD error = GetLastError();
                 std::cout << "LoadLibrary failed with error " << error << ": " << windows_error_str() << std::endl;
                 
                 // // Try to load with full path
                 // std::string full_path = std::filesystem::absolute(normalized_path).string();
                 // std::cout << "Trying full path: " << full_path << std::endl;
                 // bs_handle = LoadLibraryA(full_path.c_str());
               } else {
                  //// std::cout << "Successfully loaded library" << std::endl;
               }
           #else
               void* bs_handle = dlopen(model_so_file.c_str(), RTLD_LAZY);
               if (!bs_handle) {
                 throw std::runtime_error("Error loading .so file: " + std::string(dlerror()));
               } 
           #endif
           
           
           // Resolve the bs_model_construct symbol
           typedef bs_model* (*bs_model_construct_func)(const char*, unsigned int, char**);
           bs_model_construct_func bs_model_construct = (bs_model_construct_func)dlsym(bs_handle, "bs_model_construct"); 
           if (!bs_model_construct) {
             dlclose(bs_handle); 
             throw std::runtime_error("Error loading symbol 'bs_model_construct': " + std::string(dlerror()));
           }  
           
           // Resolve the bs_log_density_gradient symbol 
           typedef int (*bs_log_density_gradient_func)(bs_model*, bool, bool, const double*, double*, double*, char**);
           bs_log_density_gradient_func bs_log_density_gradient = (bs_log_density_gradient_func)dlsym(bs_handle, "bs_log_density_gradient");
           if (!bs_log_density_gradient) { 
             dlclose(bs_handle);
             throw std::runtime_error("Error loading symbol 'bs_log_density_gradient': " + std::string(dlerror()));
           }  
           
           // Resolve the bs_param_constrain symbol 
           typedef int (*bs_param_constrain_func)(bs_model*, bool, bool, const double*, double*, bs_rng*, char**);
           bs_param_constrain_func bs_param_constrain = (bs_param_constrain_func)dlsym(bs_handle, "bs_param_constrain");
           if (!bs_param_constrain) { 
             dlclose(bs_handle);
             throw std::runtime_error("Error loading symbol 'bs_param_constrain': " + std::string(dlerror()));
           } 
           
           // Resolve the bs_param_unc_num symbol (number of UNCONSTRAINED parameters -
           // used to validate that the nuisance + main blocks cover every declared
           // coordinate before constraining draws or evaluating gradients).
           typedef int (*bs_param_unc_num_func)(const bs_model*);
           bs_param_unc_num_func bs_param_unc_num = (bs_param_unc_num_func)dlsym(bs_handle, "bs_param_unc_num");
           if (!bs_param_unc_num) { 
             dlclose(bs_handle);
             throw std::runtime_error("Error loading symbol 'bs_param_unc_num': " + std::string(dlerror()));
           } 
           
           // Resolve the bs_rng_construct symbol
           typedef bs_rng* (*bs_rng_construct_func)(unsigned int, char**);
           bs_rng_construct_func bs_rng_construct = (bs_rng_construct_func)dlsym(bs_handle, "bs_rng_construct"); 
           if (!bs_rng_construct) {
             dlclose(bs_handle); 
             throw std::runtime_error("Error loading symbol 'bs_rng_construct': " + std::string(dlerror()));
           }   
           
           // Resolve the bs_model_destruct symbol
           typedef void (*bs_model_destruct_func)(bs_model*);
           bs_model_destruct_func bs_model_destruct = (bs_model_destruct_func)dlsym(bs_handle, "bs_model_destruct"); 
           if (!bs_model_destruct) {
             dlclose(bs_handle); 
             throw std::runtime_error("Error loading symbol 'bs_model_destruct': " + std::string(dlerror()));
           }    
           
           // Resolve the bs_rng_destruct symbol
           typedef void (*bs_rng_destruct_func)(bs_rng*);
           bs_rng_destruct_func bs_rng_destruct = (bs_rng_destruct_func)dlsym(bs_handle, "bs_rng_destruct"); 
           if (!bs_rng_destruct) {
             dlclose(bs_handle); 
             throw std::runtime_error("Error loading symbol 'bs_rng_destruct': " + std::string(dlerror()));
           }     
           
           ModelHandle_struct model_handle = {bs_handle,
                                              bs_model_construct,
                                              bs_log_density_gradient, 
                                              bs_param_constrain, 
                                              bs_param_unc_num,
                                              bs_rng_construct,
                                              bs_model_destruct,
                                              bs_rng_destruct};
           
           bs_model* bs_model_ptr = fn_convert_JSON_data_to_BridgeStan(model_handle, 
                                                                       json_file,
                                                                       seed);
   
           
           // return {bs_model_ptr, bs_handle, bs_model_construct, bs_log_density_gradient};  // Return handle and fn pointers 
           
           return {bs_handle,                 
                   bs_model_ptr,              
                   bs_model_construct,        
                   bs_log_density_gradient, 
                   bs_param_constrain, 
                   bs_param_unc_num,
                   bs_rng_construct,
                   bs_model_destruct,
                   bs_rng_destruct};   
           
   
 }
 
 
 
 
 
 
 
 
Eigen::Matrix<double, -1, 1> fn_Stan_compute_log_prob_grad(    const Stan_model_struct &Stan_model_as_cpp_struct,  
                                                               const Eigen::Matrix<double, -1, 1> &params,
                                                               const int n_params_main, 
                                                               const int n_nuisance,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, 1>> lp_and_grad_outs) { 
   
             if (!Stan_model_as_cpp_struct.bs_model_ptr || !Stan_model_as_cpp_struct.bs_log_density_gradient) {
               throw std::runtime_error("Model not properly initialized");
             }
             
             const int n_params = params.size();
             if (lp_and_grad_outs.size() != (n_params + 1)) {
               throw std::runtime_error("Output vector size mismatch");
             } 
             
             //// The COMPLETE unconstrained vector (nuisance block first, then main block)
             //// must be passed to BridgeStan: every declared coordinate participates in the
             //// log-density. Previously n_nuisance <= 10 dropped the nuisance block entirely
             //// (and misplaced the gradient), which silently corrupted lp/grad for small
             //// nuisance blocks. n_nuisance == 0 is handled naturally (empty head).
             if (params.size() != (n_nuisance + n_params_main)) {
               throw std::runtime_error("Parameter vector size does not match n_nuisance + n_params_main");
             }
             
             double log_prob_val = 0.0;
             char* error_msg = nullptr;
             
             int result = Stan_model_as_cpp_struct.bs_log_density_gradient(  Stan_model_as_cpp_struct.bs_model_ptr,
                                                                             true,
                                                                             true,
                                                                             params.data(),
                                                                             &log_prob_val,
                                                                             lp_and_grad_outs.segment(1, n_params).data(),
                                                                             &error_msg);
             
             if (result != 0) {
               throw std::runtime_error("Gradient computation failed: " + 
                                        std::string(error_msg ? error_msg : "Unknown error"));
             } 
             
             lp_and_grad_outs(0) = log_prob_val;
             return lp_and_grad_outs;
            
 }
 
 
  
 



#ifdef _WIN32
 
void fn_bs_destroy_Stan_model(Stan_model_struct &Stan_model_as_cpp_struct) {
   
         if (Stan_model_as_cpp_struct.bs_model_ptr && Stan_model_as_cpp_struct.bs_model_destruct) {
           Stan_model_as_cpp_struct.bs_model_destruct(Stan_model_as_cpp_struct.bs_model_ptr);
           Stan_model_as_cpp_struct.bs_model_ptr = nullptr; 
         }
         
         if (Stan_model_as_cpp_struct.bs_handle) {
           
           void* handle = Stan_model_as_cpp_struct.bs_handle;  // Keep a copy
           Stan_model_as_cpp_struct.bs_handle = nullptr;  // Clear it first
           if (FreeLibrary((HMODULE)handle) == 0) {  // Use the copy
             throw std::runtime_error("Error closing library: " + std::string(dlerror()));
           }
           
         }
 
   
}

#else

void fn_bs_destroy_Stan_model(Stan_model_struct &Stan_model_as_cpp_struct) {
  
        if (Stan_model_as_cpp_struct.bs_model_ptr && Stan_model_as_cpp_struct.bs_model_destruct) {
          Stan_model_as_cpp_struct.bs_model_destruct(Stan_model_as_cpp_struct.bs_model_ptr);
          Stan_model_as_cpp_struct.bs_model_ptr = nullptr;
        }
        
        if (Stan_model_as_cpp_struct.bs_handle) {
          
            void* handle = Stan_model_as_cpp_struct.bs_handle;  // Keep a copy
            Stan_model_as_cpp_struct.bs_handle = nullptr;  // Clear it first
            if (dlclose(handle) != 0) {  // Use the copy
              throw std::runtime_error("Error closing library: " + std::string(dlerror()));
            }
          
        }
  
}
 
#endif
  
  
  
  
  
  
 
  
 
 
 
 
 

  
 
  
  