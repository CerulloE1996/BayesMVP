
#pragma once

 


#include <random>

  

#include <Eigen/Dense>
 
 
 

#include <unsupported/Eigen/SpecialFunctions>
 
 
 
 
using namespace Eigen;
 
 
 
 
 
 


 

 

 
// HMC sampler functions   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 
 
// #ifdef RNG_TYPE_CPP_STD
//  template<typename T = std::mt19937>
// #endif
// #ifdef RNG_TYPE_pcg64
//  template<typename T = pcg64>
// #endif
// #ifdef RNG_TYPE_dqrng_pcg64
//  template<typename T =  Rcpp::XPtr<dqrng::random_64bit_generator>>
// #endif
// ALWAYS_INLINE   void generate_random_std_norm_vec_dqrng(   Eigen::Ref<Eigen::Matrix<double, -1, 1>> std_norm_vec,
//                                                            int n_params, 
//                                                            T &rng) {
//   
//        std::normal_distribution<double> dist(0.0, 1.0); 
//   
//        // Initialise at zero:
//        std_norm_vec.setZero();
//        // Fill vector:
//        for (int d = 0; d < n_params; d++) {
//           double norm_draw = dist(rng);
//           std_norm_vec(d) = norm_draw;
//        }
//    
// }
//  
 
 
 
// Eigen::Matrix<double, -1, 1>  generate_random_std_norm_vec_R(int n_params) {
//          
//          Eigen::Matrix<double, -1, 1> std_norm_vec = Eigen::Matrix<double, -1, 1>::Zero(n_params);
//          
//          for (int d = 0; d < n_params; d++) {
//            std_norm_vec(d) = R::rnorm(0.0, 1.0);
//          }
//          
//          return std_norm_vec;
//    
// } 
 
 
 
 

 
#ifdef RNG_TYPE_CPP_STD
   template<typename T = std::mt19937>
#endif
#ifdef RNG_TYPE_pcg64
   template<typename T = pcg64>
#endif
#ifdef RNG_TYPE_dqrng_pcg64
   template<typename T =  Rcpp::XPtr<dqrng::random_64bit_generator>>
#endif
ALWAYS_INLINE   Eigen::Matrix<double, -1, 1>  generate_random_std_norm_vec( int n_params, 
                                                                            T &rng) {
   
         Eigen::Matrix<double, -1, 1> std_norm_vec = Eigen::Matrix<double, -1, 1>::Zero(n_params);
         
         std::normal_distribution<double> dist(0.0, 1.0); 
         
         // Initialise at zero:
         std_norm_vec.setZero();
         // Fill vector:
         for (int d = 0; d < n_params; d++) {
           double norm_draw = dist(rng);
           std_norm_vec(d) = norm_draw;
         }
         
         return std_norm_vec;
   
}
 
 
 
 


 
 
 
 
 
// double generate_random_tau_ii_R(double tau) {
//    
//        double tau_ii = R::runif(0.0, 2.0 * tau);
//        return tau_ii;
//        
// }
 

 
 
 
#ifdef RNG_TYPE_CPP_STD
 template<typename T = std::mt19937>
#endif
#ifdef RNG_TYPE_pcg64
 template<typename T = pcg64>
#endif
#ifdef RNG_TYPE_dqrng_pcg64
 template<typename T =  Rcpp::XPtr<dqrng::random_64bit_generator>>
#endif
ALWAYS_INLINE  double generate_random_tau_ii(  double tau, 
                                               T &rng) {
  
        std::uniform_real_distribution<double> dist(0.0, 2.0 * tau);
        double tau_ii = dist(rng);
        return tau_ii;

}





 
 
 // int main() {
 //   std::mt19937 g;
 //   std::normal_distribution<double> d;
 //   for (int i = 0; i < 10; ++i) {
 //     g.seed(65472381);
 //     d.reset();
 //     std::cout << "List[65472381] = " << d(g) << "\n";
 //   }
 // }
 
 
 
 
 
 
 
 
 
 
 
 
 

 
 