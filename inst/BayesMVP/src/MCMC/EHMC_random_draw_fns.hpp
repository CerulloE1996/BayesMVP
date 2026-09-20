
#pragma once

 


#include <random>
#include <cstdint>

  

#include <Eigen/Dense>
 
 
 

#include <unsupported/Eigen/SpecialFunctions>
 
 
 
 
using namespace Eigen;
 
 
 
 
 
 


 

 

 
// HMC sampler functions   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Burn-in calls the single-thread loop with one iteration at a time. Its incoming seed
// already includes iteration + chain, so adding them again cannot distinguish pairs such
// as (iteration 2, chain 1) and (iteration 1, chain 2). Seed from separate tuple fields.
template<typename T>
inline void fn_seed_burnin_rng(T &rng,
                               const int incoming_chain_seed,
                               const int chain_id,
                               const int iteration,
                               const std::uint32_t parameter_block) {
    std::seed_seq seed_sequence{static_cast<std::uint32_t>(incoming_chain_seed),
                               static_cast<std::uint32_t>(chain_id),
                               static_cast<std::uint32_t>(iteration),
                               parameter_block, std::uint32_t{0x4255524e}};
#if RNG_TYPE_CPP_STD == 1
    rng.seed(seed_sequence);
#elif RNG_TYPE_dqrng_xoshiro256plusplus == 1
    std::uint32_t seed_words[2];
    seed_sequence.generate(seed_words, seed_words + 2);
    const std::uint64_t combined_seed = (static_cast<std::uint64_t>(seed_words[0]) << 32) | seed_words[1];
    rng.seed(combined_seed);
#endif
}

 
 


template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE   Eigen::Matrix<double, -1, 1>  generate_random_std_norm_vec( int n_params, 
                                                                            T &rng) {
   
         //// Initialise at zero:
         Eigen::Matrix<double, -1, 1> std_norm_vec = Eigen::Matrix<double, -1, 1>::Zero(n_params);
         
         std::normal_distribution<double> dist(0.0, 1.0); 
         
         //// Fill vector:
         for (int d = 0; d < n_params; d++) {
             double norm_draw = dist(rng);
             std_norm_vec(d) = norm_draw;
         }
         
         return std_norm_vec;
   
}
 
 
 
 
 
 
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE    void generate_random_std_norm_vec_InPlace( Eigen::Ref<Eigen::Matrix<double, -1, 1>> std_norm_vec,
                                                            T &rng) {
  
         const int n_params = std_norm_vec.size();
    
         std::normal_distribution<double> dist(0.0, 1.0); 
         
         //// Initialise at zero:
         std_norm_vec.setZero();
         
         //// Fill vector:
         for (int d = 0; d < n_params; d++) {
             double norm_draw = dist(rng);
             std_norm_vec(d) = norm_draw; 
         }
   
}
 
 
 
  
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE  double generate_random_std_uniform(T &rng) {
  
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const double rand_std_unif = dist(rng);
        return rand_std_unif;
  
}


 
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE  double generate_random_tau_ii(  double tau, 
                                               T &rng) {
  
        std::uniform_real_distribution<double> dist(0.0, 2.0 * tau);
        const double tau_ii = dist(rng);
        return tau_ii;

}





 
 
 

 
 
 
 
 
 
 
 
 
 

 
 
