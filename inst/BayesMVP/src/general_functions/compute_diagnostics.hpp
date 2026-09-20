
#pragma once


 


#include <tbb/concurrent_vector.h>
 
#include <RcppParallel.h>
 
#include <Eigen/Dense>
 

 
#include <Rcpp.h>
 

#if defined(__AVX2__) || defined(__AVX512F__) 
#include <immintrin.h>
#endif

  
// #include <stan/analyze/mcmc/autocovariance.hpp>
// 
// /// #include <stan/analyze/mcmc/check_chains.hpp>  ////
// #include "Stan_check_chains_copy.hpp"
// 
// //// #include <stan/analyze/mcmc/rank_normalization.hpp>
// #include "Stan_rank_normalization_copy.hpp"
//  
// #include <stan/analyze/mcmc/compute_potential_scale_reduction.hpp>
// #include <stan/analyze/mcmc/compute_effective_sample_size.hpp>
//  
// // #include <stan/analyze/mcmc/split_chains.hpp>
//  
// ///#include <stan/analyze/mcmc/split_rank_normalized_rhat.hpp> //// 
// #include "Stan_split_rhat_copy.hpp"
//  
// /// #include <stan/analyze/mcmc/split_rank_normalized_ess.hpp>  //// 
// #include "Stan_split_ess_copy.hpp"
 

#include <stan/analyze/mcmc/autocovariance.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <stan/analyze/mcmc/compute_effective_sample_size.hpp>
#include <stan/analyze/mcmc/compute_potential_scale_reduction.hpp>
 
#include <stan/analyze/mcmc/split_rank_normalized_ess.hpp>
#include <stan/analyze/mcmc/split_rank_normalized_rhat.hpp>

 
#include <limits>
#include <utility>
 
 
 

using namespace Eigen;
using namespace Rcpp;


  
  
// #include <stan/math/prim.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>
  
  
  
  
//// Note: The classic functions (compute_effective_sample_size, compute_potential_scale_reduction, and their split variants) date from Stan's original analyze module, 
//// when chains were passed around as raw const double* arrays with a matching sizes vector — the C-style interface CmdStan's stansummary originally used to read CSV 
//// columns without copying. The rank-normalised versions were added much later (Stan 2.33, 2023), after the codebase had moved to Eigen types, 
//// so they take a MatrixXd (draws × chains) directly and do the split/rank/fold internally.


// Individual constant chains/half-chains must remain in the calculation. Only a
// nonfinite or globally constant ensemble is undefined (posterior's convention).
inline bool fn_diagnostic_ensemble_varies(const Eigen::MatrixXd &draws) {
    return draws.size() > 0 && draws.allFinite()
           && draws.maxCoeff() - draws.minCoeff() >= std::numeric_limits<double>::epsilon();
}

inline Eigen::MatrixXd fn_split_diagnostic_chains(const Eigen::MatrixXd &draws) {
    const Eigen::Index half = draws.rows() / 2;
    Eigen::MatrixXd split(half, 2 * draws.cols());
    split.leftCols(draws.cols()) = draws.topRows(half);
    split.rightCols(draws.cols()) = draws.bottomRows(half);
    return split; // For an odd length, omit the middle draw, as posterior does.
}

// Keep zero-variance chains with zero autocovariance, not 0/0. Use biased
// autocovariances (lag denominator n) and posterior's positive/monotone sequence
// truncation. The local Stan Math autocovariance uses denominator n - lag.
inline double fn_ensemble_ESS(const Eigen::MatrixXd &draws) {
    const Eigen::Index n = draws.rows();
    const Eigen::Index chains = draws.cols();
    if (n < 3 || !fn_diagnostic_ensemble_varies(draws)) return std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd mean_acov = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd chain_means(chains);
    Eigen::FFT<double> fft;
    for (Eigen::Index chain = 0; chain < chains; ++chain) {
        const Eigen::VectorXd values = draws.col(chain);
        chain_means(chain) = values.mean();
        if (values.maxCoeff() == values.minCoeff()) continue;
        Eigen::VectorXd acov(n);
        stan::math::autocovariance<double>(values, acov, fft);
        for (Eigen::Index lag = 0; lag < n; ++lag) acov(lag) *= static_cast<double>(n - lag) / n;
        mean_acov += acov;
    }
    mean_acov /= chains;
    const double within = mean_acov(0) * n / (n - 1);
    const double between = chains > 1 ? (chain_means.array() - chain_means.mean()).square().sum() / (chains - 1) : 0.0;
    const double variance_plus = mean_acov(0) + between;
    Eigen::VectorXd rho = Eigen::VectorXd::Zero(n);
    double even = 1.0;
    double odd = 1.0 - (within - mean_acov(1)) / variance_plus;
    rho(0) = even;
    rho(1) = odd;
    Eigen::Index t = 0;
    while (t < n - 5 && !std::isnan(even + odd) && even + odd > 0.0) {
        t += 2;
        even = 1.0 - (within - mean_acov(t)) / variance_plus;
        odd = 1.0 - (within - mean_acov(t + 1)) / variance_plus;
        if (even + odd >= 0.0) {
            rho(t) = even;
            rho(t + 1) = odd;
        }
    }
    const Eigen::Index max_t = t;
    if (even > 0.0) rho(max_t) = even;
    for (t = 2; t <= max_t - 2; t += 2) {
        if (rho(t) + rho(t + 1) > rho(t - 2) + rho(t - 1)) {
            rho(t) = 0.5 * (rho(t - 2) + rho(t - 1));
            rho(t + 1) = rho(t);
        }
    }
    const double total = static_cast<double>(n) * chains;
    const double tau = -1.0 + 2.0 * rho.head(std::max<Eigen::Index>(1, max_t)).sum() + rho(max_t);
    return total / std::max(tau, 1.0 / std::log10(total));
}

//// Rank-normalised split-ESS (Vehtari et al. 2021): returns (bulk_ESS, tail_ESS)
inline std::pair<double, double> compute_Stan_split_ESS_rank(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> mcmc_array) {
    const double undefined = std::numeric_limits<double>::quiet_NaN();
    if (mcmc_array.rows() < 8 || mcmc_array.cols() == 0 || !mcmc_array.allFinite()) return {undefined, undefined};
    const Eigen::MatrixXd draws = mcmc_array;
    const Eigen::MatrixXd ranked = stan::analyze::rank_transform(fn_split_diagnostic_chains(draws));
    const double bulk = fn_ensemble_ESS(ranked);
    if (!fn_diagnostic_ensemble_varies(draws)) return {bulk, undefined};
    const Eigen::MatrixXd lower = fn_split_diagnostic_chains(
        (draws.array() <= stan::math::quantile(draws.reshaped(), 0.05)).cast<double>());
    const Eigen::MatrixXd upper = fn_split_diagnostic_chains(
        (draws.array() <= stan::math::quantile(draws.reshaped(), 0.95)).cast<double>());
    const double lower_ess = fn_ensemble_ESS(lower);
    const double upper_ess = fn_ensemble_ESS(upper);
    const double tail = std::isnan(lower_ess) || std::isnan(upper_ess) ? undefined : std::min(lower_ess, upper_ess);
    return {bulk, tail};
}


//// Rank-normalised split-Rhat: returns (bulk_rhat, tail_rhat); report max of the two.
inline std::pair<double, double> compute_Stan_split_Rhat_rank(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> mcmc_array) {
    const double undefined = std::numeric_limits<double>::quiet_NaN();
    if (mcmc_array.rows() < 4 || mcmc_array.cols() == 0 || !mcmc_array.allFinite()) return {undefined, undefined};
    const Eigen::MatrixXd draws = mcmc_array;
    const Eigen::MatrixXd ranked = stan::analyze::rank_transform(fn_split_diagnostic_chains(draws));
    const Eigen::MatrixXd folded = (draws.array() - stan::math::quantile(draws.reshaped(), 0.5)).abs();
    const Eigen::MatrixXd ranked_folded = stan::analyze::rank_transform(fn_split_diagnostic_chains(folded));
    const double bulk = fn_diagnostic_ensemble_varies(ranked) ? stan::analyze::rhat(ranked) : undefined;
    const double tail = fn_diagnostic_ensemble_varies(ranked_folded) ? stan::analyze::rhat(ranked_folded) : undefined;
    return {bulk, tail};
}


inline double   compute_Stan_ESS(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> mcmc_array) {
  
          const int n_iter = mcmc_array.rows();
          const int n_chains = mcmc_array.cols();
          
          // Create a vector of pointers to each column (chain)
          std::vector<const double*> draws;
          for (int chain = 0; chain < n_chains; ++chain) {
            draws.push_back(mcmc_array.col(chain).data());
          } 
          
          // Create a sizes vector (each chain has the same size in this example)
          std::vector<size_t> sizes(n_chains, n_iter);
           
          double out = stan::analyze::compute_effective_sample_size(draws, sizes);
 
          return out;
      
}


inline double  compute_Stan_Rhat(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> mcmc_array) {
  
          const int n_iter = mcmc_array.rows();
          const int n_chains = mcmc_array.cols();
          
          // Create a vector of pointers to each column (chain)
          std::vector<const double*> draws;
          for (int chain = 0; chain < n_chains; ++chain) {
            draws.push_back(mcmc_array.col(chain).data());
          } 
          
          // Create a sizes vector (each chain has the same size in this example)
          std::vector<size_t> sizes(n_chains, n_iter);
          
          double out = stan::analyze::compute_potential_scale_reduction(draws, sizes);
          
          return out;
  
}


inline double   compute_Stan_split_ESS(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> mcmc_array) {
  
          const int n_iter = mcmc_array.rows();
          const int n_chains = mcmc_array.cols();
          
          // Create a vector of pointers to each column (chain)
          std::vector<const double*> draws;
          for (int chain = 0; chain < n_chains; ++chain) {
            draws.push_back(mcmc_array.col(chain).data());
          } 
           
          // Create a sizes vector (each chain has the same size in this example)
          std::vector<size_t> sizes(n_chains, n_iter);
          
          double out = stan::analyze::compute_split_effective_sample_size(draws, sizes);
          
          return out;
  
}


inline double  compute_Stan_split_Rhat(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> mcmc_array) {
          
          const int n_iter = mcmc_array.rows();
          const int n_chains = mcmc_array.cols();
          
          // Create a vector of pointers to each column (chain)
          std::vector<const double*> draws;
          for (int chain = 0; chain < n_chains; ++chain) {
            draws.push_back(mcmc_array.col(chain).data());
          } 
          
          // Create a sizes vector (each chain has the same size in this example)
          std::vector<size_t> sizes(n_chains, n_iter);
          
          double out = stan::analyze::compute_split_potential_scale_reduction(draws, sizes);
          
          return out;
  
} 



// inline std::pair<double, double>   compute_Stan_split_Rhat_rank(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> mcmc_array) {
//   
//   const int n_iter = mcmc_array.rows();
//   const int n_chains = mcmc_array.cols();
//   
//   // Create a vector of pointers to each column (chain)
//   std::vector<const double*> draws;
//   for (int chain = 0; chain < n_chains; ++chain) {
//     draws.push_back(mcmc_array.col(chain).data());
//   } 
//   
//   // Create a sizes vector (each chain has the same size in this example)
//   std::vector<size_t> sizes(n_chains, n_iter);
//   
//   std::pair<double, double> out = stan::analyze::compute_potential_scale_reduction_rank(draws, sizes);
//   
//   return out;
//   
// }  



 
 


  
inline std::pair<double, double> Stan_compute_diagnostic( const std::string &diagnostic, 
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> mcmc_array) {
      
            if (diagnostic == "ESS") {
              double ess = compute_Stan_ESS(mcmc_array);
              return std::make_pair(ess, 0.0);
            }
            if (diagnostic == "rhat") {
              double rhat = compute_Stan_Rhat(mcmc_array);
              return std::make_pair(rhat, 0.0);
            }
            if (diagnostic == "split_ESS") {
              double ess = compute_Stan_split_ESS(mcmc_array);
              return std::make_pair(ess, 0.0);
            }
            if (diagnostic == "split_rhat") {
              double rhat = compute_Stan_split_Rhat(mcmc_array);
              return std::make_pair(rhat, 0.0);
            }
            if (diagnostic == "split_ESS_rank") {
              return compute_Stan_split_ESS_rank(mcmc_array);   //// (bulk, tail)
            }
            if (diagnostic == "split_rhat_rank") {
              return compute_Stan_split_Rhat_rank(mcmc_array);  //// (bulk, tail)
            }

            return std::make_pair(0.0, 0.0);
    
}







//// RcppParallel worker 
struct ComputeDiagnosticParallel : public RcppParallel::Worker {
 
       const int n_params;
       const std::string diagnostic;
       
       /// uses tbb container for input:
       tbb::concurrent_vector<Eigen::Matrix<double, -1, -1>> mcmc_3D_array;
       
       /// use RMatrix for output 
       RcppParallel::RMatrix<double> output;
       
       //// constructor
       ComputeDiagnosticParallel(const int &n_params_,
                                 const std::string &diagnostic_,
                                 const std::vector<Eigen::Matrix<double, -1, -1>> &mcmc_3D_array_,
                                 Rcpp::NumericMatrix &output_)
         : n_params(n_params_),
           diagnostic(diagnostic_),
           output(output_) 
         {
                 // Initialize concurrent vector:
                 mcmc_3D_array = convert_std_vec_to_concurrent_vector(mcmc_3D_array_, mcmc_3D_array);
         }
       
       //// Parallel operator
       void operator()(std::size_t begin, std::size_t end) {
         
             for (std::size_t i = begin; i < end; ++i) {
               
                     std::pair<double, double> result = Stan_compute_diagnostic(diagnostic, mcmc_3D_array[i]);
                     output(i, 0) = result.first;
                     output(i, 1) = result.second;
                 
             }
             
       }
 
};  






inline std::vector<double> compute_chain_stats(const std::string &stat_type,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> chain_data) {
  
  std::vector<double> output(3, 0.0);  // Initialize with 3 zeros
  
  if (stat_type == "mean") {
    
    const Eigen::Matrix<double, -1, 1> &means = chain_data.colwise().mean();
    output[0] = means.mean();
    
    return output;
    
  }
  
  if (stat_type == "sd") {
    
          const Eigen::Matrix<double, -1, 1> &means = chain_data.colwise().mean();
          const Eigen::Matrix<double, -1, 1> &sds = ((chain_data.rowwise() - means.transpose()).array().square().colwise().sum() 
                                   / (chain_data.rows() - 1)).sqrt();
          output[0] = sds.mean();
          
          return output;
    
  }
  
  if (stat_type == "quantiles") {
    
          const int n_chains = chain_data.cols();
          Eigen::Matrix<double, -1, 1> q025(n_chains), q50(n_chains), q975(n_chains);
          
          for (int i = 0; i < n_chains; ++i) {
            
                Eigen::Matrix<double, -1, 1> sorted = chain_data.col(i);
                std::sort(sorted.data(), sorted.data() + sorted.size());
                const int n = sorted.size();
                
                const int idx025 = static_cast<int>(std::floor(0.025 * (n-1)));
                const int idx50 = static_cast<int>(std::floor(0.5 * (n-1)));
                const int idx975 = static_cast<int>(std::floor(0.975 * (n-1)));
                
                q025(i) = sorted(idx025);
                q50(i) = sorted(idx50);
                q975(i) = sorted(idx975);
                
          }
          
          output[0] = q025.mean();
          output[1] = q50.mean();
          output[2] = q975.mean();
          
          return output;
    
  }
  
  return output;
  
}



struct ComputeStatsParallel : public RcppParallel::Worker {
  
        const int n_params;
        const std::string stat_type;
        
        tbb::concurrent_vector<Eigen::Matrix<double, -1, -1>> mcmc_3D_array;
        
        RcppParallel::RMatrix<double> output;
        
        //// constructor
        ComputeStatsParallel(const int &n_params_,
                             const std::string &stat_type_,
                             const std::vector<Eigen::Matrix<double, -1, -1>> &mcmc_3D_array_,
                             Rcpp::NumericMatrix &output_)
          : n_params(n_params_),
            stat_type(stat_type_),
            output(output_) 
        {
              // Initialize concurrent vector:
              // mcmc_3D_array = convert_vec_of_RcppMat_to_concurrent_vector(mcmc_3D_array_, mcmc_3D_array);
              mcmc_3D_array = convert_std_vec_to_concurrent_vector(mcmc_3D_array_, mcmc_3D_array);
        }
        
        //// Parallel operator
        void operator()(std::size_t begin, std::size_t end) {
          
              for (std::size_t i = begin; i < end; ++i) {
                    
                    ////  const Eigen::Matrix<double, -1, -1> mcmc_array_Eigen = fn_convert_RMatrix_to_Eigen(mcmc_3D_array[i]);
                    const Eigen::Matrix<double, -1, -1> mcmc_array_Eigen = mcmc_3D_array[i];
                    std::vector<double> result = compute_chain_stats(stat_type, mcmc_array_Eigen);
                    
                    for(int j = 0; j < result.size(); ++j) {
                      output(i, j) = result[j];
                    }
                
              }
          
        }
  
};























































