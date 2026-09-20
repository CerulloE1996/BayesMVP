
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cmath>
using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;


// Groups matrix columns into superchain blocks, computes row-wise mean 
// within each block, and replicates that mean across all columns in the block.
// NOTE: Despite the R function name saying "median", the original R code uses mean().


Eigen::Matrix<double, -1, -1> cpp_fn_grp_mat_cols_by_mean_per_superchain_grp( const Eigen::Matrix<double, -1, -1> &mat,
                                                                              int n_chains_per_superchain
) {
  
          const int n_params = mat.rows();
          const int n_chains = mat.cols();
          const int n_full_superchains = n_chains / n_chains_per_superchain;
          
          Eigen::Matrix<double, -1, -1> out = mat;  // copy
          
          // Process full superchains
          for (int i = 0; i < n_full_superchains; ++i) {
            
                int start_col = i * n_chains_per_superchain;
                // Row-wise mean across columns in this superchain block
                Eigen::Matrix<double, -1, 1> mean_vec = out.block(0, start_col, n_params, n_chains_per_superchain).rowwise().mean();
                // Replicate mean across all columns in this block
                for (int j = 0; j < n_chains_per_superchain; ++j) {
                  out.col(start_col + j) = mean_vec;
                }
            
          } 
          
          // Handle remaining columns
          int remaining_start = n_full_superchains * n_chains_per_superchain;
          int n_remaining = n_chains - remaining_start;
          if (n_remaining > 0) {
            Eigen::Matrix<double, -1, 1> mean_vec = out.block(0, remaining_start, n_params, n_remaining).rowwise().mean();
            for (int j = 0; j < n_remaining; ++j) {
              out.col(remaining_start + j) = mean_vec;
            }
          } 
           
          return out;
  
}


//// -| --------- Building the sampling-phase inits from the burn-in endpoints -------------------------------------------------------
////
//// The OLD scheme assigned sampling chain kk the burn-in column (kk % n_chains_burnin) and then
//// replaced theta_main by the row-wise mean of each superchain BLOCK. With 4 burn-in chains,
//// 180 sampling chains and blocks of 16, every block contained each burn-in chain exactly four
//// times, so:
////
////   * every block mean was the SAME vector (the mean of all four burn-in endpoints), i.e. all
////     180 chains started from ONE point in the main block, and the superchains were not
////     over-dispersed relative to each other at all - which is precisely what nested Rhat needs
////     them to be, and it is computed on the main parameters;
////   * the nuisance vectors were NOT averaged and cycled with period 4, so each superchain of 16
////     contained all four nuisance variants (the opposite of the intended structure), and only
////     FOUR distinct nuisance states existed across all 180 chains - 45 exact duplicates each.
////
//// That duplication is a real sampling defect, not just a diagnostics one: if one burn-in chain
//// ends in a poorly conditioned corner of the nuisance space, all 45 of its descendants inherit
//// it. Measured on a saved N = 10000 run, per-chain acceptance differed significantly across the
//// four inherited groups (Kruskal-Wallis p = 2e-4) and the one near-frozen chain (moving on 18%
//// of iterations, against 57-98% for every other chain) sat in the worst group.
////
//// The scheme below fixes both:
////
////   1. ONE burn-in chain per SUPERCHAIN (not per chain). Every chain inside a superchain shares
////      its main init, and different superchains start from different burn-in endpoints, so
////      nested Rhat compares what it is supposed to compare. No averaging is needed or done.
////   2. The inherited nuisance vector is JITTERED per chain, by nuisance_jitter_scale times the
////      per-coordinate spread of the burn-in endpoints themselves. That removes the 45-way exact
////      duplication at source while keeping each chain in the region its superchain reached.
////      Only the NUISANCE block is jittered, so the main-parameter inits within a superchain stay
////      identical and the nested-Rhat assumption is preserved.
////
//// Over-dispersion across superchains is still bounded by n_chains_burnin: with more superchains
//// than burn-in chains the burn-in endpoints are reused, and the function says so.
////
// [[Rcpp::export]]
List cpp_fn_post_burnin_prep_for_sampling(int n_chains_burnin,
                                          int n_chains_sampling,
                                          int n_superchains,
                                          int n_params_main,
                                          int n_nuisance,
                                          const Eigen::Matrix<double, -1, -1> &theta_main_in,
                                          const Eigen::Matrix<double, -1, -1> &theta_us_in,
                                          double nuisance_jitter_scale = 0.25,
                                          int seed = 123
) {

          // Cap n_superchains
          if (n_superchains > n_chains_sampling) {
            n_superchains = n_chains_sampling;
          }
          if (n_superchains < 1) {
            n_superchains = 1;
          }

          int n_chains_per_superchain = n_chains_sampling / n_superchains;
          if (n_chains_per_superchain < 1) {
            n_chains_per_superchain = 1;
          }

          if (n_superchains > n_chains_burnin) {
            Rcpp::Rcout << "post_burnin_prep: " << n_superchains << " superchains but only "
                        << n_chains_burnin << " burn-in chains, so burn-in endpoints are reused across"
                        << " superchains; between-superchain over-dispersion (and hence nested Rhat's"
                        << " sensitivity) is limited by n_chains_burnin." << std::endl;
          }

          // Allocate sampling matrices
          Eigen::Matrix<double, -1, -1> theta_main_sampling(n_params_main, n_chains_sampling);
          Eigen::Matrix<double, -1, -1> theta_us_sampling = Eigen::Matrix<double, -1, -1>::Zero(n_nuisance, n_chains_sampling);

          //// ---- (1) one burn-in endpoint per SUPERCHAIN. Chains in the trailing partial block
          //// (when n_superchains does not divide n_chains_sampling) join the last superchain.
          for (int kk = 0; kk < n_chains_sampling; ++kk) {

                const int superchain_id = std::min(kk / n_chains_per_superchain, n_superchains - 1);
                const int burnin_col = superchain_id % n_chains_burnin;
                theta_main_sampling.col(kk) = theta_main_in.col(burnin_col);
                if (n_nuisance > 0) {
                  theta_us_sampling.col(kk) = theta_us_in.col(burnin_col);
                }

          }
          //// NOTE: no row-wise averaging across the block any more. Every chain in a superchain is
          //// already assigned the SAME burn-in column, so the old block mean would be a no-op here,
          //// and applying it across superchains would destroy the over-dispersion added above.
          //// cpp_fn_grp_mat_cols_by_mean_per_superchain_grp() is kept above for reference.

          //// ---- (2) jitter the inherited nuisance vector, per chain, so that no two chains start
          //// from the identical 29k-dimensional point. Scale = the per-coordinate SD of the burn-in
          //// endpoints, i.e. the spread the burn-in chains themselves reached, so it is on the right
          //// scale per coordinate without needing the metric passed in.
          if (n_nuisance > 0 && nuisance_jitter_scale > 0.0 && n_chains_burnin >= 2) {

                const Eigen::Matrix<double, -1, 1> mean_us = theta_us_in.rowwise().mean();
                Eigen::Matrix<double, -1, 1> sd_us =
                    (((theta_us_in.colwise() - mean_us).array().square().rowwise().sum())
                         / static_cast<double>(n_chains_burnin - 1)).sqrt();
                //// coordinates on which the burn-in chains agree exactly get no jitter
                for (int j = 0; j < n_nuisance; ++j) {
                  if (!std::isfinite(sd_us(j)) || sd_us(j) < 0.0) sd_us(j) = 0.0;
                }

                if (sd_us.maxCoeff() <= 0.0) {
                  Rcpp::Rcout << "post_burnin_prep: burn-in nuisance endpoints are identical across chains;"
                              << " no jitter applied." << std::endl;
                } else {
                  //// ---- DIMENSION SAFETY CAP on the jitter.
                  ////
                  //// The jitter is iid across coordinates, so its effect on the log density grows
                  //// with the SIZE of the nuisance block, not with the per-coordinate scale alone.
                  //// An iid kick of s * sd on each of d coordinates multiplies E[r^2] by (1 + s^2),
                  //// which displaces the state by
                  ////
                  ////      s^2 * sqrt(d / 2)   standard deviations of the radial distribution.
                  ////
                  //// For the LC-MVP nuisance block d = N * n_tests, so d = 60,000 at N = 10,000.
                  //// The former default s = 0.25 therefore started EVERY sampling chain about
                  //// 10.7 sigma outside the typical set (5.3 sigma at N = 2,500), which produced
                  //// mass divergences during sampling, collapsed ESS and Rhat > 4 - all of it
                  //// attributable to the handover state rather than to the sampler.
                  ////
                  //// The purpose of the jitter is only to break the EXACT duplication of the
                  //// inherited nuisance vector across chains within a superchain, which needs a
                  //// perturbation that is merely nonzero. So cap the implied radial displacement
                  //// and solve back for the per-coordinate scale.
                  ////
                  const double d_us = static_cast<double>(n_nuisance);
                  const double max_radial_sigma = 0.05;   //// never start a chain further off-shell than this
                  const double s_cap = std::sqrt(max_radial_sigma / std::sqrt(0.5 * d_us));
                  ////
                  double jitter_scale_effective = nuisance_jitter_scale;
                  if (jitter_scale_effective > s_cap) {
                        Rcpp::Rcout << "post_burnin_prep: nuisance_jitter_scale = " << nuisance_jitter_scale
                                    << " would displace each sampling chain "
                                    << (nuisance_jitter_scale * nuisance_jitter_scale * std::sqrt(0.5 * d_us))
                                    << " sigma off the typical set with n_nuisance = " << n_nuisance
                                    << "; capping it at " << s_cap
                                    << " (<= " << max_radial_sigma << " sigma)." << std::endl;
                        jitter_scale_effective = s_cap;
                  }
                  Rcpp::Rcout << "post_burnin_prep: effective nuisance jitter scale = "
                              << jitter_scale_effective << " ("
                              << (jitter_scale_effective * jitter_scale_effective * std::sqrt(0.5 * d_us))
                              << " sigma off-shell, n_nuisance = " << n_nuisance << ")." << std::endl;
                  //// one stream per chain, widely separated seeds (a small seed offset leaves
                  //// mt19937-family streams poorly decorrelated at the start)
                  for (int kk = 0; kk < n_chains_sampling; ++kk) {
                        std::mt19937_64 rng(static_cast<uint64_t>(seed)
                                            + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(kk + 1));
                        std::normal_distribution<double> rnorm(0.0, 1.0);
                        for (int j = 0; j < n_nuisance; ++j) {
                          if (sd_us(j) > 0.0) {
                            theta_us_sampling(j, kk) += jitter_scale_effective * sd_us(j) * rnorm(rng);
                          }
                        }
                  }
                }

          }

          return List::create(
            Named("theta_main_vectors_all_chains_input_from_R") = theta_main_sampling,
            Named("theta_us_vectors_all_chains_input_from_R") = theta_us_sampling
          );

}
















