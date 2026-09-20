#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp17)]]

using namespace Rcpp;
using namespace Eigen;


//' Compute Se/Sp at custom baseline covariate values for LC-MVOP models
//' (mixed binary + ordinal latent class multivariate probit).
//'
//' All internal storage and computation uses Eigen.
//' Output is reshaped back into R arrays at the end.
//'
//' Se/Sp formulas (probit link):
//'   Binary test t:
//'     Se      = Phi( X_d' * beta[2][, t] )
//'     Fp      = Phi( X_nd' * beta[1][, t] )
//'   Ordinal test t_ord at threshold k:
//'     Se[k]   = Phi( X_d' * beta[2][, t]  - C_d[k] )
//'     Sp[k]   = Phi( C_nd[k] - X_nd' * beta[1][, t] )
//'
// [[Rcpp::export]]
List Rcpp_compute_LC_MVOP_Se_Sp_baseline(   NumericVector trace_beta_flat,       // Flattened 3D array [n_iter, n_chains, n_beta_params]
                                           IntegerVector beta_dims,             // c(n_iter, n_chains, n_beta_params)
                                           CharacterVector beta_names,          // e.g. "beta[1,1,1]", "beta[2,1,3]"
                                           NumericVector trace_C_flat,          // Flattened 3D array [n_iter, n_chains, n_C_params]
                                           IntegerVector C_dims,                // c(n_iter, n_chains, n_C_params)
                                           CharacterVector C_names,             // e.g. "C_vec[1,1]", "C_vec[2,5]"
                                           List baseline_case_nd,               // List of n_tests vectors
                                           List baseline_case_d,                // List of n_tests vectors
                                           IntegerVector n_covs_nd,             // Per-test covariate counts for nd class
                                           IntegerVector n_covs_d,              // Per-test covariate counts for d class
                                           int n_binary_tests,
                                           int n_ordinal_tests,
                                           IntegerVector n_thr_per_ord_test,    // Length n_ordinal_tests
                                           std::string C_param_name = "C_vec"
) {
   
         const int n_iter     = beta_dims[0];
         const int n_chains   = beta_dims[1];
         const int n_beta_p   = beta_dims[2];
         const int n_C_p      = C_dims[2];
         const int n_tests    = n_binary_tests + n_ordinal_tests;
         const int n_samples  = n_iter * n_chains;
         
         // ---- Build unified n_thr vector ----
         Eigen::Matrix<int, -1, 1> n_thr(n_tests);
         int n_thr_max = 1;
         for (int t = 0; t < n_binary_tests; t++) {
           n_thr(t) = 1;
         }
         for (int t_ord = 0; t_ord < n_ordinal_tests; t_ord++) {
           n_thr(n_binary_tests + t_ord) = n_thr_per_ord_test[t_ord];
           if (n_thr_per_ord_test[t_ord] > n_thr_max)
             n_thr_max = n_thr_per_ord_test[t_ord];
         }
         
         // Cumulative slot offsets for compact column storage
         Eigen::Matrix<int, -1, 1> slot_offset(n_tests);
         int n_total_slots = 0;
         for (int t = 0; t < n_tests; t++) {
           slot_offset(t) = n_total_slots;
           n_total_slots += n_thr(t);
         }
         
         // ---- Map R flattened 3D traces -> Eigen matrices [n_samples, n_params] ----
         //   R column-major [iter, chain, param]:
         //     flat index = iter + n_iter * (chain + n_chains * param)
         //   First n_samples entries = param 0, next n_samples = param 1, etc.
         //   Eigen ColMajor Map: column p = entries [p*n_samples .. (p+1)*n_samples-1] ✓
         
         Map<const Eigen::Matrix<double, -1, -1>> beta_mat(trace_beta_flat.begin(), n_samples, n_beta_p);
         Map<const Eigen::Matrix<double, -1, -1>> C_mat(trace_C_flat.begin(), n_samples, n_C_p);
         
         // ---- Build parameter name -> column index maps ----
         std::map<std::string, int> beta_map;
         std::map<std::string, int> C_map;
         
         for (int i = 0; i < beta_names.size(); i++)
           beta_map[as<std::string>(beta_names[i])] = i;
         for (int i = 0; i < C_names.size(); i++)
           C_map[as<std::string>(C_names[i])] = i;
         
         // ---- Pre-resolve all parameter indices (avoid string lookups in hot loop) ----
         
         // Beta: beta_idx_nd[t][j] = column in beta_mat for beta[1, j+1, t+1]
         //       beta_idx_d[t][j]  = column in beta_mat for beta[2, j+1, t+1]
         std::vector<Eigen::Matrix<int, -1, 1>> beta_idx_nd(n_tests);
         std::vector<Eigen::Matrix<int, -1, 1>> beta_idx_d(n_tests);
         
         for (int t = 0; t < n_tests; t++) {
           const int t1 = t + 1;
           
           beta_idx_nd[t].resize(n_covs_nd[t]);
           for (int j = 0; j < n_covs_nd[t]; j++) {
             std::string pname = "beta[1," + std::to_string(j + 1) + "," + std::to_string(t1) + "]";
             auto it = beta_map.find(pname);
             if (it != beta_map.end()) {
               beta_idx_nd[t](j) = it->second;
             } else {
               Rcpp::stop("Beta parameter not found: " + pname);
             }
           }
           
           beta_idx_d[t].resize(n_covs_d[t]);
           for (int j = 0; j < n_covs_d[t]; j++) {
             std::string pname = "beta[2," + std::to_string(j + 1) + "," + std::to_string(t1) + "]";
             auto it = beta_map.find(pname);
             if (it != beta_map.end()) {
               beta_idx_d[t](j) = it->second;
             } else {
               Rcpp::stop("Beta parameter not found: " + pname);
             }
           }
         }
         
         // Cutpoint indices for ordinal tests
         std::vector<Eigen::Matrix<int, -1, 1>> C_col_nd(n_ordinal_tests);
         std::vector<Eigen::Matrix<int, -1, 1>> C_col_d(n_ordinal_tests);
         
         {
           int flat_base = 0;
           for (int t_ord = 0; t_ord < n_ordinal_tests; t_ord++) {
             const int n_thr_t = n_thr_per_ord_test[t_ord];
             C_col_nd[t_ord].resize(n_thr_t);
             C_col_d[t_ord].resize(n_thr_t);
             
             for (int k = 0; k < n_thr_t; k++) {
               const int flat_1based = flat_base + k + 1;
               std::string cname_nd = C_param_name + "[1," + std::to_string(flat_1based) + "]";
               std::string cname_d  = C_param_name + "[2," + std::to_string(flat_1based) + "]";
               
               auto it_nd = C_map.find(cname_nd);
               auto it_d  = C_map.find(cname_d);
               
               if (it_nd != C_map.end() && it_d != C_map.end()) {
                 C_col_nd[t_ord](k) = it_nd->second;
                 C_col_d[t_ord](k)  = it_d->second;
               } else {
                 Rcpp::stop("C parameter not found: " + cname_nd + " or " + cname_d);
               }
             }
             flat_base += n_thr_t;
           }
         }
         
         // ---- Load baseline covariate vectors into Eigen ----
         std::vector<Eigen::Matrix<double, -1, 1>> X_nd(n_tests);
         std::vector<Eigen::Matrix<double, -1, 1>> X_d(n_tests);
         
         for (int t = 0; t < n_tests; t++) {
           NumericVector rvec_nd = baseline_case_nd[t];
           NumericVector rvec_d  = baseline_case_d[t];
           X_nd[t] = Map<const Eigen::Matrix<double, -1, 1>>(rvec_nd.begin(), n_covs_nd[t]);
           X_d[t]  = Map<const Eigen::Matrix<double, -1, 1>>(rvec_d.begin(),  n_covs_d[t]);
         }
         
         // ---- Allocate output: [n_samples, n_total_slots] ----
         Eigen::Matrix<double, -1, -1> Se_out = Eigen::Matrix<double, -1, -1>::Constant(n_samples, n_total_slots, -1.0);
         Eigen::Matrix<double, -1, -1> Sp_out = Eigen::Matrix<double, -1, -1>::Constant(n_samples, n_total_slots, -1.0);
         Eigen::Matrix<double, -1, -1> Fp_out = Eigen::Matrix<double, -1, -1>::Constant(n_samples, n_total_slots, -1.0);
         
         // ---- Vectorised computation per test ----
         for (int t = 0; t < n_tests; t++) {
           
           const bool is_binary = (t < n_binary_tests);
           const int col_base = slot_offset(t);
           
           // Xbeta_nd[s] = sum_j X_nd[t](j) * beta_mat(s, beta_idx_nd[t](j))
           Eigen::Matrix<double, -1, 1> Xbeta_nd = Eigen::Matrix<double, -1, 1>::Zero(n_samples);
           for (int j = 0; j < n_covs_nd[t]; j++) {
             Xbeta_nd.noalias() += X_nd[t](j) * beta_mat.col(beta_idx_nd[t](j));
           }
           
           Eigen::Matrix<double, -1, 1> Xbeta_d = Eigen::Matrix<double, -1, 1>::Zero(n_samples);
           for (int j = 0; j < n_covs_d[t]; j++) {
             Xbeta_d.noalias() += X_d[t](j) * beta_mat.col(beta_idx_d[t](j));
           }
           
           if (is_binary) {     // Se = Phi(Xbeta_d),  Fp = Phi(Xbeta_nd),  Sp = 1 - Fp
             
                 for (int s = 0; s < n_samples; s++) {
                     const double Se_val = R::pnorm(Xbeta_d(s),  0.0, 1.0, 1, 0);
                     const double Fp_val = R::pnorm(Xbeta_nd(s), 0.0, 1.0, 1, 0);
                     Se_out(s, col_base) = Se_val;
                     Fp_out(s, col_base) = Fp_val;
                     Sp_out(s, col_base) = 1.0 - Fp_val;
                 }
             
           } else {
             
                 const int t_ord = t - n_binary_tests;
                 const int n_thr_t = n_thr_per_ord_test[t_ord];
                 
                 for (int k = 0; k < n_thr_t; k++) {
                   
                       // Vectorised: extract cutpoint column, compute argument, apply Phi
                       const Eigen::Matrix<double, -1, 1> C_nd_k = C_mat.col(C_col_nd[t_ord](k));
                       const Eigen::Matrix<double, -1, 1> C_d_k  = C_mat.col(C_col_d[t_ord](k));
                       
                       Eigen::Matrix<double, -1, 1> arg_se = Xbeta_d  - C_d_k;    // Se[k] = Phi(Xbeta_d - C_d[k])
                       Eigen::Matrix<double, -1, 1> arg_sp = C_nd_k - Xbeta_nd;   // Sp[k] = Phi(C_nd[k] - Xbeta_nd)
                       
                       const int col = col_base + k;
                       for (int s = 0; s < n_samples; s++) {
                           const double Se_val = R::pnorm(arg_se(s), 0.0, 1.0, 1, 0);
                           const double Sp_val = R::pnorm(arg_sp(s), 0.0, 1.0, 1, 0);
                           Se_out(s, col) = Se_val;
                           Sp_out(s, col) = Sp_val;
                           Fp_out(s, col) = 1.0 - Sp_val;
                       }
                   
                 }
             
           }
           
         }
         
         // ---- Compute column summaries ----
         Eigen::Matrix<double, -1, -1> Se_summary(n_total_slots, 5);
         Eigen::Matrix<double, -1, -1> Sp_summary(n_total_slots, 5);
         Eigen::Matrix<double, -1, -1> Fp_summary(n_total_slots, 5);
         
         auto compute_col_summary = [&](const Eigen::Matrix<double, -1, 1> &v, 
                                        int row, 
                                        Eigen::Matrix<double, -1, -1> &out) {
           
               const double mean_val = v.mean();
               const double sd_val   = std::sqrt((v.array() - mean_val).square().sum() / 
                                                 static_cast<double>(n_samples - 1));
               
               std::vector<double> sorted(v.data(), v.data() + n_samples);
               std::sort(sorted.begin(), sorted.end());
               
               out(row, 0) = mean_val;
               out(row, 1) = sd_val;
               out(row, 2) = sorted[static_cast<int>(std::floor(n_samples * 0.025))];
               out(row, 3) = sorted[static_cast<int>(std::floor(n_samples * 0.5))];
               out(row, 4) = sorted[std::min(n_samples - 1,
                   static_cast<int>(std::ceil(n_samples * 0.975)))];
               
         };
         
         for (int col = 0; col < n_total_slots; col++) {
           
               if ((Se_out.col(col).array() < -0.5).all()) {
                 Se_summary.row(col).setConstant(-1.0);
                 Sp_summary.row(col).setConstant(-1.0);
                 Fp_summary.row(col).setConstant(-1.0);
               } else {
                 compute_col_summary(Se_out.col(col), col, Se_summary);
                 compute_col_summary(Sp_out.col(col), col, Sp_summary);
                 compute_col_summary(Fp_out.col(col), col, Fp_summary);
               }
           
         }
         
         // ---- Compute AUC per ordinal test ----
         Eigen::Matrix<double, -1, -1> AUC_out;
         Eigen::Matrix<double, -1, -1> AUC_summary_mat;
         
         if (n_ordinal_tests > 0) {
               
               AUC_out.resize(n_samples, n_ordinal_tests);
               AUC_summary_mat.resize(n_ordinal_tests, 5);
               
               for (int t_ord = 0; t_ord < n_ordinal_tests; t_ord++) {
                     
                     const int t = n_binary_tests + t_ord;
                     const int col_base = slot_offset(t);
                     const int n_thr_t = n_thr_per_ord_test[t_ord];
                     
                     for (int s = 0; s < n_samples; s++) {
                       std::vector<std::pair<double, double>> pts;
                       pts.reserve(n_thr_t + 2);
                       pts.emplace_back(0.0, 0.0);
                       
                       for (int k = 0; k < n_thr_t; k++) {
                         const double se = Se_out(s, col_base + k);
                         const double sp = Sp_out(s, col_base + k);
                         if (se > -0.5 && sp > -0.5)
                           pts.emplace_back(1.0 - sp, se);
                       }
                       pts.emplace_back(1.0, 1.0);
                       std::sort(pts.begin(), pts.end());
                       
                       double auc = 0.0;
                       for (size_t i = 0; i + 1 < pts.size(); i++) {
                         auc += 0.5 * (pts[i].second + pts[i + 1].second) *
                           (pts[i + 1].first - pts[i].first);
                       }
                       AUC_out(s, t_ord) = auc;
                     }
                     
                     compute_col_summary( AUC_out.col(t_ord),
                                          t_ord,
                                          AUC_summary_mat);
                 
               }
           
         }
         
         // ---- Reshape Se/Sp/Fp: [n_samples, n_total_slots] -> R 4D [n_iter, n_chains, n_tests, n_thr_max] ----
         auto reshape_to_4d = [&](const Eigen::Matrix<double, -1, -1>& mat_2d) -> NumericVector {
               
               NumericVector out(n_iter * n_chains * n_tests * n_thr_max);
               std::fill(out.begin(), out.end(), -1.0);
               int dims[] = {n_iter, n_chains, n_tests, n_thr_max};
               out.attr("dim") = IntegerVector(dims, dims + 4);
               
               for (int t = 0; t < n_tests; t++) {
                 const int cb = slot_offset(t);
                 for (int k = 0; k < n_thr(t); k++) {
                   const int col = cb + k;
                   for (int chain = 0; chain < n_chains; chain++) {
                     for (int iter = 0; iter < n_iter; iter++) {
                       const int s     = iter + n_iter * chain;
                       const int r_idx = iter + n_iter * (chain + n_chains * (t + n_tests * k));
                       out[r_idx] = mat_2d(s, col);
                     }
                   }
                 }
               }
               
               return out;
           
         };
         
         NumericVector Se_4d = reshape_to_4d(Se_out);
         NumericVector Sp_4d = reshape_to_4d(Sp_out);
         NumericVector Fp_4d = reshape_to_4d(Fp_out);
         
         // ---- Copy Eigen summaries -> R matrices ----
         auto eigen_to_nummat = [](const Eigen::Matrix<double, -1, -1> &emat) -> NumericMatrix {
           
               const int nr = emat.rows(), nc = emat.cols();
               NumericMatrix rmat(nr, nc);
               for (int i = 0; i < nr; i++)
                 for (int j = 0; j < nc; j++)
                   rmat(i, j) = emat(i, j);
               
               return rmat;
           
         };
         
         NumericMatrix Se_summary_R = eigen_to_nummat(Se_summary);
         NumericMatrix Sp_summary_R = eigen_to_nummat(Sp_summary);
         NumericMatrix Fp_summary_R = eigen_to_nummat(Fp_summary);
         
         CharacterVector summary_colnames = CharacterVector::create("mean", "sd", "2.5%", "50%", "97.5%");
         colnames(Se_summary_R) = summary_colnames;
         colnames(Sp_summary_R) = summary_colnames;
         colnames(Fp_summary_R) = summary_colnames;
         
         // Threshold mapping
         IntegerMatrix thresh_map(n_total_slots, 2);
         colnames(thresh_map) = CharacterVector::create("test", "threshold");
         for (int t = 0; t < n_tests; t++)
           for (int k = 0; k < n_thr(t); k++) {
             const int row = slot_offset(t) + k;
             thresh_map(row, 0) = t + 1;
             thresh_map(row, 1) = k + 1;
           }
           
           // n_thr as IntegerVector
           IntegerVector n_thr_out(n_tests);
         for (int t = 0; t < n_tests; t++) n_thr_out[t] = n_thr(t);
         
         // ---- Build result list ----
         List result = List::create(
           Named("Se_baseline")    = Se_4d,
           Named("Sp_baseline")    = Sp_4d,
           Named("Fp_baseline")    = Fp_4d,
           Named("Se_summary")     = Se_summary_R,
           Named("Sp_summary")     = Sp_summary_R,
           Named("Fp_summary")     = Fp_summary_R,
           Named("thresh_mapping") = thresh_map,
           Named("n_thr")          = n_thr_out,
           Named("n_thr_max")      = n_thr_max,
           Named("n_tests")        = n_tests
         );
         
         // ---- Attach AUC ----
         if (n_ordinal_tests > 0) {
           
           auto reshape_to_3d = [&](const Eigen::Matrix<double, -1, -1>& mat_2d, int n_cols) -> NumericVector {
             NumericVector out(n_iter * n_chains * n_cols);
             int dims[] = {n_iter, n_chains, n_cols};
             out.attr("dim") = IntegerVector(dims, dims + 3);
             for (int c = 0; c < n_cols; c++)
               for (int chain = 0; chain < n_chains; chain++)
                 for (int iter = 0; iter < n_iter; iter++) {
                   const int s     = iter + n_iter * chain;
                   const int r_idx = iter + n_iter * (chain + n_chains * c);
                   out[r_idx] = mat_2d(s, c);
                 }
                 return out;
           };
           
           result["AUC_baseline"] = reshape_to_3d(AUC_out, n_ordinal_tests);
           
           NumericMatrix AUC_summary_R = eigen_to_nummat(AUC_summary_mat);
           colnames(AUC_summary_R) = summary_colnames;
           result["AUC_summary"] = AUC_summary_R;
           
           // Pairwise AUC differences
           if (n_ordinal_tests > 1) {
                 
                 const int n_pairs = n_ordinal_tests * (n_ordinal_tests - 1) / 2;
                 Eigen::Matrix<double, -1, -1> AUC_diff(n_samples, n_pairs);
                 Eigen::Matrix<double, -1, -1> AUC_diff_summ(n_pairs, 7);
                 
                 int pair = 0;
                 for (int t1 = 0; t1 < n_ordinal_tests - 1; t1++) {
                   for (int t2 = t1 + 1; t2 < n_ordinal_tests; t2++) {
                     AUC_diff.col(pair) = AUC_out.col(t1) - AUC_out.col(t2);
                     
                     const VectorXd& diff_col = AUC_diff.col(pair);
                     const double mean_val = diff_col.mean();
                     const double sd_val   = std::sqrt((diff_col.array() - mean_val).square().sum() /
                                                       static_cast<double>(n_samples - 1));
                     
                     std::vector<double> sorted(diff_col.data(), diff_col.data() + n_samples);
                     std::sort(sorted.begin(), sorted.end());
                     
                     AUC_diff_summ(pair, 0) = mean_val;
                     AUC_diff_summ(pair, 1) = sd_val;
                     AUC_diff_summ(pair, 2) = sorted[static_cast<int>(std::floor(n_samples * 0.025))];
                     AUC_diff_summ(pair, 3) = sorted[static_cast<int>(std::floor(n_samples * 0.5))];
                     AUC_diff_summ(pair, 4) = sorted[std::min(n_samples - 1,
                                   static_cast<int>(std::ceil(n_samples * 0.975)))];
                     AUC_diff_summ(pair, 5) = static_cast<double>((diff_col.array() > 0.0).count()) / n_samples;
                     AUC_diff_summ(pair, 6) = static_cast<double>((diff_col.array() < 0.0).count()) / n_samples;
                     
                     pair++;
                   }
                 }
                 
                 result["AUC_diff"] = reshape_to_3d(AUC_diff, n_pairs);
                 
                 NumericMatrix AUC_diff_summary_R = eigen_to_nummat(AUC_diff_summ);
                 colnames(AUC_diff_summary_R) = CharacterVector::create(
                   "mean", "sd", "2.5%", "50%", "97.5%", "prob_gt_0", "prob_lt_0");
                 result["AUC_diff_summary"] = AUC_diff_summary_R;
             
           }
           
         }
         
         return result;
   
}
 
 
 
 
 
 
 