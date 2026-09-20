

#pragma once


#include <filesystem>


class HMCResult {
           
         /////////// -------------- PRIVATE members (ONLY accessible WITHIN this class)
         private:
           //// Making states private prevents accidental misuse
           Eigen::Matrix<double, -1, 1> lp_and_grad_outs_;
           Eigen::Matrix<double, -1, 1> lp_and_grad_outs_0_;
           
           Eigen::Matrix<double, -1, 1> main_theta_vec_0_;
           Eigen::Matrix<double, -1, 1> main_theta_vec_;
           Eigen::Matrix<double, -1, 1> main_theta_vec_proposed_;
           ////
           Eigen::Matrix<double, -1, 1> main_velocity_0_vec_;
           Eigen::Matrix<double, -1, 1> main_velocity_vec_proposed_;
           Eigen::Matrix<double, -1, 1> main_velocity_vec_;
           
           double main_p_jump_;
           int main_div_;
           
           Eigen::Matrix<double, -1, 1> us_theta_vec_0_;
           Eigen::Matrix<double, -1, 1> us_theta_vec_;
           Eigen::Matrix<double, -1, 1> us_theta_vec_proposed_;
           ////
           Eigen::Matrix<double, -1, 1> us_velocity_0_vec_;
           Eigen::Matrix<double, -1, 1> us_velocity_vec_proposed_;
           Eigen::Matrix<double, -1, 1> us_velocity_vec_;
           ////
           Eigen::Matrix<double, -1, 1> us_theta_vec_current_segment_;
           Eigen::Matrix<double, -1, 1> us_velocity_vec_current_segment_;
           
           double us_p_jump_;
           int us_div_;
           
           int N_;
           
         /////////// --------------- PUBLIC members (accessible from OUTSIDE this class [e.g. can do: "class.public_member"])
         public:
           //// Burn-in-only endpoint derivatives. Capture before rejection restores the cached gradient.
           double kinetic_energy_rate_main_proposed = 0.0;
           double kinetic_energy_rate_us_proposed = 0.0;
           void record_kinetic_energy_rate_main() {
             kinetic_energy_rate_main_proposed = main_velocity_vec_proposed_.dot(
                 lp_and_grad_outs_.segment(1 + us_theta_vec_.size(), main_theta_vec_.size()));
           }
           void record_kinetic_energy_rate_us() {
             kinetic_energy_rate_us_proposed = us_velocity_vec_proposed_.dot(
                 lp_and_grad_outs_.segment(1, us_theta_vec_.size()));
           }
           //// Constructor 
           HMCResult(int n_params_main, 
                     int n_nuisance, 
                     int N)
           : lp_and_grad_outs_(Eigen::Matrix<double, -1, 1>::Zero((1 + N + n_params_main + n_nuisance)))
           , lp_and_grad_outs_0_(Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params_main + n_nuisance))
           //// main
           , main_theta_vec_0_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           , main_theta_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           , main_theta_vec_proposed_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           ////
           , main_velocity_0_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           , main_velocity_vec_proposed_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           , main_velocity_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_params_main))
           ////
           , main_p_jump_(0.0)
           , main_div_(0)
           //// nuisance
           , us_theta_vec_0_(Eigen::Matrix<double, -1, 1>::Zero(n_nuisance))
           , us_theta_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_nuisance))
           , us_theta_vec_proposed_(Eigen::Matrix<double, -1, 1>::Zero(n_nuisance))
           ////
           , us_velocity_0_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_nuisance))
           , us_velocity_vec_proposed_(Eigen::Matrix<double, -1, 1>::Zero(n_nuisance))
           , us_velocity_vec_(Eigen::Matrix<double, -1, 1>::Zero(n_nuisance))
           ////
           , us_theta_vec_current_segment_(Eigen::Matrix<double, -1, 1>::Zero(n_nuisance))
           , us_velocity_vec_current_segment_(Eigen::Matrix<double, -1, 1>::Zero(n_nuisance))
           ////
           , us_p_jump_(0.0)
           , us_div_(0),
           N_(N)
           {}
           
           // // Add move operations (delete copy)
           // HMCResult(const HMCResult&) = delete;
           // HMCResult& operator=(const HMCResult&) = delete;
           // HMCResult(HMCResult&&) = default;
           // HMCResult& operator=(HMCResult&&) = default;
           
           // Keep default copy and move (all members are copyable/movable):
           HMCResult(const HMCResult&) = default;
           HMCResult& operator=(const HMCResult&) = default;
           HMCResult(HMCResult&&) = default;
           HMCResult& operator=(HMCResult&&) = default;
           
           //// "getters"/"setters" to access the private members
           Eigen::Matrix<double, -1, 1> &lp_and_grad_outs() { 
             return lp_and_grad_outs_; 
           }
           Eigen::Matrix<double, -1, 1> &lp_and_grad_outs_0() { 
             return lp_and_grad_outs_0_; 
           }
           // const Eigen::Matrix<double, -1, 1> &lp_and_grad_outs() const { return lp_and_grad_outs_; }
           Eigen::Matrix<double, -1, 1> log_lik() { return lp_and_grad_outs_.tail(N_); }
           Eigen::Matrix<double, -1, 1> log_lik() const { return lp_and_grad_outs_.tail(N_); } 
           // // cant do this (line below), since .tail() creates a temp. expression, not reference to existing storage!
           // const Eigen::Matrix<double, -1, 1> &log_lik() const {  return lp_and_grad_outs_.tail(N_); } 
           ////
           //// main:
           ////
           Eigen::Matrix<double, -1, 1> &main_theta_vec_0() { return main_theta_vec_0_; }
           const Eigen::Matrix<double, -1, 1> &main_theta_vec_0() const { return main_theta_vec_0_; }
           Eigen::Matrix<double, -1, 1> &main_theta_vec() { return main_theta_vec_; }
           const Eigen::Matrix<double, -1, 1> &main_theta_vec() const { return main_theta_vec_; }
           Eigen::Matrix<double, -1, 1> &main_theta_vec_proposed() { return main_theta_vec_proposed_; }
           const Eigen::Matrix<double, -1, 1> &main_theta_vec_proposed() const { return main_theta_vec_proposed_; }
           Eigen::Matrix<double, -1, 1> &main_velocity_0_vec() { return main_velocity_0_vec_; }
           const Eigen::Matrix<double, -1, 1> &main_velocity_0_vec() const { return main_velocity_0_vec_; }
           Eigen::Matrix<double, -1, 1> &main_velocity_vec_proposed() { return main_velocity_vec_proposed_; }
           const Eigen::Matrix<double, -1, 1> &main_velocity_vec_proposed() const { return main_velocity_vec_proposed_; }
           Eigen::Matrix<double, -1, 1> &main_velocity_vec() { return main_velocity_vec_; }
           const Eigen::Matrix<double, -1, 1> &main_velocity_vec() const { return main_velocity_vec_; }
           double &main_p_jump() { return main_p_jump_; }
           const double &main_p_jump() const { return main_p_jump_; }
           int &main_div() { return main_div_; }
           const int &main_div() const { return main_div_; }
           ////
           //// nuisance:
           ////
           Eigen::Matrix<double, -1, 1> &us_theta_vec_0() { return us_theta_vec_0_; }
           const Eigen::Matrix<double, -1, 1> &us_theta_vec_0() const { return us_theta_vec_0_; }
           Eigen::Matrix<double, -1, 1> &us_theta_vec() { return us_theta_vec_; }
           const Eigen::Matrix<double, -1, 1> &us_theta_vec() const { return us_theta_vec_; }
           Eigen::Matrix<double, -1, 1> &us_theta_vec_proposed() { return us_theta_vec_proposed_; }
           const Eigen::Matrix<double, -1, 1> &us_theta_vec_proposed() const { return us_theta_vec_proposed_; }
           Eigen::Matrix<double, -1, 1> &us_velocity_0_vec() { return us_velocity_0_vec_; }
           const Eigen::Matrix<double, -1, 1> &us_velocity_0_vec() const { return us_velocity_0_vec_; }
           Eigen::Matrix<double, -1, 1> &us_velocity_vec_proposed() { return us_velocity_vec_proposed_; }
           const Eigen::Matrix<double, -1, 1> &us_velocity_vec_proposed() const { return us_velocity_vec_proposed_; }
           Eigen::Matrix<double, -1, 1> &us_velocity_vec() { return us_velocity_vec_; }
           const Eigen::Matrix<double, -1, 1> &us_velocity_vec() const { return us_velocity_vec_; }
           ////
           Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment() { return us_theta_vec_current_segment_; }
           const Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment() const { return us_theta_vec_current_segment_; }
           Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment() { return us_velocity_vec_current_segment_; }
           const Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment() const { return us_velocity_vec_current_segment_; }
           ////
           double &us_p_jump() { return us_p_jump_; }
           const double &us_p_jump() const { return us_p_jump_; }
           int &us_div() { return us_div_; }
           const int &us_div() const { return us_div_; }
           
           /////////// --------------  PUBLIC HELPER FNS - local to this class 
           //// HMC helper fns
           void store_current_state() {
             main_theta_vec_0_ = main_theta_vec_;
             us_theta_vec_0_   = us_theta_vec_;
             main_velocity_0_vec_ = main_velocity_vec_;
             us_velocity_0_vec_   = us_velocity_vec_;
           }  
           
           void accept_proposal_main() {
             main_theta_vec_    = main_theta_vec_proposed_;
             main_velocity_vec_ = main_velocity_vec_proposed_;
           }  
           
           void accept_proposal_us() {
             us_theta_vec_    = us_theta_vec_proposed_;
             us_velocity_vec_ = us_velocity_vec_proposed_;
           }  
           
           void reject_proposal_main() {
             main_theta_vec_    = main_theta_vec_0_;
             main_velocity_vec_ = main_velocity_0_vec_;
             lp_and_grad_outs_  = lp_and_grad_outs_0_;
           } 
            
           void reject_proposal_us() {
             us_theta_vec_    = us_theta_vec_0_;
             us_velocity_vec_ = us_velocity_0_vec_;
             lp_and_grad_outs_ = lp_and_grad_outs_0_;
           }  
           
           //// fn to check state validity
           bool check_state_valid() const {
               return  main_theta_vec_.allFinite() && 
                       main_velocity_vec_.allFinite() && 
                       us_theta_vec_.allFinite() && 
                       us_velocity_vec_.allFinite();
           }
            

   
};














class HMC_output_single_chain {
  
       //// -------------- PRIVATE members (ONLY accessible WITHIN this class)
       private:
          // Make result_input private since it's internal state
          HMCResult result_input_;
          
          // Internal trace storage structures
          struct TraceBuffers {
                Eigen::Matrix<double, -1, -1> main;
                Eigen::Matrix<double, -1, -1> div;
                Eigen::Matrix<double, -1, -1> nuisance;
                Eigen::Matrix<double, -1, -1> log_lik;
                
                // Modified constructor - takes flag for disk mode
                TraceBuffers(int n_params_main, int n_iter, int n_nuisance_to_track, int N, bool use_disk = false) 
                    : main(Eigen::Matrix<double, -1, -1>::Zero(n_params_main, n_iter))
                    , div(Eigen::Matrix<double, -1, -1>::Zero(1, n_iter))
                    , nuisance(use_disk ? Eigen::Matrix<double, -1, -1>() : Eigen::Matrix<double, -1, -1>::Zero(n_nuisance_to_track, n_iter))
                    , log_lik(use_disk ? Eigen::Matrix<double, -1, -1>() : Eigen::Matrix<double, -1, -1>::Zero(N, n_iter))
                {}
                
                // Default move operations are fine for Eigen matrices:
                TraceBuffers(TraceBuffers&&) = default;
                TraceBuffers& operator=(TraceBuffers&&) = default;
                TraceBuffers(const TraceBuffers&) = delete;
                TraceBuffers& operator=(const TraceBuffers&) = delete;
          };

          struct DiagnosticBuffers {
                Eigen::Matrix<int, -1, 1> div_us;
                Eigen::Matrix<int, -1, 1> div_main;
                Eigen::Matrix<double, -1, 1> p_jump_us;
                Eigen::Matrix<double, -1, 1> p_jump_main;
                
                DiagnosticBuffers(int n_iter)
                  : div_us(Eigen::Matrix<int, -1, 1>::Zero(n_iter))
                  , div_main(Eigen::Matrix<int, -1, 1>::Zero(n_iter))
                  , p_jump_us(Eigen::Matrix<double, -1, 1>::Zero(n_iter))
                  , p_jump_main(Eigen::Matrix<double, -1, 1>::Zero(n_iter)) 
                  {}
                
                // Default move operations are fine for Eigen matrices:
                DiagnosticBuffers(DiagnosticBuffers&&) = default;
                DiagnosticBuffers& operator=(DiagnosticBuffers&&) = default;
                DiagnosticBuffers(const DiagnosticBuffers&) = delete;
                DiagnosticBuffers& operator=(const DiagnosticBuffers&) = delete;
          };

          // In HMC_output_single_chain, add file streams:
          struct TraceFiles {
                std::ofstream nuisance_file;
                std::ofstream log_lik_file;
                bool is_open = false;
                std::string base_path;

                TraceFiles() = default;

                // Move constructor
                TraceFiles(TraceFiles&& other) noexcept
                  : nuisance_file(std::move(other.nuisance_file))
                  , log_lik_file(std::move(other.log_lik_file))
                  , is_open(other.is_open)
                  , base_path(std::move(other.base_path))
                {
                  other.is_open = false;
                }

                // Move assignment
                TraceFiles& operator=(TraceFiles&& other) noexcept {
                  if (this != &other) {
                    close();  // Close current files first
                    nuisance_file = std::move(other.nuisance_file);
                    log_lik_file = std::move(other.log_lik_file);
                    is_open = other.is_open;
                    base_path = std::move(other.base_path);
                    other.is_open = false;
                  }
                  return *this;
                }

                // Delete copy
                TraceFiles(const TraceFiles&) = delete;
                TraceFiles& operator=(const TraceFiles&) = delete;

                void open(int chain_id, const std::string& dir = "/tmp/hmc_traces") {
                    std::filesystem::create_directories(dir);
                    base_path = dir + "/chain_" + std::to_string(chain_id);
                    nuisance_file.open(base_path + "_nuisance.bin", std::ios::binary);
                    log_lik_file.open(base_path + "_loglik.bin", std::ios::binary);
                    is_open = true;
                }

                void close() {
                    if (is_open) {
                        nuisance_file.close();
                        log_lik_file.close();
                        is_open = false;
                    }
                }
          };
          
          TraceBuffers traces_;
          DiagnosticBuffers diagnostics_;
          ////
          TraceFiles trace_files_;
          bool use_disk_;
          
        /////////// --------------- PUBLIC members (accessible from OUTSIDE this class [e.g. can do: "class.public_member"])
        public:
          //// Constructor
          //// n_log_lik_rows: rows of the in-RAM log-lik trace (-1 = N, the default; 0 = do not keep it).
          HMC_output_single_chain(int n_iter,
                                  int n_nuisance_to_track,
                                  int n_params_main,
                                  int n_nuisance,
                                  int N,
                                  bool use_disk = false,
                                  int chain_id = 0,
                                  const std::string& trace_dir = "/tmp/hmc_traces",
                                  const int n_log_lik_rows = -1)
            : 
            //// result_input_ is WRITE-ONLY (the chain's final state is copied into it at the end of
            //// fn_sample_HMC_multi_iter_single_thread; nothing reads it). It starts EMPTY so the constructor -
            //// run serially for every chain before the parallel loop - no longer zero-fills ~5 MB per chain
            //// (~0.9 GB at 180 chains); that final copy now allocates it inside the chain's own thread.
            result_input_(0, 0, 0)
          , traces_(n_params_main, n_iter, n_nuisance_to_track, (n_log_lik_rows < 0) ? N : n_log_lik_rows, use_disk)
          , diagnostics_(n_iter)
          , use_disk_(use_disk)
          {
            if (use_disk_) {
              trace_files_.open(chain_id, trace_dir);
            }
          }

            // Destructor to close files
            ~HMC_output_single_chain() {
              trace_files_.close();
          }
          
          // Delete copy constructor/assignment (ofstream not copyable)
          HMC_output_single_chain(const HMC_output_single_chain&) = delete;
          HMC_output_single_chain& operator=(const HMC_output_single_chain&) = delete;
          
          // Move constructor
          HMC_output_single_chain(HMC_output_single_chain&& other) noexcept
            : result_input_(std::move(other.result_input_))
            , traces_(std::move(other.traces_))
            , diagnostics_(std::move(other.diagnostics_))
            , trace_files_(std::move(other.trace_files_))
            , use_disk_(other.use_disk_)
          {}
          
          // Move assignment
          HMC_output_single_chain& operator=(HMC_output_single_chain&& other) noexcept {
            if (this != &other) {
              result_input_ = std::move(other.result_input_);
              traces_ = std::move(other.traces_);
              diagnostics_ = std::move(other.diagnostics_);
              trace_files_ = std::move(other.trace_files_);
              use_disk_ = other.use_disk_;
            }
            return *this;
          }
          
          //// Getters for result_input
          HMCResult &result_input() { return result_input_; }
          //const HMCResult &result_input() const { return result_input_; }
          
          //// Getters for traces
          Eigen::Matrix<double, -1, -1> &trace_main() { return traces_.main; }
          Eigen::Matrix<double, -1, -1> &trace_div() { return traces_.div; }
          Eigen::Matrix<double, -1, -1> &trace_nuisance() { return traces_.nuisance; }
          Eigen::Matrix<double, -1, -1> &trace_log_lik() { return traces_.log_lik; }
          
          //// Getters for diagnostics
          Eigen::Matrix<int, -1, 1> &diagnostics_div_us() { return diagnostics_.div_us; }
          Eigen::Matrix<int, -1, 1> &diagnostics_div_main() { return diagnostics_.div_main; }
          Eigen::Matrix<double, -1, 1> &diagnostics_p_jump_us() { return diagnostics_.p_jump_us; }
          Eigen::Matrix<double, -1, 1> &diagnostics_p_jump_main() { return diagnostics_.p_jump_main; }
          
          //// More getters:
          bool using_disk() const { return use_disk_; }
          std::string trace_dir() const { return trace_files_.base_path; }
          
          //// More getters:
          bool trace_files_is_open() const {
            return trace_files_.is_open;
          }
          
          ////
          //// --------------  PUBLIC HELPER FNS - local to this class:
          ////
          void close_trace_files() {
            trace_files_.close();
          }
          
          void reopen_trace_files(int chain_id, 
                                  const std::string &dir) {
            
                  if (!trace_files_.is_open) {
                    trace_files_.open(chain_id, dir);
                  }
            
          }
          
          //// Helper function to store current iteration results
          void store_iteration(int ii, 
                               bool sample_nuisance,
                               const HMCResult &result) {
            
                  trace_main().col(ii) = result.main_theta_vec();
                  
                  if (sample_nuisance == true) {
                    trace_div()(0, ii) = 0.5 * (result.main_div() + result.us_div());
                  } else {
                    trace_div()(0, ii) = result.main_div();
                  }
                  //// NOTE: the nuisance column used to be written HERE as well as in the RAM
                  //// branch below - the whole latent vector copied TWICE per iteration per
                  //// chain (~1 MB of pointless memory traffic each time at n_nuisance = 60,000).
                  //// It is now written exactly once, below, and only when it is wanted.
                  
                  //// Store diagnostics
                  diagnostics_div_us()(ii) = result.us_div();
                  diagnostics_div_main()(ii) = result.main_div();
                  diagnostics_p_jump_us()(ii) = result.us_p_jump();
                  diagnostics_p_jump_main()(ii) = result.main_p_jump();
    
                  //// Nuisance + log_lik - disk or RAM
                  if (use_disk_) {

                        //// Write nuisance to disk (only when it is actually sampled -
                        //// the stream is then expected to be open):
                        if (sample_nuisance) {
                            if (!trace_files_.nuisance_file.is_open()) {
                              std::cout << "ERROR: nuisance file not open for chain!" << std::endl;
                            }
                            if (!trace_files_.nuisance_file.good()) {
                              std::cout << "ERROR: nuisance file stream bad!" << std::endl;
                            }
                            const Eigen::Matrix<double, -1, 1> &us_col = result.us_theta_vec();
                            trace_files_.nuisance_file.write(
                                reinterpret_cast<const char*>(us_col.data()),
                                us_col.size() * sizeof(double)
                            );
                            //// Force flush to check if write works:
                            trace_files_.nuisance_file.flush();
                        }
                        //// Write log_lik to disk:
                        Eigen::VectorXd ll_col = result.log_lik();
                        trace_files_.log_lik_file.write(
                            reinterpret_cast<const char*>(ll_col.data()),
                            ll_col.size() * sizeof(double)
                        );
                  } else {
                        //// Store in RAM.
                        ////
                        //// A ZERO-ROW buffer means the caller asked for the nuisance trace NOT to be
                        //// kept (n_nuisance_to_track = 0). Its ONLY consumer is the post-hoc
                        //// param_constrain, which needs it solely when the model being constrained
                        //// DECLARES the nuisance coordinates (an external Stan model with a nuisance
                        //// block, or the latent_trait skeleton). The other built-in skeletons declare
                        //// the main parameters only, so for them the trace has no reader at all and
                        //// storing 60,000 x n_iter x n_chains doubles is pure cost. When it IS wanted,
                        //// the buffer is grown to the full length so no coordinate is dropped.
                        if (sample_nuisance && trace_nuisance().rows() > 0) {
                            const int n_us = result.us_theta_vec().size();
                            if (trace_nuisance().rows() != n_us) {
                              trace_nuisance().resize(n_us, trace_nuisance().cols());
                            }
                            trace_nuisance().col(ii) = result.us_theta_vec();
                        }
                        //// Store log-lik (a ZERO-ROW buffer = not wanted: store_log_lik_trace = FALSE):
                        if (trace_log_lik().rows() > 0) trace_log_lik().col(ii) = result.log_lik();
                  }
            
          }
          
          //// Function to check if storage is valid
          bool check_storage_valid() const {
            
                // return traces_.main.allFinite() &&
                //   traces_.div.allFinite() &&
                //   traces_.nuisance.allFinite() &&
                //   traces_.log_lik.allFinite() &&
                //   diagnostics_.p_jump_us.allFinite() &&
                //   diagnostics_.p_jump_main.allFinite();
            
                bool valid = traces_.main.allFinite() &&
                     traces_.div.allFinite() &&
                     diagnostics_.p_jump_us.allFinite() &&
                     diagnostics_.p_jump_main.allFinite();
                
                // Only check nuisance/log_lik if not using disk (they're empty otherwise)
                if (!use_disk_) {
                    valid = valid &&
                      traces_.nuisance.allFinite() &&
                      traces_.log_lik.allFinite();
                }
                
                return valid;

          }
          
};







