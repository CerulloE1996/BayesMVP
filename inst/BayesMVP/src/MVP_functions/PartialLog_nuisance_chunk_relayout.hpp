#pragma once

#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>

//// ============================================================================================================
//// The PartialLog (log-scale) evaluation of the LC_MVP / MVP and LC_MVOP / MVOP models runs with
//// n_chunks_multiplier times as many chunks as the stored num_chunks (smaller chunks for the log-scale work),
//// with n_chunks_multiplier = the sample() argument n_chunks_multiplier_for_PartialLog_log_scale_evaluation
//// (default 3), carried in Model_args_as_cpp_struct.PartialLog_n_chunks_multiplier.
//// The nuisance vector is stored chunk by chunk (chunk k: rows [row_start, row_start + chunk_size) as a
//// column-major (chunk_size x n_tests) block from position row_start * n_tests), so the position of an
//// (observation, test) entry depends on the number of chunks. Before an evaluation with more chunks the
//// nuisance values are re-laid out from the stored chunk layout into the layout of the evaluation, and after
//// it the nuisance gradient is laid back. The per-person log-likelihood is in observation order in both
//// layouts. This is the C++ counterpart of fn_nuisance_chunk_layout_positions() and
//// fn_nuisance_chunk_remap_index() (R_fns_misc_BayesMVP.R); its buffers are allocated once per thread and
//// size, never per evaluation.
//// ============================================================================================================


////
//// ---- position (0-based) in the nuisance vector of (observation n, test t), at index n + N * t, under the
////      chunking of n_chunks chunks (calculate_chunk_sizes() in structures.hpp):
////
inline void fn_nuisance_chunk_layout_positions_cpp( const int N,
                                                    const int n_tests,
                                                    const int n_chunks,
                                                    const int vec_size,
                                                    std::vector<int> &position_of_observation_and_test) {

        const ChunkSizeInfo chunk_info = calculate_chunk_sizes(N, vec_size, n_chunks);
        position_of_observation_and_test.resize(static_cast<size_t>(N) * static_cast<size_t>(n_tests));
        for (int chunk_index = 0; chunk_index < chunk_info.n_total_chunks; ++chunk_index) {
              const int row_start  = chunk_index * chunk_info.normal_chunk_size;
              const int chunk_size = (chunk_index < chunk_info.n_full_chunks) ? chunk_info.normal_chunk_size :
                                                                                 chunk_info.last_chunk_size;
              for (int t = 0; t < n_tests; ++t) {
                    for (int r = 0; r < chunk_size; ++r) {
                          position_of_observation_and_test[(row_start + r) + N * t] =
                                row_start * n_tests + t * chunk_size + r;
                    }
              }
        }

}


////
//// ---- per-thread work space: the map between the two layouts and the two buffers, rebuilt only when N,
////      n_tests, a number of chunks or vec_size changes:
////
struct PartialLog_nuisance_chunk_relayout_workspace {

        int N = -1;
        int n_tests = -1;
        int n_chunks_stored = -1;
        int n_chunks_evaluation = -1;
        int vec_size = -1;
        //// stored_position_of_evaluation_position[p] = the position in the stored layout of the entry at
        //// position p of the evaluation layout:
        std::vector<int> stored_position_of_evaluation_position;
        std::vector<int> positions_stored_layout;
        std::vector<int> positions_evaluation_layout;
        Eigen::Matrix<double, -1, 1> theta_us_in_evaluation_layout;
        Eigen::Matrix<double, -1, 1> grad_us_in_evaluation_layout;

        void prepare( const int N_now,
                      const int n_tests_now,
                      const int n_chunks_stored_now,
                      const int n_chunks_evaluation_now,
                      const int vec_size_now) {

                if ( (N_now == N) && (n_tests_now == n_tests) && (n_chunks_stored_now == n_chunks_stored) &&
                     (n_chunks_evaluation_now == n_chunks_evaluation) && (vec_size_now == vec_size) ) return;
                N = N_now ; n_tests = n_tests_now ; n_chunks_stored = n_chunks_stored_now ;
                n_chunks_evaluation = n_chunks_evaluation_now ; vec_size = vec_size_now;
                fn_nuisance_chunk_layout_positions_cpp( N, n_tests, n_chunks_stored, vec_size,
                                                        positions_stored_layout);
                fn_nuisance_chunk_layout_positions_cpp( N, n_tests, n_chunks_evaluation, vec_size,
                                                        positions_evaluation_layout);
                const int n_us = N * n_tests;
                stored_position_of_evaluation_position.resize(static_cast<size_t>(n_us));
                for (int i = 0; i < n_us; ++i) {
                      stored_position_of_evaluation_position[positions_evaluation_layout[i]] =
                            positions_stored_layout[i];
                }
                theta_us_in_evaluation_layout.resize(n_us);
                grad_us_in_evaluation_layout.resize(n_us);

        }

};

inline PartialLog_nuisance_chunk_relayout_workspace &fn_PartialLog_relayout_workspace_of_this_thread() {
        thread_local PartialLog_nuisance_chunk_relayout_workspace workspace;
        return workspace;
}


////
//// ---- the number of chunks of a PartialLog evaluation, as fn_MVOP_prep and the LC_MVP PartialLog function
////      compute it (capped at N / vec_size chunks, never fewer than n_chunks):
////
inline int fn_PartialLog_n_chunks_of_evaluation( const int N,
                                                 const int n_chunks,
                                                 const int vec_size,
                                                 const int n_chunks_multiplier) {
        return std::min(n_chunks * std::max(1, n_chunks_multiplier), std::max(n_chunks, N / vec_size));
}

inline int fn_vec_size_of_vect_type(const std::string &vect_type) {
        if      (vect_type == "AVX512") return 8;
        else if (vect_type == "AVX2")   return 4;
        else if (vect_type == "AVX")    return 2;
        return 1;
}


////
//// ---- one PartialLog evaluation with n_chunks_multiplier: evaluate(out_mat, theta_us) runs it with the
////      nuisance values in the layout of its own number of chunks, and the nuisance gradient (out_mat entries
////      1, ..., n_us) is laid back into the stored layout. With the stored number of chunks the evaluation reads
////      theta_us_vec_ref as it is (nothing is copied):
////
template <typename EvaluatePartialLog>
inline void fn_PartialLog_evaluation_in_its_own_chunk_layout(
                  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                  const int N,
                  const int n_tests,
                  const int n_chunks_stored,
                  const int vec_size,
                  const int n_chunks_multiplier,
                  EvaluatePartialLog &&evaluate) {

        const int n_chunks_evaluation = fn_PartialLog_n_chunks_of_evaluation( N, n_chunks_stored, vec_size,
                                                                              n_chunks_multiplier);
        if (n_chunks_evaluation == n_chunks_stored) {
              evaluate(out_mat, theta_us_vec_ref);
              return;
        }
        const int n_us = N * n_tests;
        if (theta_us_vec_ref.size() != n_us) {
              throw std::invalid_argument( std::string("PartialLog chunk re-layout: the nuisance vector must ") +
                                           "have N x n_tests entries.");
        }
        PartialLog_nuisance_chunk_relayout_workspace &workspace =
              fn_PartialLog_relayout_workspace_of_this_thread();
        workspace.prepare(N, n_tests, n_chunks_stored, n_chunks_evaluation, vec_size);
        const std::vector<int> &stored_position = workspace.stored_position_of_evaluation_position;
        for (int p = 0; p < n_us; ++p) {
              workspace.theta_us_in_evaluation_layout(p) = theta_us_vec_ref(stored_position[p]);
        }
        evaluate(out_mat, workspace.theta_us_in_evaluation_layout);
        workspace.grad_us_in_evaluation_layout = out_mat.segment(1, n_us);
        for (int p = 0; p < n_us; ++p) {
              out_mat(1 + stored_position[p]) = workspace.grad_us_in_evaluation_layout(p);
        }

}






















