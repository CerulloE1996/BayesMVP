#pragma once

//// Joint flow-kick-flow splitting, following Alenlov, Doucet and Lindsten (2021),
//// Pseudo-Marginal Hamiltonian Monte Carlo, section 2.4, equations (16)-(19):
//// https://jmlr.org/papers/volume22/19-486/19-486.pdf
////
//// Ordering: F(h/2) K(h) F(h) ... K(h) F(h/2), where
////   F  = flow:   main drift + exact nuisance Gaussian rotation about theta_hat_us,
////   K  = kick:   the coupled residual gradient force (main + nuisance).
////
//// The nuisance rotation frequency is 1 / sqrt(M_us). M_us is the OSCILLATOR
//// frequency/mass, NOT the reference covariance: the exact flow is for the
//// reference N(theta_hat_us, I). The residual kick
////   M_inv_us * (grad_log_pi + (theta - theta_hat_us))
//// is correct as written: with BridgeStan's jacobian = TRUE, grad_log_pi
//// already contains the -theta prior term from the implicit standard normal,
//// so adding (theta - theta_hat_us) leaves only the non-Gaussian residual
//// (grad_log_lik - theta_hat_us). This is a valid symmetric splitting for ANY
//// positive-definite M_us; M_us is not the same object as Beskos's C.
////
//// Four flat variants (dense/diag main metric x general/identity nuisance),
//// mirroring the existing kick-flow-kick integrator dispatch exactly.

ALWAYS_INLINE void leapfrog_integrator_dense_M_diffusion_HMC_dual_flow_kick_flow_InPlace(
        Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
        ////
        Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment,
        Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment,
        ////
        const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
        const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
        const Eigen::Matrix<double, -1, -1> &M_inv_dense_main,
        const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
        const Eigen::Matrix<double, -1, 1> &M_us_vec,
        const Eigen::Matrix<double, -1, 1> &theta_hat_us_vec,
        const Eigen::Matrix<int, -1, -1> &y_ref,
        const int L_ii,
        const double eps,
        const std::string &Model_type,
        const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
        const std::string &grad_option,
        const Model_fn_args_struct &Model_args_as_cpp_struct,
        const Stan_model_struct &Stan_model_as_cpp_struct,
        std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
        const int n_threads_WCP,
        std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                           const std::string,
                           const bool, const bool, const bool,
                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                           const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                           const std::string,
                           const Model_fn_args_struct &,
                           const Stan_model_struct &,
                           std::vector<LC_MVP_workspace_struct> &,
                           const int &)> fn_lp_grad_InPlace
) {

        const int n_nuisance = Model_args_as_cpp_struct.n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct.n_params_main;
        const double half_eps = 0.5 * eps;
        if (L_ii < 1) return;
        ////
        const Eigen::Matrix<double, -1, 1> sqrt_M = M_us_vec.array().sqrt();
        const Eigen::Matrix<double, -1, 1> inv_sqrt_M = M_inv_us_vec.array().sqrt();
        const Eigen::Matrix<double, -1, 1> full_angle = eps * inv_sqrt_M;
        const Eigen::Matrix<double, -1, 1> half_angle = half_eps * inv_sqrt_M;
        const Eigen::Matrix<double, -1, 1> cos_full_flow = full_angle.array().cos();
        const Eigen::Matrix<double, -1, 1> sin_full_flow = full_angle.array().sin();
        const Eigen::Matrix<double, -1, 1> cos_half_flow = half_angle.array().cos();
        const Eigen::Matrix<double, -1, 1> sin_half_flow = half_angle.array().sin();
        //// Initial half flow. Adjacent half flows inside the trajectory are combined below.
        theta_main_vec_proposed_ref.array() += half_eps * velocity_main_vec_proposed_ref.array();
        us_theta_vec_current_segment.array() = theta_us_vec_proposed_ref.array() - theta_hat_us_vec.array();
        us_velocity_vec_current_segment = velocity_us_vec_proposed_ref;
        theta_us_vec_proposed_ref.array() = cos_half_flow.array() * us_theta_vec_current_segment.array()
                                            + sqrt_M.array() * sin_half_flow.array() * us_velocity_vec_current_segment.array()
                                            + theta_hat_us_vec.array();
        velocity_us_vec_proposed_ref.array() = cos_half_flow.array() * us_velocity_vec_current_segment.array()
                                               - inv_sqrt_M.array() * sin_half_flow.array() * us_theta_vec_current_segment.array();

        for (int leapfrog_step = 0; leapfrog_step < L_ii; leapfrog_step++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }

                //// Evaluate both residual forces at the same position, then apply a full kick.
                fn_lp_grad_InPlace( lp_and_grad_outs,
                                    Model_type,
                                    force_autodiff, force_PartialLog, multi_attempts,
                                    theta_main_vec_proposed_ref,
                                    theta_us_vec_proposed_ref,
                                    y_ref,
                                    grad_option,
                                    Model_args_as_cpp_struct,
                                    Stan_model_as_cpp_struct,
                                    LC_MVP_ws_structs,
                                    n_threads_WCP);
                if (!std::isfinite(lp_and_grad_outs(0))) return;
                velocity_main_vec_proposed_ref.array() +=
                    (eps * M_inv_dense_main * lp_and_grad_outs.segment(1 + n_nuisance, n_params_main)).array();
                velocity_us_vec_proposed_ref.array() +=
                    eps * M_inv_us_vec.array() * (lp_and_grad_outs.segment(1, n_nuisance).array()
                                                + theta_us_vec_proposed_ref.array() - theta_hat_us_vec.array());
                //// Full interior flows; the trajectory finishes with a half flow.
                const bool final_flow = (leapfrog_step + 1 == L_ii);
                const double flow_step = final_flow ? half_eps : eps;
                const Eigen::Matrix<double, -1, 1> &cos_flow = final_flow ? cos_half_flow : cos_full_flow;
                const Eigen::Matrix<double, -1, 1> &sin_flow = final_flow ? sin_half_flow : sin_full_flow;
                theta_main_vec_proposed_ref.array() += flow_step * velocity_main_vec_proposed_ref.array();
                us_theta_vec_current_segment.array() = theta_us_vec_proposed_ref.array() - theta_hat_us_vec.array();
                us_velocity_vec_current_segment = velocity_us_vec_proposed_ref;
                theta_us_vec_proposed_ref.array() = cos_flow.array() * us_theta_vec_current_segment.array()
                                                    + sqrt_M.array() * sin_flow.array() * us_velocity_vec_current_segment.array()
                                                    + theta_hat_us_vec.array();
                velocity_us_vec_proposed_ref.array() = cos_flow.array() * us_velocity_vec_current_segment.array()
                                                       - inv_sqrt_M.array() * sin_flow.array() * us_theta_vec_current_segment.array();

        }
        //// Refresh the endpoint density and gradient after the final flow for MH and adaptation.
        fn_lp_grad_InPlace( lp_and_grad_outs,
                            Model_type,
                            force_autodiff, force_PartialLog, multi_attempts,
                            theta_main_vec_proposed_ref,
                            theta_us_vec_proposed_ref,
                            y_ref,
                            grad_option,
                            Model_args_as_cpp_struct,
                            Stan_model_as_cpp_struct,
                            LC_MVP_ws_structs,
                            n_threads_WCP);
        if (!std::isfinite(lp_and_grad_outs(0))) return;

}


ALWAYS_INLINE void leapfrog_integrator_dense_M_diffusion_HMC_dual_Id_M_us_zero_theta_us_flow_kick_flow_InPlace(
        Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
        ////
        Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment,
        Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment,
        ////
        const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
        const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
        const Eigen::Matrix<double, -1, -1> &M_inv_dense_main,
        const Eigen::Matrix<double, -1, 1> &M_us_vec,
        const Eigen::Matrix<int, -1, -1> &y_ref,
        const int L_ii,
        const double eps,
        const std::string &Model_type,
        const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
        const std::string &grad_option,
        const Model_fn_args_struct &Model_args_as_cpp_struct,
        const Stan_model_struct &Stan_model_as_cpp_struct,
        std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
        const int n_threads_WCP,
        std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                           const std::string,
                           const bool, const bool, const bool,
                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                           const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                           const std::string,
                           const Model_fn_args_struct &,
                           const Stan_model_struct &,
                           std::vector<LC_MVP_workspace_struct> &,
                           const int &)> fn_lp_grad_InPlace
) {

        const int n_nuisance = Model_args_as_cpp_struct.n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct.n_params_main;
        const double half_eps = 0.5 * eps;
        if (L_ii < 1) return;
        ////
        const double cos_full_flow = std::cos(eps);
        const double sin_full_flow = std::sin(eps);
        const double cos_half_flow = std::cos(half_eps);
        const double sin_half_flow = std::sin(half_eps);
        //// Initial half flow. Adjacent half flows inside the trajectory are combined below.
        theta_main_vec_proposed_ref.array() += half_eps * velocity_main_vec_proposed_ref.array();
        us_theta_vec_current_segment = theta_us_vec_proposed_ref;
        us_velocity_vec_current_segment = velocity_us_vec_proposed_ref;
        theta_us_vec_proposed_ref.array() = cos_half_flow * us_theta_vec_current_segment.array()
                                            + sin_half_flow * us_velocity_vec_current_segment.array();
        velocity_us_vec_proposed_ref.array() = cos_half_flow * us_velocity_vec_current_segment.array()
                                               - sin_half_flow * us_theta_vec_current_segment.array();

        for (int leapfrog_step = 0; leapfrog_step < L_ii; leapfrog_step++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }

                //// Evaluate both residual forces at the same position, then apply a full kick.
                fn_lp_grad_InPlace( lp_and_grad_outs,
                                    Model_type,
                                    force_autodiff, force_PartialLog, multi_attempts,
                                    theta_main_vec_proposed_ref,
                                    theta_us_vec_proposed_ref,
                                    y_ref,
                                    grad_option,
                                    Model_args_as_cpp_struct,
                                    Stan_model_as_cpp_struct,
                                    LC_MVP_ws_structs,
                                    n_threads_WCP);
                if (!std::isfinite(lp_and_grad_outs(0))) return;
                velocity_main_vec_proposed_ref.array() +=
                    (eps * M_inv_dense_main * lp_and_grad_outs.segment(1 + n_nuisance, n_params_main)).array();
                velocity_us_vec_proposed_ref.array() +=
                    eps * (lp_and_grad_outs.segment(1, n_nuisance).array() + theta_us_vec_proposed_ref.array());
                //// Full interior flows; the trajectory finishes with a half flow.
                const bool final_flow = (leapfrog_step + 1 == L_ii);
                const double flow_step = final_flow ? half_eps : eps;
                const double cos_flow = final_flow ? cos_half_flow : cos_full_flow;
                const double sin_flow = final_flow ? sin_half_flow : sin_full_flow;
                theta_main_vec_proposed_ref.array() += flow_step * velocity_main_vec_proposed_ref.array();
                us_theta_vec_current_segment = theta_us_vec_proposed_ref;
                us_velocity_vec_current_segment = velocity_us_vec_proposed_ref;
                theta_us_vec_proposed_ref.array() = cos_flow * us_theta_vec_current_segment.array()
                                                    + sin_flow * us_velocity_vec_current_segment.array();
                velocity_us_vec_proposed_ref.array() = cos_flow * us_velocity_vec_current_segment.array()
                                                       - sin_flow * us_theta_vec_current_segment.array();

        }
        //// Refresh the endpoint density and gradient after the final flow for MH and adaptation.
        fn_lp_grad_InPlace( lp_and_grad_outs,
                            Model_type,
                            force_autodiff, force_PartialLog, multi_attempts,
                            theta_main_vec_proposed_ref,
                            theta_us_vec_proposed_ref,
                            y_ref,
                            grad_option,
                            Model_args_as_cpp_struct,
                            Stan_model_as_cpp_struct,
                            LC_MVP_ws_structs,
                            n_threads_WCP);
        if (!std::isfinite(lp_and_grad_outs(0))) return;

}


ALWAYS_INLINE void leapfrog_integrator_diag_M_diffusion_HMC_dual_flow_kick_flow_InPlace(
        Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
        ////
        Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment,
        Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment,
        ////
        const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
        const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
        const Eigen::Matrix<double, -1, 1> &M_inv_main_vec,
        const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
        const Eigen::Matrix<double, -1, 1> &M_us_vec,
        const Eigen::Matrix<double, -1, 1> &theta_hat_us_vec,
        const Eigen::Matrix<int, -1, -1> &y_ref,
        const int L_ii,
        const double eps,
        const std::string &Model_type,
        const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
        const std::string &grad_option,
        const Model_fn_args_struct &Model_args_as_cpp_struct,
        const Stan_model_struct &Stan_model_as_cpp_struct,
        std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
        const int n_threads_WCP,
        std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                           const std::string,
                           const bool, const bool, const bool,
                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                           const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                           const std::string,
                           const Model_fn_args_struct &,
                           const Stan_model_struct &,
                           std::vector<LC_MVP_workspace_struct> &,
                           const int &)> fn_lp_grad_InPlace
) {

        const int n_nuisance = Model_args_as_cpp_struct.n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct.n_params_main;
        const double half_eps = 0.5 * eps;
        if (L_ii < 1) return;
        ////
        const Eigen::Matrix<double, -1, 1> sqrt_M = M_us_vec.array().sqrt();
        const Eigen::Matrix<double, -1, 1> inv_sqrt_M = M_inv_us_vec.array().sqrt();
        const Eigen::Matrix<double, -1, 1> full_angle = eps * inv_sqrt_M;
        const Eigen::Matrix<double, -1, 1> half_angle = half_eps * inv_sqrt_M;
        const Eigen::Matrix<double, -1, 1> cos_full_flow = full_angle.array().cos();
        const Eigen::Matrix<double, -1, 1> sin_full_flow = full_angle.array().sin();
        const Eigen::Matrix<double, -1, 1> cos_half_flow = half_angle.array().cos();
        const Eigen::Matrix<double, -1, 1> sin_half_flow = half_angle.array().sin();
        //// Initial half flow. Adjacent half flows inside the trajectory are combined below.
        theta_main_vec_proposed_ref.array() += half_eps * velocity_main_vec_proposed_ref.array();
        us_theta_vec_current_segment.array() = theta_us_vec_proposed_ref.array() - theta_hat_us_vec.array();
        us_velocity_vec_current_segment = velocity_us_vec_proposed_ref;
        theta_us_vec_proposed_ref.array() = cos_half_flow.array() * us_theta_vec_current_segment.array()
                                            + sqrt_M.array() * sin_half_flow.array() * us_velocity_vec_current_segment.array()
                                            + theta_hat_us_vec.array();
        velocity_us_vec_proposed_ref.array() = cos_half_flow.array() * us_velocity_vec_current_segment.array()
                                               - inv_sqrt_M.array() * sin_half_flow.array() * us_theta_vec_current_segment.array();

        for (int leapfrog_step = 0; leapfrog_step < L_ii; leapfrog_step++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }

                //// Evaluate both residual forces at the same position, then apply a full kick.
                fn_lp_grad_InPlace( lp_and_grad_outs,
                                    Model_type,
                                    force_autodiff, force_PartialLog, multi_attempts,
                                    theta_main_vec_proposed_ref,
                                    theta_us_vec_proposed_ref,
                                    y_ref,
                                    grad_option,
                                    Model_args_as_cpp_struct,
                                    Stan_model_as_cpp_struct,
                                    LC_MVP_ws_structs,
                                    n_threads_WCP);
                if (!std::isfinite(lp_and_grad_outs(0))) return;
                velocity_main_vec_proposed_ref.array() +=
                    eps * M_inv_main_vec.array() * lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array();
                velocity_us_vec_proposed_ref.array() +=
                    eps * M_inv_us_vec.array() * (lp_and_grad_outs.segment(1, n_nuisance).array()
                                                + theta_us_vec_proposed_ref.array() - theta_hat_us_vec.array());
                //// Full interior flows; the trajectory finishes with a half flow.
                const bool final_flow = (leapfrog_step + 1 == L_ii);
                const double flow_step = final_flow ? half_eps : eps;
                const Eigen::Matrix<double, -1, 1> &cos_flow = final_flow ? cos_half_flow : cos_full_flow;
                const Eigen::Matrix<double, -1, 1> &sin_flow = final_flow ? sin_half_flow : sin_full_flow;
                theta_main_vec_proposed_ref.array() += flow_step * velocity_main_vec_proposed_ref.array();
                us_theta_vec_current_segment.array() = theta_us_vec_proposed_ref.array() - theta_hat_us_vec.array();
                us_velocity_vec_current_segment = velocity_us_vec_proposed_ref;
                theta_us_vec_proposed_ref.array() = cos_flow.array() * us_theta_vec_current_segment.array()
                                                    + sqrt_M.array() * sin_flow.array() * us_velocity_vec_current_segment.array()
                                                    + theta_hat_us_vec.array();
                velocity_us_vec_proposed_ref.array() = cos_flow.array() * us_velocity_vec_current_segment.array()
                                                       - inv_sqrt_M.array() * sin_flow.array() * us_theta_vec_current_segment.array();

        }
        //// Refresh the endpoint density and gradient after the final flow for MH and adaptation.
        fn_lp_grad_InPlace( lp_and_grad_outs,
                            Model_type,
                            force_autodiff, force_PartialLog, multi_attempts,
                            theta_main_vec_proposed_ref,
                            theta_us_vec_proposed_ref,
                            y_ref,
                            grad_option,
                            Model_args_as_cpp_struct,
                            Stan_model_as_cpp_struct,
                            LC_MVP_ws_structs,
                            n_threads_WCP);
        if (!std::isfinite(lp_and_grad_outs(0))) return;

}


ALWAYS_INLINE void leapfrog_integrator_diag_M_diffusion_HMC_dual_Id_M_us_zero_theta_us_flow_kick_flow_InPlace(
        Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
        Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
        ////
        Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment,
        Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment,
        ////
        const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
        const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
        const Eigen::Matrix<double, -1, 1> &M_inv_main_vec,
        const Eigen::Matrix<double, -1, 1> &M_us_vec,
        const Eigen::Matrix<int, -1, -1> &y_ref,
        const int L_ii,
        const double eps,
        const std::string &Model_type,
        const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
        const std::string &grad_option,
        const Model_fn_args_struct &Model_args_as_cpp_struct,
        const Stan_model_struct &Stan_model_as_cpp_struct,
        std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
        const int n_threads_WCP,
        std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                           const std::string,
                           const bool, const bool, const bool,
                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                           const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                           const std::string,
                           const Model_fn_args_struct &,
                           const Stan_model_struct &,
                           std::vector<LC_MVP_workspace_struct> &,
                           const int &)> fn_lp_grad_InPlace
) {

        const int n_nuisance = Model_args_as_cpp_struct.n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct.n_params_main;
        const double half_eps = 0.5 * eps;
        if (L_ii < 1) return;
        ////
        const double cos_full_flow = std::cos(eps);
        const double sin_full_flow = std::sin(eps);
        const double cos_half_flow = std::cos(half_eps);
        const double sin_half_flow = std::sin(half_eps);
        //// Initial half flow. Adjacent half flows inside the trajectory are combined below.
        theta_main_vec_proposed_ref.array() += half_eps * velocity_main_vec_proposed_ref.array();
        us_theta_vec_current_segment = theta_us_vec_proposed_ref;
        us_velocity_vec_current_segment = velocity_us_vec_proposed_ref;
        theta_us_vec_proposed_ref.array() = cos_half_flow * us_theta_vec_current_segment.array()
                                            + sin_half_flow * us_velocity_vec_current_segment.array();
        velocity_us_vec_proposed_ref.array() = cos_half_flow * us_velocity_vec_current_segment.array()
                                               - sin_half_flow * us_theta_vec_current_segment.array();

        for (int leapfrog_step = 0; leapfrog_step < L_ii; leapfrog_step++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }

                //// Evaluate both residual forces at the same position, then apply a full kick.
                fn_lp_grad_InPlace( lp_and_grad_outs,
                                    Model_type,
                                    force_autodiff, force_PartialLog, multi_attempts,
                                    theta_main_vec_proposed_ref,
                                    theta_us_vec_proposed_ref,
                                    y_ref,
                                    grad_option,
                                    Model_args_as_cpp_struct,
                                    Stan_model_as_cpp_struct,
                                    LC_MVP_ws_structs,
                                    n_threads_WCP);
                if (!std::isfinite(lp_and_grad_outs(0))) return;
                velocity_main_vec_proposed_ref.array() +=
                    eps * M_inv_main_vec.array() * lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array();
                velocity_us_vec_proposed_ref.array() +=
                    eps * (lp_and_grad_outs.segment(1, n_nuisance).array() + theta_us_vec_proposed_ref.array());
                //// Full interior flows; the trajectory finishes with a half flow.
                const bool final_flow = (leapfrog_step + 1 == L_ii);
                const double flow_step = final_flow ? half_eps : eps;
                const double cos_flow = final_flow ? cos_half_flow : cos_full_flow;
                const double sin_flow = final_flow ? sin_half_flow : sin_full_flow;
                theta_main_vec_proposed_ref.array() += flow_step * velocity_main_vec_proposed_ref.array();
                us_theta_vec_current_segment = theta_us_vec_proposed_ref;
                us_velocity_vec_current_segment = velocity_us_vec_proposed_ref;
                theta_us_vec_proposed_ref.array() = cos_flow * us_theta_vec_current_segment.array()
                                                    + sin_flow * us_velocity_vec_current_segment.array();
                velocity_us_vec_proposed_ref.array() = cos_flow * us_velocity_vec_current_segment.array()
                                                       - sin_flow * us_theta_vec_current_segment.array();

        }
        //// Refresh the endpoint density and gradient after the final flow for MH and adaptation.
        fn_lp_grad_InPlace( lp_and_grad_outs,
                            Model_type,
                            force_autodiff, force_PartialLog, multi_attempts,
                            theta_main_vec_proposed_ref,
                            theta_us_vec_proposed_ref,
                            y_ref,
                            grad_option,
                            Model_args_as_cpp_struct,
                            Stan_model_as_cpp_struct,
                            LC_MVP_ws_structs,
                            n_threads_WCP);
        if (!std::isfinite(lp_and_grad_outs(0))) return;

}
