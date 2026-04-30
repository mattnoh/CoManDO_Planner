#include "ocp/ocp_tracking_bodyrate_tf_imu.hpp"
#include "ocp/ocp_tracking_bodyrate_tf_noimu.hpp"

#include <Eigen/Dense>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<int NX, int NX_SS, int NU_SS, int HORIZON, int IDX_Q, int IDX_THETA>
void buildWithFeedbackWarmStart(
    std::shared_ptr<OptimalControlProblem<double>> (*create_fn)(
        const Eigen::VectorXd&,
        const std::vector<Eigen::VectorXd>&,
        const std::vector<Eigen::VectorXd>&,
        const std::vector<Eigen::MatrixXd>&,
        double),
    void (*sanitize_fn)(Eigen::VectorXd&),
    const char* name) {
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(NX);
    x0(IDX_Q) = 1.0;

    std::vector<Eigen::VectorXd> prev_u(HORIZON, Eigen::VectorXd::Zero(NU_SS));
    std::vector<Eigen::VectorXd> prev_x(HORIZON + 1, Eigen::VectorXd::Zero(NX_SS));
    std::vector<Eigen::MatrixXd> prev_k(HORIZON, Eigen::MatrixXd::Zero(NU_SS, NX_SS));
    for (int k = 0; k < HORIZON; ++k) {
        prev_u[k](0) = 9.81;
        prev_u[k](IDX_THETA) = 0.1;
        prev_x[k](IDX_Q) = 1.0;
        prev_k[k](0, 0) = 100.0;
    }
    prev_x.back()(IDX_Q) = 1.0;

    Eigen::VectorXd dirty = Eigen::VectorXd::Zero(NU_SS);
    dirty(0) = 1000.0;
    dirty(1) = 1000.0;
    dirty(IDX_THETA) = 1000.0;
    sanitize_fn(dirty);
    require(dirty(0) < 100.0 && dirty(1) < 100.0 && dirty(IDX_THETA) < 1.0,
            "warm-start sanitize did not clamp expected fields");

    auto problem = create_fn(x0, prev_u, prev_x, prev_k, 0.1);
    require(static_cast<bool>(problem), name);
}

}  // namespace

int main() {
    try {
        require(TrackingBodyrateTfNoImuOCP::descriptor().warm_start ==
                    OCPDescriptor::WarmStart::Feedback,
                "noimu descriptor did not opt into feedback warm start");
        require(TrackingBodyrateTfImuOCP::descriptor().warm_start ==
                    OCPDescriptor::WarmStart::Feedback,
                "imu descriptor did not opt into feedback warm start");

        buildWithFeedbackWarmStart<
            TrackingBodyrateTfNoImuOCP::NX,
            TrackingBodyrateTfNoImuOCP::NX_SS,
            TrackingBodyrateTfNoImuOCP::NU_SS,
            TrackingBodyrateTfNoImuOCP::HORIZON,
            TrackingBodyrateTfNoImuOCP::IDX_Q,
            TrackingBodyrateTfNoImuOCP::IDX_THETA>(
            TrackingBodyrateTfNoImuOCP::create,
            TrackingBodyrateTfNoImuOCP::sanitizeControl,
            "failed to create noimu OCP with feedback warm start");

        buildWithFeedbackWarmStart<
            TrackingBodyrateTfImuOCP::NX,
            TrackingBodyrateTfImuOCP::NX_SS,
            TrackingBodyrateTfImuOCP::NU_SS,
            TrackingBodyrateTfImuOCP::HORIZON,
            TrackingBodyrateTfImuOCP::IDX_Q,
            TrackingBodyrateTfImuOCP::IDX_THETA>(
            TrackingBodyrateTfImuOCP::create,
            TrackingBodyrateTfImuOCP::sanitizeControl,
            "failed to create imu OCP with feedback warm start");
    } catch (const std::exception& e) {
        std::cerr << "Target-frame warm-start test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Target-frame warm-start tests passed\n";
    return 0;
}
