/// @file ocp_registry.hpp
/// @brief OCP lookup table — maps string names to OCPDescriptor instances.
///
/// To add a new OCP:
///   1. Create include/ocp/ocp_mynew.hpp with a descriptor() function.
///   2. Add one line here: {"mynew", MyNewOCP::descriptor()}.
///   That's it. No changes to planner_node.cpp, trajectory_replayer, or logging.
#pragma once

#include "ocp/ocp_hover.hpp"
#include "ocp/ocp_landing.hpp"
#include "ocp/ocp_stateswitch.hpp"
#include "ocp/ocp_tracking_circle_target.hpp"
#include "ocp/ocp_tracking_bodyrate_tf_noimu.hpp"
#include "ocp/ocp_tracking_bodyrate_tf_imu.hpp"
#include "ocp/ocp_tracking_bodyrate_bf_imu.hpp"
#include "ocp/ocp_tracking_bodyrate_bf_noimu.hpp"

// core types re-exported so existing code using ocp_registry.hpp doesn't need updating
#include "core/ocp_descriptor.hpp"
#include "core/target_snapshot.hpp"

#include <map>
#include <stdexcept>
#include <string>

namespace OCPRegistry {

inline const std::map<std::string, OCPDescriptor>& getTable() {
    static const std::map<std::string, OCPDescriptor> table = {
        {"hover",                          HoverOCP::descriptor()},
        {"landing",                        LandingOCP::descriptor()},
        {"stateswitch",                    StateswitchOCP::descriptor()},
        {"tracking_circle_target",         TrackingCircleTargetOCP::descriptor()},
        {"tracking_bodyrate_tf_noimu",     TrackingBodyrateTfNoImuOCP::descriptor()},
        {"tracking_bodyrate_tf_imu",       TrackingBodyrateTfImuOCP::descriptor()},
        {"tracking_bodyrate_bf_imu",       TrackingBodyrateBfImuOCP::descriptor()},
        {"tracking_bodyrate_bf_noimu",     TrackingBodyrateBfNoImuOCP::descriptor()},
    };
    return table;
}

inline const OCPDescriptor& getDescriptor(const std::string& ocp_type) {
    const auto& table = getTable();
    auto it = table.find(ocp_type);
    if (it == table.end()) {
        throw std::runtime_error("Unknown OCP type: " + ocp_type);
    }
    return it->second;
}

inline double getDT(const std::string& ocp_type)            { return getDescriptor(ocp_type).dt; }
inline int    getDefaultNReplay(const std::string& ocp_type) { return getDescriptor(ocp_type).default_n_replay; }
inline double getDefaultMassKg(const std::string& ocp_type)  { return getDescriptor(ocp_type).default_mass_kg; }
inline Param  getSolverParams(const std::string& ocp_type)   { return getDescriptor(ocp_type).getSolverParams(); }
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const std::string& ocp_type, const OCPCreateArgs& args) {
    return getDescriptor(ocp_type).create(args);
}

} // namespace OCPRegistry
