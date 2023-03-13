/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_COMMON_LOGGING_HPP_
#define NOESIS_GYM_ENVS_RAISIM_COMMON_LOGGING_HPP_

// C/C++
#include <string>

// Noesis
#include "noesis/framework/utils/string.hpp"
#include "noesis/framework/log/tensorboard.hpp"
#include "noesis/gym/envs/raisim/common/math.hpp"

namespace noesis {
namespace gym {
namespace logging {

static inline void add_3d_vector(noesis::log::TensorBoardLogger& logger, const std::string& name) {
  logger.addLoggingSignal(name + "_x");
  logger.addLoggingSignal(name + "_y");
  logger.addLoggingSignal(name + "_z");
}

static inline void append_3d_vector(
  noesis::log::TensorBoardLogger& logger,
  const std::string& name,
  Vector3 vec
                                   ) {
  logger.appendScalar(name + "_x", vec.x());
  logger.appendScalar(name + "_y", vec.y());
  logger.appendScalar(name + "_z", vec.z());
}

static inline void add_3d_attitude(noesis::log::TensorBoardLogger& logger, const std::string& name) {
  logger.addLoggingSignal(name + "_roll");
  logger.addLoggingSignal(name + "_pitch");
  logger.addLoggingSignal(name + "_yaw");
}

static inline void append_3d_attitude(
  noesis::log::TensorBoardLogger& logger,
  const std::string& name,
  RotationMatrix R
) {
  auto rpy = math::rotation_matrix_to_euler_angles(R);
  logger.appendScalar(name + "_roll", rpy.x());
  logger.appendScalar(name + "_pitch", rpy.y());
  logger.appendScalar(name + "_yaw", rpy.z());
}

static inline void add_foot(noesis::log::TensorBoardLogger& logger, const std::string& name) {
  logger.addLoggingSignal(name + "/position_x");
  logger.addLoggingSignal(name + "/position_y");
  logger.addLoggingSignal(name + "/position_z");
  logger.addLoggingSignal(name + "/velocity_x");
  logger.addLoggingSignal(name + "/velocity_y");
  logger.addLoggingSignal(name + "/velocity_z");
  logger.addLoggingSignal(name + "/contact_state");
}

static inline void append_foot(
  noesis::log::TensorBoardLogger& logger,
  const std::string& name,
  const Position& W_r_WF,
  const LinearVelocity& W_v_WF,
  int c_F
) {
  logger.appendScalar(name + "/position_x", W_r_WF.x());
  logger.appendScalar(name + "/position_y", W_r_WF.y());
  logger.appendScalar(name + "/position_z", W_r_WF.z());
  logger.appendScalar(name + "/velocity_x", W_v_WF.x());
  logger.appendScalar(name + "/velocity_y", W_v_WF.y());
  logger.appendScalar(name + "/velocity_z", W_v_WF.z());
  logger.appendScalar(name + "/contact_state", c_F);
}

static inline void add_joint(noesis::log::TensorBoardLogger& logger, const std::string& name) {
  logger.addLoggingSignal(name + "/joint_command");
  logger.addLoggingSignal(name + "/joint_position");
  logger.addLoggingSignal(name + "/joint_velocity");
  logger.addLoggingSignal(name + "/joint_torque");
}

static inline void append_joint(
  noesis::log::TensorBoardLogger& logger,
  const std::string& name,
  double q_star,
  double q,
  double dq,
  double tau
) {
  logger.appendScalar(name + "/joint_command", q_star);
  logger.appendScalar(name + "/joint_position", q);
  logger.appendScalar(name + "/joint_velocity", dq);
  logger.appendScalar(name + "/joint_torque", tau);
}

static inline void add_leg(noesis::log::TensorBoardLogger& logger, const std::string& scope, const std::string& name) {
  auto namescope = noesis::utils::make_namescope({scope,name});
  std::vector<std::string> joints = {"HAA", "HFE", "KFE"};
  for (const auto& joint: joints) { add_joint(logger, namescope + "_" + joint); }
  add_foot(logger, namescope + "_FOOT");
}

static inline void add_legs(noesis::log::TensorBoardLogger& logger, const std::string& scope) {
  std::vector<std::string> legs = {"LF", "RF", "LH", "RH"};
  for (const auto& leg: legs) { add_leg(logger, scope, leg); }
}

static inline void add_base(noesis::log::TensorBoardLogger& logger, const std::string& scope) {
  add_3d_vector(logger, scope + "/Base/position");
  add_3d_attitude(logger, scope + "/Base/attitude");
  add_3d_vector(logger, scope + "/Base/linear_velocity");
  add_3d_vector(logger, scope + "/Base/angular_velocity");
}

static inline void append_base(
  noesis::log::TensorBoardLogger& logger,
  const std::string& scope,
  const RotationMatrix& R_WB,
  const Position& W_r_WB,
  const LinearVelocity& W_v_WB,
  const AngularVelocity& W_omega_WB
) {
  append_3d_vector(logger, scope + "/Base/position", W_r_WB);
  append_3d_attitude(logger, scope + "/Base/attitude", R_WB);
  append_3d_vector(logger, scope + "/Base/linear_velocity", W_v_WB);
  append_3d_vector(logger, scope + "/Base/angular_velocity", W_omega_WB);
}

} // namespace logging
} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_COMMON_LOGGING_HPP_

/* EOF */
