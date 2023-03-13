/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_COMMON_TYPES_HPP_
#define NOESIS_GYM_ENVS_RAISIM_COMMON_TYPES_HPP_

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace noesis {
namespace gym {

/*
 * Common algebraic types
 */
using Vector2i = Eigen::Vector2i;
using Vector2 = Eigen::Vector2d;
using Vector3 = Eigen::Vector3d;
using Vector4 = Eigen::Vector4d;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Matrix3 = Eigen::Matrix3d;

/*
 * Common kinematics types
 */
using Position = Vector3;
using RotationMatrix = Matrix3;
using Quaternion = Vector4;
using EulerRpy = Vector3;
using AngleAxis = Eigen::AngleAxis<double>;
using LinearVelocity = Vector3;
using AngularVelocity = Vector3;
using LinearAcceleration = Vector3;
using AngularAcceleration = Vector3;
using LinearImpulse = Vector3;
using LinearForce = Vector3;

/*
 * Aggregate types
 */
using Pose = std::pair<RotationMatrix, Position>;
using Trajectory3 = std::vector<Vector3, Eigen::aligned_allocator<Vector3>>;
using Trajectory6 = std::vector<Vector6, Eigen::aligned_allocator<Vector6>>;

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_COMMON_TYPES_HPP_

/* EOF */
