/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_COMMON_RAISIM_HPP_
#define NOESIS_GYM_ENVS_RAISIM_COMMON_RAISIM_HPP_

// C/C++
#include <vector>
#include <unordered_map>

// raiSim
#include <raisim/World.hpp>

namespace noesis {
namespace gym {
namespace raisim {

/*
 * Configuration helpers
 */

struct RaiSimSolverConfig {
  //! @brief Parameter which controls the over-relaxation of the contact solver
  //! @note This is always set to 1.0, i.e. results in the solver behaving as a true Gauss-Seidel solver.
  double alpha_initial = 1.0;
  //! @warning Will be deprecated in future version of raiSim
  double alpha_minimum = 0.7;
  //! @warning Will be deprecated in future version of raiSim
  double alpha_decay = 1.0;
  //! @brief Threshold for error tolerance for the squared contact velocity multiplied by the timestep: avg(error_cvel^2)*dt^2
  //! @note Defaults to 10^-7*dt
  double threshold = 1.0e-7;
  //! The Error Reduction Parameter that fixes position error after one simulation step by applying additional force.
  //! @note Defaults to 0.0
  double error_reduction_parameter = 0.0;
  //! @brief Number of times the concat solver iterates over all contacts.
  //! @note Increase iterations for cases where multiple contacts arise in coupled systems.
  int max_iterations = 50;
};

struct RaiSimContactProperties {
  //! @brief The pair of material defining the type of interaction.
  std::pair<std::string, std::string> materials;
  //! @brief The coefficient of Coulomb (static) friction.
  //! @note Isotropic friction is assumed, meaning all closed contacts result in a friction cone with circular profile.
  double friction = 0.8;
  //! @brief The coefficient of restitution (for Newton-type impacts) determines the compliance of collisions.
  double restitution = 0.0;
  //! @brief The restitution threshold.
  double threshold = 0.01;
};

static inline std::string articulated_system_info(::raisim::ArticulatedSystem& system) {
  std::stringstream ss;
  ss << "Model Info:";
  ss << "\n====== [ Dimensions ] ============";
  ss << "\n-DoFs: " << system.getDOF();
  ss << "\n-dim(u): " << system.getGeneralizedVelocityDim();
  ss << "\n-dim(q): " << system.getGeneralizedCoordinateDim();
  ss << "\n-Num. of frames: " << system.getFrames().size();
  ss << "\n-Num. of bodies: " << system.getBodyNames().size();
  ss << "\n-Num. of collision bodies: " << system.getCollisionBodies().size();
  ss << "\n-Num. of Names: " << system.getBodyNames().size();
  ss << "\n-Num. of Masses: " << system.getMass().size();
  ss << "\n-Num. of Inertias: " << system.getInertia().size();
  ss << "\n-Num. of Link CoMs: " << system.getLinkCOM().size();
  ss << "\n-Total Mass [kg]: " << system.getTotalMass();
  ss << "\n====== [ Frames ] ================\n";
  const auto& frames = system.getFrames();
  for (size_t j = 0; j < frames.size(); ++j) {
    ss << "\n[Frame '" << j << "']:";
    ss << "\n-Name: " << frames[j].name;
    ss << "\n-Body Name: " << frames[j].bodyName;
    ss << "\n-Body Id: " << frames[j].currentBodyId;
    ss << "\n-Parent Name: " << frames[j].parentName;
    ss << "\n-Parent Id: " << frames[j].parentId;
    ss << "\n-Is Child: " << frames[j].isChild;
    ss << "\n-Joint Type: " << static_cast<int>(frames[j].jointType);
    ss << "\n-Position:\n" << frames[j].position;
    ss << "\n-Orientation:\n" << frames[j].orientation;
  }
  ss << "\n====== [ Bodies ] ================\n";
  const auto& body_names = system.getBodyNames();
  const auto& body_masses = system.getMass();
  const auto& body_inertias = system.getInertia();
  const auto& body_link_coms = system.getLinkCOM();
  for (size_t j = 0; j < body_names.size(); ++j) {
    ss << "\n[Body '" << j << "']:";
    ss << "\n-Name: " << body_names[j];
    ss << "\n-Mass: " << body_masses[j];
    ss << "\n-Inertia:\n" << body_inertias[j];
    ss << "-Link CoM:\n" << body_link_coms[j];
  }
  ss << "\n====== [ Collision Bodies ] ======\n";
  const auto& colBodies = system.getCollisionBodies();
  for (size_t j = 0; j < colBodies.size(); ++j) {
    ss << "\n[Collision Body '" << j << "']:";
    ss << "\n-Name: " << colBodies[j].name;
    ss << "\n-Local Index: " << colBodies[j].localIdx;
    ss << "\n-Position:\n" << colBodies[j].posOffset;
    ss << "\n-Orientation:\n" << colBodies[j].rotOffset;
  }
  return ss.str();
}

} // namespace raisim
} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_COMMON_RAISIM_HPP_

/* EOF */
