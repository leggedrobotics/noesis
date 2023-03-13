/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Noesis
#include <noesis/noesis.hpp>
#include <noesis/gym/envs/raisim/kinova3/Kinova3Simulation.hpp>
#include <noesis/gym/envs/raisim/kinova3/Kinova3Visualizer.hpp>

int main(int argc, char** argv)
{
  using namespace noesis::gym;
  
  noesis::init("noesis_gym_kinova3_example");
  
  // Create a simple single-instance environment
  const Kinova3Simulation::World worldType = Kinova3Simulation::World::Empty;
  const bool usePidController = true;
  Kinova3Simulation sim(worldType, usePidController);
  
  // Print system description information to console.
  NINFO("Simulation:\n" << sim.info());
  
  // Create an external visualizer
  Kinova3VisConfig config;
  config.show_goal_pose = true;
  config.show_goal_force = true;
  Kinova3Visualizer vis(config, sim);
  vis.launch();
  vis.pause();
  
  // Initialize
  const RotationMatrix R_G = math::rotation_z(M_PI_2) * math::rotation_x(M_PI_2);
  const Position r_G(0.5, 0.0, 0.5);
  auto q0 = sim.getNominalJointConfiguration();
  auto u0 = sim.getGeneralizedVelocities();
  auto tau = sim.getJointTorques();
  sim.reset(q0, u0.setZero());
  vis.update(R_G, r_G);
  
  // Execute demo episode
  for (size_t t = 0; t < 10 * static_cast<size_t>(1.0/sim.getTimeStep()); ++t) {
    sim.step((usePidController) ? q0 : tau.setZero());
    vis.update(R_G, r_G);
  }
  
  NINFO("[kinova3]: R_ee:\n" << sim.getOrientationWorldToEndEffector())
  NINFO("[kinova3]: r_ee:\n" << sim.getPositionWorldToEndEffectorInWorldFrame())

  // Success
  return 0;
}

/* EOF */
