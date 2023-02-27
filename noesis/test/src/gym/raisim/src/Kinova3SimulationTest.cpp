/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// google test
#include <gtest/gtest.h>

// Noesis
#include <noesis/gym/envs/raisim/kinova3/Kinova3Simulation.hpp>
#include <noesis/gym/envs/raisim/kinova3/Kinova3Visualizer.hpp>

namespace noesis {
namespace gym {
namespace test {

/*
 * WARNING: READ THIS BEFORE RUNNING ANY TESTS
 *
 * Due to a problem w/ OpenGL/OGRE/RaiSimOgre, multiple tests cannot be run
 * in tandem or sequentially because the underlying rendering class does not fully
 * release all resources upon instance destruction (visualizer is a singleton).
 *
 * Therefore ***EACH TEST MUST BE RUN ON ITS OWN****
 *
 * 1. From terminal manually:
 *  https://github.com/google/googletest/blob/master/googletest/docs/advanced.md#running-a-subset-of-the-tests
 * 2. From CLion: Use the built-in play buttons in the editor's gutter:
 *  https://www.jetbrains.com/help/clion/creating-google-test-run-debug-configuration-for-test.html#run-tests
 */

/*
 * Test fixtures
 */

class Kinova3SimulationTest : public ::testing::Test
{
protected:
  Kinova3SimulationTest() = default;
  ~Kinova3SimulationTest() override = default;
};

/*
 * Tests
 */

TEST_F(Kinova3SimulationTest, ModelInfo) {
  Kinova3Simulation sim;
  NINFO("[Kinova3Simulation]: " << sim.info());
}

TEST_F(Kinova3SimulationTest, Headless) {
  Kinova3Simulation sim;
  std::cout << std::endl;
  sim.kinova3()->printOutFrameNamesInOrder();
  sim.kinova3()->printOutBodyNamesInOrder();
  for (size_t t = 0; t < 5*400; ++t) { sim.step(sim.getNominalJointConfiguration()); }
  NINFO("[Kinova3Simulation]: Total DoFs: " << sim.kinova3()->getDOF());
  NINFO("[Kinova3Simulation]: dim(u): " << sim.kinova3()->getGeneralizedVelocityDim());
  NINFO("[Kinova3Simulation]: dim(q): " << sim.kinova3()->getGeneralizedCoordinateDim());
  NINFO("[Kinova3Simulation]: Total mass: " << sim.kinova3()->getTotalMass());
}

TEST_F(Kinova3SimulationTest, InEmptyWorld) {
  Kinova3Simulation sim(Kinova3Simulation::World::Empty);
  Kinova3Visualizer vis(sim);
  vis.launch();
  std::cout << std::endl;
  sim.kinova3()->printOutFrameNamesInOrder();
  sim.kinova3()->printOutBodyNamesInOrder();
  for (size_t t = 0; t < 5*400; ++t) { sim.step(Eigen::VectorXd::Zero(12)); }
  std::cout << "\n\n-----------------------------------------\n";
  std::cout << "qj:\n" << sim.getJointPositions().transpose();
  std::cout << "\n-----------------------------------------\n";
  std::cout << "W_r_ee:\n" << sim.getPositionWorldToEndEffectorInWorldFrame().transpose();
  std::cout << "\n-----------------------------------------\n";
  std::cout << std::endl;
}

TEST_F(Kinova3SimulationTest, InGridWorld) {
  Kinova3Simulation sim(Kinova3Simulation::World::Grid);
  Kinova3Visualizer vis(sim);
  vis.launch();
  vis.pause();
  auto q = sim.getGeneralizedCoordinates();
  auto u = sim.getGeneralizedVelocities();
  auto qj = sim.getNominalJointConfiguration();
  Position W_r_WB(0.0, 0.0, 0.8);
  q << W_r_WB, math::euler_angles_to_quaternion(EulerRpy(0, 0, 0)), qj;
  sim.reset(q, u);
  std::cout << "\n\n-----------------------------------------\n";
  std::cout << "W_r_ee:\n" << sim.getPositionWorldToEndEffectorInWorldFrame().transpose();
  std::cout << "\n-----------------------------------------\n";
  std::cout << "qj:\n" << sim.getJointPositions().transpose();
  std::cout << "\n-----------------------------------------\n";
  std::cout << std::endl;
  for (size_t t = 0; t < 10*400; ++t) { sim.step(qj); }
}

TEST_F(Kinova3SimulationTest, CheckContactStates) {
  // TODO
}

TEST_F(Kinova3SimulationTest, CheckKinematics) {
  // TODO
}

TEST_F(Kinova3SimulationTest, SimulationThroughput) {
  Kinova3Simulation sim(Kinova3Simulation::World::Grid);
  auto q = sim.getGeneralizedCoordinates();
  auto u = sim.getGeneralizedVelocities();
  auto qj = sim.getNominalJointConfiguration();
  Position W_r_WB(0.0, 0.0, 0.6);
  q << W_r_WB, math::euler_angles_to_quaternion(EulerRpy(0, 0, 0)), qj;
  constexpr auto total_sim_steps = static_cast<size_t>(1e+6);
  noesis::Time timer = noesis::Time::Now();
  sim.reset(q, u);
  for (size_t t = 0; t < total_sim_steps; ++t) { sim.step(qj); }
  auto dt = timer.elapsed().toSeconds();
  NINFO("[Kinova3Simulation]: FlatWorld: Simulation Throughput: FPS: " << static_cast<double>(total_sim_steps)/dt);
}

TEST_F(Kinova3SimulationTest, RecordVideo) {
  Kinova3Simulation sim(Kinova3Simulation::World::Grid);
  Kinova3Visualizer vis(sim);
  vis.launch();
  vis.pause();
  auto q = sim.getGeneralizedCoordinates();
  auto u = sim.getGeneralizedVelocities();
  auto qj = sim.getNominalJointConfiguration();
  Position W_r_WB(0.0, 0.0, 0.8);
  q << W_r_WB, math::euler_angles_to_quaternion(EulerRpy(0, 0, 0)), qj;
  sim.reset(q, u);
  vis.startRecording(noesis::logpath() + "/video", "test");
  for (size_t t = 0; t < 10*400; ++t) { sim.step(qj); }
  vis.stopRecording();
}

TEST_F(Kinova3SimulationTest, EndEffectorContactForces) {
  // TODO
}

TEST_F(Kinova3SimulationTest, Randomize) {
  Kinova3Simulation sim(Kinova3Simulation::World::Grid);
  Kinova3Visualizer vis(sim);
  vis.launch();
  vis.pause();
  auto q = sim.getGeneralizedCoordinates();
  auto u = sim.getGeneralizedVelocities();
  auto qj = sim.getNominalJointConfiguration();
  Position W_r_WB(0.0, 0.0, 0.8);
  q << W_r_WB, math::euler_angles_to_quaternion(EulerRpy(0, 0, 0)), qj;
  sim.reset(q, u);
  sim.randomize();
  for (size_t t = 0; t < 30*400; ++t) {
    if (t%400==0) {
      sim.randomize(t);
      sim.reset(q, u);
    }
    sim.step(qj);
  }
}

TEST_F(Kinova3SimulationTest, SelfCollisions) {
  // TODO
}

TEST_F(Kinova3SimulationTest, CentroidProperties) {
  Kinova3Simulation sim(Kinova3Simulation::World::Grid);
  auto q = sim.getGeneralizedCoordinates();
  auto u = sim.getGeneralizedVelocities();
  auto qj = sim.getNominalJointConfiguration();
  Position W_r_WB(0.0, 0.0, 0.7);
  q << W_r_WB, math::euler_angles_to_quaternion(EulerRpy(0, 0, 0)), qj;
  sim.reset(q, u);
  for (size_t t = 0; t < 10*400; ++t) { sim.step(qj); }
  const Position r_com = sim.getPositionWorldToComInWorldFrame();
  const Matrix3 I_com = sim.getInertiaComInWorldFrame();
  NWARNING("[Kinova3Simulation]: CoM Position: " << r_com.transpose());
  NWARNING("[Kinova3Simulation]: CoM Inertia:\n" << I_com);
}

} // namespace test
} // namespace gym
} // namespace noesis

/* EOF */
