/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// google test
#include <gtest/gtest.h>

// Noesis
#include <noesis/gym/envs/raisim/capler/CaplerSimulation.hpp>
#include <noesis/gym/envs/raisim/capler/CaplerVisualizer.hpp>

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

class CaplerSimulationTest : public ::testing::Test
{
protected:
  CaplerSimulationTest() = default;
  ~CaplerSimulationTest() override = default;
};

/*
 * Tests
 */

TEST_F(CaplerSimulationTest, ModelInfo) {
  CaplerSimulation sim;
  NINFO("[CaplerSimulation]: " << sim.info());
}

TEST_F(CaplerSimulationTest, Headless) {
  CaplerSimulation sim;
  std::cout << std::endl;
  sim.capler()->printOutFrameNamesInOrder();
  sim.capler()->printOutBodyNamesInOrder();
  for (size_t t = 0; t < 5*400; ++t) { sim.step(sim.getNominalJointConfiguration()); }
  NINFO("[CaplerSimulation]: Total DoFs: " << sim.capler()->getDOF());
  NINFO("[CaplerSimulation]: dim(u): " << sim.capler()->getGeneralizedVelocityDim());
  NINFO("[CaplerSimulation]: dim(q): " << sim.capler()->getGeneralizedCoordinateDim());
  NINFO("[CaplerSimulation]: Total mass: " << sim.capler()->getTotalMass());
}

TEST_F(CaplerSimulationTest, InEmptyWorld) {
  CaplerSimulation sim(CaplerSimulation::World::Empty);
  CaplerVisualizer vis(sim);
  vis.launch();
  std::cout << std::endl;
  sim.capler()->printOutFrameNamesInOrder();
  sim.capler()->printOutBodyNamesInOrder();
  for (size_t t = 0; t < 5*400; ++t) { sim.step(Eigen::VectorXd::Zero(12)); }
  std::cout << "\n\n-----------------------------------------\n";
  std::cout << "qj:\n" << sim.getJointPositions().transpose();
  std::cout << "\n-----------------------------------------\n";
  std::cout << "B_r_BF:\n" << sim.getPositionBaseToFootInBaseFrame().transpose();
  std::cout << "\n-----------------------------------------\n";
  std::cout << std::endl;
}

TEST_F(CaplerSimulationTest, InGridWorld) {
  CaplerSimulation sim(CaplerSimulation::World::Grid);
  CaplerVisualizer vis(sim);
  vis.launch();
  vis.pause();
  auto q = sim.getGeneralizedCoordinates();
  auto u = sim.getGeneralizedVelocities();
  auto qj = sim.getNominalJointConfiguration();
  Position W_r_WB(0.0, 0.0, 0.8);
  q << W_r_WB, math::euler_angles_to_quaternion(EulerRpy(0, 0, 0)), qj;
  sim.reset(q, u);
  std::cout << "\n\n-----------------------------------------\n";
  std::cout << "B_r_BF:\n" << sim.getPositionBaseToFootInBaseFrame().transpose();
  std::cout << "\n-----------------------------------------\n";
  std::cout << "qj:\n" << sim.getJointPositions().transpose();
  std::cout << "\n-----------------------------------------\n";
  std::cout << std::endl;
  for (size_t t = 0; t < 10*400; ++t) { sim.step(qj); }
}

TEST_F(CaplerSimulationTest, CheckContactStates) {
  CaplerSimulation sim(CaplerSimulation::World::Grid);
  CaplerVisualizer vis(sim);
  vis.launch();
  const auto phiStar = sim.getNominalJointConfiguration();
  for (size_t t = 0; t < 5*400; ++t) {
    sim.step(phiStar);
    if (t%40 == 0) {
      const auto& R_WB = sim.getOrientationBaseToWorld();
      const auto& W_r_WB = sim.getPositionWorldToBaseInWorldFrame();
      const auto& W_v_WB = sim.getLinearVelocityWorldToBaseInWorldFrame();
      const auto& W_r_WF = sim.getPositionWorldToFootInWorldFrame();
      const auto c_F = sim.getFootContact();
      std::cout << "\n\n-----------------------------------------\n";
      std::cout << "R_WB:\n" << R_WB << "\n";
      std::cout << "W_r_WB: " << W_r_WB.transpose() << "\n";
      std::cout << "W_v_WB: " << W_v_WB.transpose() << "\n";
      std::cout << "W_r_WF:\n" << W_r_WF.transpose() << "\n";
      std::cout << "c_F: " << c_F << "\n";
      std::cout << "\n-----------------------------------------\n";
      std::cout << std::endl;
    }
  }
}

TEST_F(CaplerSimulationTest, CheckKinematics) {
  CaplerSimulation sim;
  CaplerVisualizer vis(sim);
  vis.launch();
  const auto qjStar = sim.getNominalJointConfiguration();
  for (size_t t = 0; t < 3*400; ++t) { sim.step(qjStar); }
  // Retrieve measured kinematic measurements
  const auto qj = sim.getJointPositions();
  const auto B_r_BF = sim.getPositionBaseToFootInBaseFrame();
  // Compute forward and inverse kinematics
  const auto B_r_BF_FK = sim.foot_forward_kinematics(qj);
  const auto qj_IK = sim.foot_inverse_kinematics(B_r_BF);
  std::cout << "\n\n-----------------------------------------\n";
  std::cout << "B_r_BF:\n" << B_r_BF.transpose() << "\n";
  std::cout << "B_r_BF_FK:\n" << B_r_BF_FK.transpose() << "\n";
  std::cout << "\n-----------------------------------------\n";
  std::cout << "qj:\n" << qj.transpose() << "\n";
  std::cout << "qj_IK:\n" << qj_IK.transpose() << "\n";
  std::cout << "\n-----------------------------------------\n";
  std::cout << std::endl;
}

TEST_F(CaplerSimulationTest, SimulationThroughput) {
  CaplerSimulation sim(CaplerSimulation::World::Grid);
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
  NINFO("[CaplerSimulation]: FlatWorld: Simulation Throughput: FPS: " << static_cast<double>(total_sim_steps)/dt);
}

TEST_F(CaplerSimulationTest, RecordVideo) {
  CaplerSimulation sim(CaplerSimulation::World::Grid);
  CaplerVisualizer vis(sim);
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

TEST_F(CaplerSimulationTest, FootContactForces) {
  CaplerSimulation sim(CaplerSimulation::World::Grid);
  CaplerVisualizer vis(sim);
  vis.launch();
  vis.pause();
  auto q = sim.getGeneralizedCoordinates();
  auto u = sim.getGeneralizedVelocities();
  auto qj = sim.getNominalJointConfiguration();
  Position W_r_WB(0.0, 0.0, 0.8);
  q << W_r_WB, math::euler_angles_to_quaternion(EulerRpy(0, 0, 0)), qj;
  sim.reset(q, u);
  for (size_t t = 0; t < 5*400; ++t) { sim.step(qj); }
  auto& W_p_F = sim.getNetImpulseOfFootContacts();
  auto& W_f_F = sim.getFootForceInWorldFrame();
  auto n_F = sim.getNumberOfFootContacts();
  auto n_B = sim.getBaseContacts();
  auto n_L = sim.getLegContacts();
  auto n_S = sim.getShankContacts();
  auto n_C = sim.getTotalContacts();
  std::cout << "\n\n-----------------------------------------\n";
  std::cout << "W_p_F:\n" << W_p_F.transpose() << "\n";
  std::cout << "W_f_F:\n" << W_f_F.transpose() << "\n";
  std::cout << "n_F: " << n_F << "\n";
  std::cout << "n_B: " << n_B << "\n";
  std::cout << "n_L: " << n_L << "\n";
  std::cout << "n_S: " << n_S << "\n";
  std::cout << "n_C: " << n_C << "\n";
  std::cout << "\n-----------------------------------------\n";
  std::cout << std::endl;
}

TEST_F(CaplerSimulationTest, Randomize) {
  CaplerSimulation sim(CaplerSimulation::World::Grid);
  CaplerVisualizer vis(sim);
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

TEST_F(CaplerSimulationTest, SelfCollisions) {
  CaplerSimulation sim(CaplerSimulation::World::Empty);
  CaplerVisualizer vis(sim);
  vis.launch();
  vis.pause();
  auto q = sim.getGeneralizedCoordinates();
  auto u = sim.getGeneralizedVelocities();
  auto qj = sim.getNominalJointConfiguration();
  qj << 0.0, M_PI_2, M_PI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0;
  q << Position(0.0, 0.0, 0.0), math::euler_angles_to_quaternion(EulerRpy(0, 0, 0)), qj;
  sim.reset(q, u);
  const auto dt = sim.getTimeStep();
  for (size_t t = 0; t < 10*400; ++t) {
    qj(0) = M_PI_2 * std::sin(0.5*M_PI*t*dt);
    sim.step(qj);
    NWARNING_IF(sim.getLegContacts() > 0, "[CaplerSimulation]: Leg is in contact with the base!");
  }
}

TEST_F(CaplerSimulationTest, CentroidProperties) {
  CaplerSimulation sim(CaplerSimulation::World::Grid);
  auto q = sim.getGeneralizedCoordinates();
  auto u = sim.getGeneralizedVelocities();
  auto qj = sim.getNominalJointConfiguration();
  Position W_r_WB(0.0, 0.0, 0.7);
  q << W_r_WB, math::euler_angles_to_quaternion(EulerRpy(0, 0, 0)), qj;
  sim.reset(q, u);
  for (size_t t = 0; t < 10*400; ++t) { sim.step(qj); }
  const Position r_com = sim.getPositionWorldToComInWorldFrame();
  const Matrix3 I_com = sim.getInertiaComInWorldFrame();
  NWARNING("[CaplerSimulation]: CoM Position: " << r_com.transpose());
  NWARNING("[CaplerSimulation]: CoM Inertia:\n" << I_com);
}

} // namespace test
} // namespace gym
} // namespace noesis

/* EOF */
