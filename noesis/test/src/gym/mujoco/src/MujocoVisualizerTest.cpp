/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// google test
#include <gtest/gtest.h>

// noesis
#include <noesis/framework/system/time.hpp>

// noesis_environments
#include <noesis/gym/envs/mujoco/common/visualizer.hpp>
#include <noesis/gym/envs/mujoco/common/simulation.hpp>

namespace noesis {
namespace gym {
namespace tests {

/*
 * Test fixtures
 */

class MujocoVisualizerTest : public ::testing::Test
{
protected:
  MujocoVisualizerTest() = default;
  ~MujocoVisualizerTest() = default;
};

/*
 * Tests
 */

TEST_F(MujocoVisualizerTest, Allocation) {
  // Enable mujoco
  mujoco::init();

  MujocoVisualizerConfig visConf;
  visConf.name = "visualizer";
  visConf.scope = "tests";
  visConf.real_time_factor = 1.0;
  visConf.use_render_thread = true;
  visConf.verbose = true;
  MujocoVisualizer visualizer(visConf);

  // Disable mujoco
  mujoco::exit();
}

TEST_F(MujocoVisualizerTest, SynchronousRendering) {

  // Enable mujoco
  mujoco::init();

  // Create a simulation instance
  MujocoSimulationConfig config;
  std::string modelFile = mujoco::assets() + "/benchmarks/humanoid.xml";
  NFATAL_IF(!boost::filesystem::exists(modelFile), "MJCF file not found: " << modelFile);
  config.model_file = modelFile;
  MujocoSimulation simulation(config);

  // Configs
  size_t T = 5000;
  double dt = simulation.timestep();

  // Create a visualizer
  MujocoVisualizerConfig visConf;
  visConf.name = "visualizer";
  visConf.scope = "tests";
  visConf.real_time_factor = 1.0;
  visConf.use_render_thread = false;
  visConf.verbose = true;
  MujocoVisualizer visualizer(visConf);

  // Configure the visualizer with the simulation instance
  visualizer.setSimulation(&simulation);
  visualizer.launch();
  visualizer.setBodyForCameraTracking("torso");

  NNOTIFY("Starting basic operation test with random actions ...");

  // Run for fixed number of steps w/ asynchronous rendering
  auto timer = noesis::Time::Now();
  // Demo input as zero
  auto input = Eigen::VectorXd::Zero(simulation.state().input_dims());
  // Simulated time since the last visual update
  double total = 0;
  double time = 0;
  int i = 0;
  while (total < T * dt) {
    while (time < 0) {
      if (i % 1000 == 0) {
        auto q = simulation.state().getInitialConfiguration();
        auto u = Eigen::VectorXd::Zero(simulation.state().u_dims());
        q.setRandom();
        q(2) = std::max(1.0, q(2));
        simulation.reset(q, u);
      }
      simulation.step(input);
      total += simulation.timestep();
      time += simulation.timestep();
      i++;
    }
    time -= visualizer.getRealTimeFactor() / visualizer.getFramesPerSecond();
    visualizer.render();
  }
  auto elapsedSecs = timer.elapsed().toSeconds();

  NINFO("Elapsed time: " << elapsedSecs);
  NINFO("Expected time: " << static_cast<double>(T) * dt);
  EXPECT_NEAR(elapsedSecs, static_cast<double>(T) * dt, 0.5);

  // Disable mujoco
  mujoco::exit();
}

TEST_F(MujocoVisualizerTest, AsynchronousRendering) {
  // Enable mujoco
  mujoco::init();

  // Create humanoid simulation
  MujocoSimulationConfig config;
  std::string modelFile = mujoco::assets() + "/benchmarks/humanoid.xml";
  NFATAL_IF(!boost::filesystem::exists(modelFile), "MJCF file not found: " << modelFile);
  config.model_file = modelFile;
  MujocoSimulation simulation(config);

  // Configs
  size_t T = 5000;
  double dt = simulation.timestep();

  // Create a visualizer
  MujocoVisualizerConfig visConf;
  visConf.name = "visualizer";
  visConf.scope = "tests";
  visConf.real_time_factor = 1.0;
  visConf.use_render_thread = true;
  visConf.verbose = true;
  MujocoVisualizer visualizer(visConf);

  // Configure the visualizer with the simulation instance
  visualizer.setSimulation(&simulation);
  visualizer.launch();
  visualizer.setBodyForCameraTracking("torso");

  NNOTIFY("Starting basic operation test with random actions ...");

  // Demo input as zero
  auto input = Eigen::VectorXd::Zero(simulation.state().input_dims());
  // Run for fixed number of steps w/ asynchronous rendering
  auto timer = noesis::Time::Now();
  for (size_t i = 0; i < T; i++) {
    if (i % 1000 == 0) {
      auto q = simulation.state().getInitialConfiguration();
      auto u = Eigen::VectorXd::Zero(simulation.state().u_dims());
      q.setRandom();
      q(2) = std::max(1.0, q(2));
      simulation.reset(q, u);
    }
    simulation.step(input);
  }
  auto elapsedSecs = timer.elapsed().toSeconds();

  NINFO("Elapsed time: " << elapsedSecs);
  NINFO("Expected time: " << static_cast<double>(T) * dt);
  EXPECT_NEAR(elapsedSecs, static_cast<double>(T) * dt, 0.5);

  // Disable mujoco
  mujoco::exit();
}

} // namespace tests
} // namespace gym
} // namespace noesis

/* EOF */
