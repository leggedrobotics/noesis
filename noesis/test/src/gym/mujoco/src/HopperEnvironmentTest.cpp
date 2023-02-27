/*!
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2019 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// google test
#include <gtest/gtest.h>

// noesis
#include <noesis/framework/system/process.hpp>
// noesis_environments
#include <noesis/gym/envs/mujoco/hopper/HopperEnvironment.hpp>

namespace noesis {
namespace gym {
namespace tests {


/*
 * Helper Functions
 */

template<typename ScalarType_>
void read_data_from_file(const std::string& filename, std::vector<ScalarType_>& temp, const bool& verbose = false) {
  std::ifstream infile;
  infile.open(filename.c_str());
  // check if file exists
  if (infile.fail()) {
    NFATAL("Could not open file name: " << filename);
  }
  // Read data from file
  NNOTIFY("Reading the file name: " << filename);
  double data;
  int i = 0;
  while (infile >> data) {
    if(verbose) {
      NINFO("Index " << i << ": " << data);
    }
    temp.push_back(static_cast<ScalarType_>(data));
    i = i + 1;
  }
  NWARNING_IF(temp.empty(), "Could not ready anything! File is empty!");
}

/*
 * Test fixtures
 */

class HopperEnvironmentTest : public ::testing::Test
{
protected:
  HopperEnvironmentTest() = default;
  ~HopperEnvironmentTest() override = default;

  // aliases
  using Scalar = HopperEnvironment::Scalar;
  static constexpr int InputDim = HopperEnvironment::InputDim;
  static constexpr int StateDim = HopperEnvironment::StateDim;
  static constexpr int Nq = HopperEnvironment::BasePoseDim + HopperEnvironment::JointsDim;
  static constexpr int Nu = HopperEnvironment::BaseVelocityDim + HopperEnvironment::JointsDim;
};

/*
 * Tests
 */

TEST_F(HopperEnvironmentTest, Basic) {

  // Enable mujoco
  mujoco::init();
  
  // Create an environment instance
  size_t control_decimation = 4;
  double discount_factor = 0.995;
  double reset_noise_factor = 5e-3;
  bool enable_logging = false;
  bool visualize = true;
  std::string name = "Hopper";
  std::string scope = "tests";
  bool verbose = false;

  HopperEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
                                name, scope, verbose);

  // Startup environment and visualizer
  environment.configure();

  // Execute simulation
  for(int i = 0; i < 500; i++) {
    if(i % 100 == 0) {
      environment.reset();
    }
    environment.actions().setRandom();
    environment.step();
  }

  
  // Disable mujoco
  mujoco::exit();
}


TEST_F(HopperEnvironmentTest, Observation) {
  using ObservationsType = HopperObservations<Scalar>;
  
  // Enable mujoco
  mujoco::init();
  
  // Set initial state (same initial state used in OpenAI-Gym)
  Eigen::VectorXd state{Eigen::VectorXd::Zero(StateDim)};
  state << -2.37504274e-02, 1.19587356e+00,-1.51207096e-01, 1.23077714e-03,
           -3.28876620e-01, 2.00823874e-01,-3.37595407e-01,-9.53039535e-01,
           -2.41755324e+00,-7.77601250e-01,-3.61482622e+00, 2.29466713e-01;
  
  // Set random action as input (same input used in OpenAI-Gym)
  Eigen::VectorXd input{Eigen::VectorXd::Zero(InputDim)};
  input << -0.8616561 ,-0.1654141 ,-0.84684837;
  
  // OpenAI-Gym variables
  ObservationsType observationsOpenAI("openai/observations", 1, 1);
  observationsOpenAI[ObservationsType::GenCoord].asFlat() << 1.188241, -0.17557223, -0.00985804, -0.35798921, 0.19776572;
  observationsOpenAI[ObservationsType::GenVel].asFlat() << -0.48686236, -0.90207418, -3.65522611, -1.97981309, -3.59478322, -0.74014047;
  
  // Tolerance for comparison
  const double tol = 1e-5;

  // Create an environment instance
  size_t control_decimation = 4;
  double discount_factor = 0.995;
  double reset_noise_factor = 5e-3;
  bool enable_logging = false;
  bool visualize = false;
  std::string name = "Hopper";
  std::string scope = "tests";
  bool verbose = false;

  HopperEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
                                name, scope, verbose);
  
  // Startup environment
  environment.configure();

  // Extract generalized coordinates and velocities
  Eigen::VectorXd q = state.head(Nq);
  Eigen::VectorXd u = state.tail(Nu);
  // Set the state
  auto& sim = environment.simulation();
  sim.reset(q, u);
  EXPECT_TRUE(q.isApprox(sim.state().getGeneralizedCoordinates(), tol));
  EXPECT_TRUE(u.isApprox(sim.state().getGeneralizedVelocities(), tol));
  
  // Take a step in the environment
  environment.actions()[0].asFlat() = input.template cast<Scalar>();
  environment.step();
  const auto& observations = environment.observations();
  
  // Compare openAI output and noesis output
  EXPECT_TRUE(observations[ObservationsType::GenCoord].asFlat().isApprox(observationsOpenAI[ObservationsType::GenCoord].asFlat(), tol));
  EXPECT_TRUE(observations[ObservationsType::GenVel].asFlat().isApprox(observationsOpenAI[ObservationsType::GenVel].asFlat(), tol));

  // Disable mujoco
  mujoco::exit();
  
  // If the test ran until here interpret as a success
  SUCCEED();
}

TEST_F(HopperEnvironmentTest, TerminationConditions) {
  /*
   * In this unit test, we run several episodes with a random action agent and check that the average number of steps taken per episode is
   * similar to the behavior of OpenAI gym environments. This is to ensure that the terminal conditions are correctly activated.
   */
  // Enable mujoco
  mujoco::init();
  
  // Create an environment instance
  size_t control_decimation = 4;
  double discount_factor = 0.995;
  double reset_noise_factor = 5e-3;
  bool enable_logging = false;
  bool visualize = false;
  std::string name = "Hopper";
  std::string scope = "tests";
  bool verbose = false;

  HopperEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
                                name, scope, verbose);
  
  // Startup environment and visualizer
  environment.configure();

  // Define unit test configuration
  size_t number_of_episodes = 100;
  size_t total_steps_count = 0;
  size_t curr_episode_step_count;
  
  // Execute simulation
  for(size_t e = 0; e < number_of_episodes; e++) {
    NNOTIFY("Starting new episode...");
    environment.reset();
    curr_episode_step_count = 0;
    while(true) {
      if (environment.terminations().back().type != HopperEnvironment::Termination::Type::Unterminated ||
          curr_episode_step_count >= 1000) {
        total_steps_count += curr_episode_step_count;
        break;
      }
      environment.actions().setRandom();
      environment.step();
      curr_episode_step_count++;
    }
  }

  
  // Over a 100 episodes in Hopper (OpenAI Gym), number of steps observed (mean +- std): 21.86 +- 9.73
  double mean_episodic_step_count = static_cast<double>(total_steps_count) / number_of_episodes;
  NINFO("Average number of steps taken per episode: " << mean_episodic_step_count);
  ASSERT_NEAR(mean_episodic_step_count, 21.86, 5.0);
  
  // Disable mujoco
  mujoco::exit();
}

TEST_F(HopperEnvironmentTest, Episode) {
  /*
   * In this unit test, we run a single episode with the data captured from OpenAI Gym implementation. Starting with the same initial state
   * and action, we check whether the agent reaches the same state and obtains the same reward and termination information as recorded in
   * the data.
   */
  // Enable mujoco
  mujoco::init();
  
  // Create an environment instance
  size_t control_decimation = 4;
  double discount_factor = 0.995;
  double reset_noise_factor = 5e-3;
  bool enable_logging = false;
  bool visualize = true;
  std::string name = "Hopper";
  std::string scope = "tests";
  bool verbose = false;

  HopperEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
                                name, scope, verbose);
  
  // Tolerance for comparison
  const double tol = 1e-3;

  // Startup environment and visualizer
  environment.configure();
  
  // Retrieve package directory
  std::string rootDir = noesis::rootpath() + "/noesis/test/src/gym/mujoco/resources/hopper_gym_episode/";
  NFATAL_IF(!boost::filesystem::exists(rootDir), "Folder with episode information has not been set!");
  // Read data from gym episode
  std::vector<double> state_init;
  std::vector<double> states_next;
  std::vector<bool> termination_stats;
  std::vector<double> actions;
  std::vector<double> rewards;
  read_data_from_file<double>(rootDir + "state_init.txt", state_init);
  read_data_from_file<double>(rootDir + "states_next.txt", states_next);
  read_data_from_file<bool>(rootDir + "termination_stats.txt", termination_stats);
  read_data_from_file<double>(rootDir + "actions.txt", actions);
  read_data_from_file<double>(rootDir + "rewards.txt", rewards);
  
  // count number of steps in the episode
  auto num_of_steps = static_cast<size_t>(termination_stats.size());
  
  // Create external state instance
  Eigen::VectorXd input{Eigen::VectorXd::Zero(InputDim)};
  
  // Set initial state
  NFATAL_IF(StateDim != state_init.size(), "Please ensure that same number of states present!")
  Eigen::Map<Eigen::VectorXd> state(state_init.data(), StateDim);
  auto &sim = environment.simulation();
  sim.reset(state);

  // Execute simulation
  for(size_t i = 0; i < num_of_steps; i++) {
    DNINFO("Step No.: " << i);
    // Add input to take from the buffer
    for(size_t j = 0; j < InputDim; j++) {
      input.coeffRef(j, 0) = actions.at(i * InputDim + j);
    }
    // Add next state from the buffer
    for(size_t j = 0; j < StateDim; j++) {
      state.coeffRef(j, 0) = states_next.at(i * StateDim + j);
    }
    // Take a step in the environment
    environment.actions()[0].asFlat() = input.template cast<Scalar>();
    environment.step();
    // Compare OpenAI and noesis next state
    Eigen::VectorXd q = state.head(Nq);
    Eigen::VectorXd u = state.tail(Nu);
    EXPECT_TRUE(q.isApprox(sim.state().getGeneralizedCoordinates(), tol));
    EXPECT_TRUE(u.isApprox(sim.state().getGeneralizedVelocities(), tol));
    // Compare openAI and noesis reward calculations
    DNINFO("OpenAI Reward: " << rewards.at(i));
    DNINFO("Noesis Reward: " << environment.rewards()[0]);
    EXPECT_NEAR(environment.rewards()[0], rewards.at(i), tol);
    // Compare termination states calculations
    EXPECT_TRUE(termination_stats.at(i) == static_cast<bool>(environment.terminations().back().type));
    DNINFO("--------")
  }

  // Disable mujoco
  mujoco::exit();
}

} // namespace tests
} // namespace gym
} // namespace noesis

/* EOF */
