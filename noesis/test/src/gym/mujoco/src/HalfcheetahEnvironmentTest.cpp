/*!
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// google test
#include <gtest/gtest.h>

// noesis
#include <noesis/framework/system/process.hpp>
// noesis_environments
#include <noesis/gym/envs/mujoco/halfcheetah/HalfcheetahEnvironment.hpp>

namespace noesis {
namespace gym {
namespace tests {

/*
 * Helper Functions
 */

template<typename ScalarType_>
void read_data_from_file(const std::string &filename, std::vector<ScalarType_> &temp, const bool &verbose = false) {
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
    if (verbose) {
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

class HalfcheetahEnvironmentTest : public ::testing::Test
{
protected:
  HalfcheetahEnvironmentTest() = default;

  ~HalfcheetahEnvironmentTest() override = default;

  // aliases
  using Scalar = HalfcheetahEnvironment::Scalar;
  static constexpr int InputDim = HalfcheetahEnvironment::InputDim;
  static constexpr int StateDim = HalfcheetahEnvironment::StateDim;
  static constexpr int Nq = HalfcheetahEnvironment::BasePoseDim + HalfcheetahEnvironment::JointsDim;
  static constexpr int Nu = HalfcheetahEnvironment::BaseVelocityDim + HalfcheetahEnvironment::JointsDim;
};

/*
 * Tests
 */

TEST_F(HalfcheetahEnvironmentTest, Basic) {
  // Enable mujoco
  mujoco::init();

  // Create an environment instance
  size_t control_decimation = 5;
  double discount_factor = 0.995;
  double reset_noise_factor = 0.1;
  bool enable_logging = false;
  bool visualize = true;
  std::string name = "Halfcheetah";
  std::string scope = "tests";
  bool verbose = false;

  HalfcheetahEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
                                     name, scope, verbose);

  // Startup environment and visualizer
  environment.configure();

  // Execute simulation
  for (int i = 0; i < 500; i++) {
    if (i % 100 == 0) {
      environment.reset();
    }
    environment.actions().setRandom();
    environment.step();
  }

  // Disable mujoco
  mujoco::exit();
}

TEST_F(HalfcheetahEnvironmentTest, Observation) {
  using ObservationsType = HalfcheetahObservations<Scalar>;

  // Enable mujoco
  mujoco::init();

  // Set initial state (same initial state used in OpenAI-Gym)
  Eigen::VectorXd state{Eigen::VectorXd::Zero(StateDim)};
  state << 1.976361, -0.04918126, 1.22516841, 0.03732042, 0.00656512, -0.24327807,
    0.03691775, 0.39683841, -0.08940568, 0.86553398, 0.82980364, 1.83489399,
    -3.69794828, -5.0316737, -1.7427417, -3.70838078, 3.82866776, -4.1244445;

  // Set random action as input (same input used in OpenAI-Gym)
  Eigen::VectorXd input{Eigen::VectorXd::Zero(InputDim)};
  input << 0.06504647, -0.9196211, 0.14338115, 0.42904022, 0.8367073, 0.37141538;

  // OpenAI-Gym variables
  ObservationsType observationsOpenAI("openai/observations", 1, 1);
  observationsOpenAI[ObservationsType::GenCoord].asFlat()
    << -0.02203424, 1.29658553, -0.02244093, -0.44614184, -0.0273064,
    0.04065023, 0.40125305, -0.0624985;
  observationsOpenAI[ObservationsType::GenVel].asFlat() << 0.84822269, 0.35086632, 0.9000789, -0.38382695, -8.34022391,
    5.86731364, 2.23696419, -1.49099364, 3.01715069;

  // Tolerance for comparison
  const double tol = 1e-5;

  // Create an environment instance
  size_t control_decimation = 5;
  double discount_factor = 0.995;
  double reset_noise_factor = 0.1;
  bool enable_logging = false;
  bool visualize = false;
  std::string name = "Halfcheetah";
  std::string scope = "tests";
  bool verbose = false;

  HalfcheetahEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
                                     name, scope, verbose);

  // Startup environment and visualizer
  environment.configure();

  // Extract generalized coordinates and velocities
  Eigen::VectorXd q = state.head(Nq);
  Eigen::VectorXd u = state.tail(Nu);
  // Set the state
  auto &sim = environment.simulation();
  sim.reset(q, u);
  EXPECT_TRUE(q.isApprox(sim.state().getGeneralizedCoordinates(), tol));
  EXPECT_TRUE(u.isApprox(sim.state().getGeneralizedVelocities(), tol));

  // Take a step in the environment
  environment.actions()[0].asFlat() = input.template cast<Scalar>();
  environment.step();
  const auto &observations = environment.observations();

  // Compare openAI output and noesis output
  EXPECT_TRUE(observations[ObservationsType::GenCoord].asFlat().isApprox(
    observationsOpenAI[ObservationsType::GenCoord].asFlat(), tol));
  EXPECT_TRUE(observations[ObservationsType::GenVel].asFlat().isApprox(
    observationsOpenAI[ObservationsType::GenVel].asFlat(), tol));

  // Disable mujoco
  mujoco::exit();

  // If the test ran until here interpret as a success
  SUCCEED();
}

TEST_F(HalfcheetahEnvironmentTest, TerminationConditions) {
  /*
   * In this unit test, we run several episodes with a random action agent and check that the average number of steps taken per episode is
   * similar to the behavior of OpenAI gym environments. This is to ensure that the terminal conditions are correctly activated.
   * @note: Halfcheetah environment has no terminal conditions.
   */
  // Enable mujoco
  mujoco::init();

  // Create an environment instance
  size_t control_decimation = 5;
  double discount_factor = 0.995;
  double reset_noise_factor = 0.1;
  bool enable_logging = false;
  bool visualize = false;
  std::string name = "Halfcheetah";
  std::string scope = "tests";
  bool verbose = false;

  HalfcheetahEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
                                     name, scope, verbose);

  // Startup environment and visualizer
  environment.configure();


  // Define unit test configuration
  size_t number_of_episodes = 10;
  size_t total_steps_count = 0;
  size_t curr_episode_step_count;

  // Execute simulation
  for (size_t e = 0; e < number_of_episodes; e++) {
    NNOTIFY("Starting new episode...");
    environment.reset();
    curr_episode_step_count = 0;
    while (true) {
      if (environment.terminations().back().type != HalfcheetahEnvironment::Termination::Type::Unterminated ||
          curr_episode_step_count >= 1000) {
        total_steps_count += curr_episode_step_count;
        break;
      }
      environment.actions().setRandom();
      environment.step();
      curr_episode_step_count++;
    }
  }

  // Over a 10 episodes in Halfcheetah (OpenAI Gym), number of steps observed (mean +- std): 1000.0 +- 0.0
  double mean_episodic_step_count = static_cast<double>(total_steps_count) / number_of_episodes;
  NINFO("Average number of steps taken per episode: " << mean_episodic_step_count);
  ASSERT_NEAR(mean_episodic_step_count, 1000, 0.0);

  // Disable mujoco
  mujoco::exit();
}

TEST_F(HalfcheetahEnvironmentTest, Episode) {
  /*
   * In this unit test, we run a single episode with the data captured from OpenAI Gym implementation. Starting with the same initial state
   * and action, we check whether the agent reaches the same state and obtains the same reward and termination information as recorded in
   * the data.
   */
  // Enable mujoco
  mujoco::init();

  // Create an environment instance
  size_t control_decimation = 5;
  double discount_factor = 0.995;
  double reset_noise_factor = 0.1;
  bool enable_logging = false;
  bool visualize = true;
  std::string name = "Halfcheetah";
  std::string scope = "tests";
  bool verbose = false;

  HalfcheetahEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
                                     name, scope, verbose);

  // Tolerance for comparison
  const double tol = 1e-3;

  // Startup environment and visualizer
  environment.configure();

  // Retrieve package directory
  std::string rootDir = noesis::rootpath() + "/noesis/test/src/gym/mujoco/resources/halfcheetah_gym_episode/";
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
  auto num_of_steps = 200;

  // Create external state instance
  Eigen::VectorXd input{Eigen::VectorXd::Zero(InputDim)};

  // Set initial state
  NFATAL_IF(StateDim != state_init.size(), "Please ensure that same number of states present!")
  Eigen::Map<Eigen::VectorXd> state(state_init.data(), StateDim);
  auto &sim = environment.simulation();
  sim.reset(state);

  // Execute simulation
  for (size_t i = 0; i < num_of_steps; i++) {
    NINFO("Step No.: " << i);
    // Add input to take from the buffer
    for (size_t j = 0; j < InputDim; j++) {
      input.coeffRef(j, 0) = actions.at(i * InputDim + j);
    }
    // Add next state from the buffer
    for (size_t j = 0; j < StateDim; j++) {
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
