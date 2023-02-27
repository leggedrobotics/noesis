/*!
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// google test
#include <gtest/gtest.h>

// noesis
#include <noesis/framework/system/process.hpp>
// gym
#include <noesis/gym/envs/mujoco/ant/AntEnvironment.hpp>

namespace noesis {
namespace gym {
namespace tests {

/*
 * Helper Functions
 */

template<typename Scalar_>
void read_data_from_file(const std::string &filename, std::vector<Scalar_> &temp, const bool &verbose = false) {
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
    temp.push_back(static_cast<Scalar_>(data));
    i = i + 1;
  }
  NWARNING_IF(temp.empty(), "Could not ready anything! File is empty!");
}

/*
 * Test fixtures
 */

class AntEnvironmentTest : public ::testing::Test
{
protected:

  AntEnvironmentTest() = default;

  ~AntEnvironmentTest() override = default;

  // aliases
  using Scalar = AntEnvironment::Scalar;
  static constexpr int InputDim = AntEnvironment::InputDim;
  static constexpr int StateDim = AntEnvironment::StateDim;
  static constexpr int Nq = AntEnvironment::BasePoseDim + AntEnvironment::JointsDim;
  static constexpr int Nu = AntEnvironment::BaseVelocityDim + AntEnvironment::JointsDim;
};

/*
 * Tests
 */

TEST_F(AntEnvironmentTest, Basic) {
  // Enable mujoco
  mujoco::init();

  // Create an environment instance
  size_t control_decimation = 5;
  double discount_factor = 0.995;
  double reset_noise_factor = 0.1;
  bool enable_logging = false;
  bool visualize = true;
  std::string name = "Ant";
  std::string scope = "tests";
  bool verbose = false;

  AntEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize, name,
                             scope, verbose);

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

  // If the test ran until here interpret as a success
  SUCCEED();
}

TEST_F(AntEnvironmentTest, Observation) {
  using ObservationsType = AntObservations<Scalar>;

  // Enable mujoco
  mujoco::init();

  // Set initial state (same initial state used in OpenAI-Gym)
  Eigen::VectorXd state{Eigen::VectorXd::Zero(StateDim)};
  state
    << 0.21725871, 0.32618475, 0.85388884, 0.92902668, 0.03964122, 0.29661061, -0.21762388, -0.03344447, 0.6951408, 0.15376108,
    -1.30075812, -0.21514724, -1.24230422, 0.44696246, 0.60915615, -1.75278785, 0.36157226, 2.34261117, -2.59726898, -0.15747079,
    3.73633612, -7.85600234, -5.05298285, -4.1719183, 1.53908301, -12.21583577, 0.63751456, 4.82825823, 5.459245;

  // Set random action as input (same input used in OpenAI-Gym)
  Eigen::VectorXd input{Eigen::VectorXd::Zero(InputDim)};
  input << 0.13251911, 0.67795366, 0.7992045, -0.26598462, 0.40202537, -0.9037211, -0.07721774, -0.64112204;

  // OpenAI-Gym variables
  ObservationsType observationsOpenAI("openai/observations", 1, 1);
  observationsOpenAI[ObservationsType::GenCoord].asFlat()
    << 9.57201991e-01, 9.46965066e-01, 4.95615316e-04, 2.97405945e-01, -1.21682468e-01, -2.71455565e-01, 4.74124598e-01, 2.43849604e-02,
    -1.24004960e+00, -6.38764804e-01, -1.22638292e+00, 5.64499239e-01, 9.99019078e-01;
  observationsOpenAI[ObservationsType::GenVel].asFlat()
    << -1.34387061e+00, 7.71041670e-01, 1.79421230e+00, -1.53698550e+00, -8.22556202e-01, 2.17455197e+00, -1.71282126e+00, -7.79059118e-02,
    -1.00773705e+00, 5.92713379e-01, 1.03935440e+00, 1.12948640e-01, -8.39022249e-01, 1.01003488e+01;
  observationsOpenAI[ObservationsType::CartExtFrc].setZero();

  // Tolerance for comparison
  const double tol = 1e-5;


  // Create an environment instance
  size_t control_decimation = 5;
  double discount_factor = 0.995;
  double reset_noise_factor = 0.1;
  bool enable_logging = false;
  bool visualize = true;
  std::string name = "Ant";
  std::string scope = "tests";
  bool verbose = false;

  AntEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize, name,
                             scope, verbose);

  // Startup environment
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
  EXPECT_TRUE(observations[ObservationsType::CartExtFrc].asFlat().isApprox(
    observationsOpenAI[ObservationsType::CartExtFrc].asFlat(), tol));

  // Disable mujoco
  mujoco::exit();

  // If the test ran until here interpret as a success
  SUCCEED();
}

TEST_F(AntEnvironmentTest, TerminationConditions) {
  /*
   * In this unit test, we run several episodes with a random action agent and check that the average number of steps taken per episode is
   * similar to the behavior of OpenAI gym environments. This is to ensure that the terminal conditions are correctly activated.
   */
  // Enable mujoco
  mujoco::init();


  // Create an environment instance
  size_t control_decimation = 5;
  double discount_factor = 0.995;
  double reset_noise_factor = 0.1;
  bool enable_logging = false;
  bool visualize = true;
  std::string name = "Ant";
  std::string scope = "tests";
  bool verbose = false;

  AntEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize, name,
                             scope, verbose);

  // Startup environment and visualizer
  environment.seed(0);
  environment.configure();

  // Define unit test configuration
  size_t number_of_episodes = 100;
  size_t total_steps_count = 0;
  size_t curr_episode_step_count;

  // Execute simulation
  for (size_t e = 0; e < number_of_episodes; e++) {
    environment.reset();
    curr_episode_step_count = 0;
    while (true) {
      if (environment.terminations().back().type != AntEnvironment::Termination::Type::Unterminated ||
          curr_episode_step_count >= 1000) {
        total_steps_count += curr_episode_step_count;
        break;
      }
      environment.actions().setRandom();
      environment.step();
      curr_episode_step_count++;
    }
  }

  // Over a 100 episodes in Ant (OpenAI Gym), number of steps observed (mean +- std): 162.59 +- 273.48
  double mean_episodic_step_count = static_cast<double>(total_steps_count) / number_of_episodes;
  NINFO("Average number of steps taken per episode: " << mean_episodic_step_count);
  ASSERT_NEAR(mean_episodic_step_count, 162.59, 50.0);

  // Disable mujoco
  mujoco::exit();
}

TEST_F(AntEnvironmentTest, Episode) {
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
  std::string name = "Ant";
  std::string scope = "tests";
  bool verbose = false;

  AntEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize, name,
                             scope, verbose);

  // Startup environment
  environment.configure();

  // Tolerance for comparison
  const double tol = 1e-3;

  // Startup environment and visualizer
  environment.configure();

  // Retrieve package directory
  std::string rootDir = noesis::rootpath() + "/noesis/test/src/gym/mujoco/resources/ant_gym_episode/";
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
  for (size_t i = 0; i < num_of_steps; i++) {
    DNINFO("Step No.: " << i);
    // Add input to take from the buffer
    for (size_t j = 0; j < InputDim; j++) {
      input(j) = actions.at(i * InputDim + j);
    }
    // Add next state from the buffer
    for (size_t j = 0; j < StateDim; j++) {
      state(j) = states_next.at(i * StateDim + j);
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
