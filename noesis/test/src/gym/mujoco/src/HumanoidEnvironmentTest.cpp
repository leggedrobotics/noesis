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
#include <noesis/gym/envs/mujoco/humanoid/HumanoidEnvironment.hpp>

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

class HumanoidEnvironmentTest : public ::testing::Test
{
protected:
  HumanoidEnvironmentTest() = default;

  ~HumanoidEnvironmentTest() override = default;

  // aliases
  using Scalar = HumanoidEnvironment::Scalar;
  static constexpr int InputDim = HumanoidEnvironment::InputDim;
  static constexpr int StateDim = HumanoidEnvironment::StateDim;
  static constexpr int Nq = HumanoidEnvironment::BasePoseDim + HumanoidEnvironment::JointsDim;
  static constexpr int Nu = HumanoidEnvironment::BaseVelocityDim + HumanoidEnvironment::JointsDim;
};

/*
 * Tests
 */

TEST_F(HumanoidEnvironmentTest, Basic) {

  // Enable mujoco
  mujoco::init();

  // Create an environment instance
  size_t control_decimation = 5;
  double discount_factor = 0.995;
  double reset_noise_factor = 0.01;
  bool enable_logging = false;
  bool visualize = true;
  std::string name = "Humanoid";
  std::string scope = "tests";
  bool verbose = false;

  HumanoidEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
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

TEST_F(HumanoidEnvironmentTest, Observation) {
  using ObservationsType = HumanoidObservations<Scalar>;

  // Enable mujoco
  mujoco::init();

  // Set initial state (same initial state used in OpenAI-Gym)
  Eigen::VectorXd state{Eigen::VectorXd::Zero(StateDim)};
  state
    << -0.70385602, -0.45753689, 0.09879762, 0.69847772, 0.11832631, -0.64152207, 0.29424004, -0.25618965, -0.41764033, -0.21606265,
    -0.27320787, 0.08020791, -0.75740358, -2.2591636, -0.13424866, -0.17241167, -0.57874475, -2.77815039, 0.64097056, -0.84096273,
    -1.34931651, -0.52099807, 0.34083019, -1.39598124, -0.39710294, -0.13314964, -0.09583486, 0.85405047, -1.81194749, 1.16993966,
    -3.00436114, 6.94154699, -4.05631899, 2.15017507, 0.44331791, -8.72070492, 3.36375265, -1.72785617, 1.4970507, -13.79536919,
    -2.44186967, -0.67447019, -1.24946851, -3.68486233, -1.5691683, 0.08649049, -2.23359748;

  Eigen::VectorXd input{Eigen::VectorXd::Zero(InputDim)};
  input << -0.17453478, 0.27154967, 0.17755671, 0.16887759, 0.20282158, -0.27889562, 0.3168578, -0.16473362, 0.09935772,
    0.28238496, 0.3161085, 0.20578642, 0.35811982, 0.22794355, 0.20396696, -0.026802, -0.39890778;

  // OpenAI-Gym variables
  ObservationsType observationsOpenAI("openai/observations", 1, 1);
  observationsOpenAI[ObservationsType::GenCoord].asFlat()
    << 9.77293627e-02, 7.00250880e-01, 1.36556736e-01, -6.42227275e-01, 2.80258969e-01, -2.62118793e-01, -3.77455211e-01, -2.22860740e-01,
    -2.71578391e-01, 1.26469246e-01, -8.30571226e-01, -2.19956918e+00, -1.67016246e-01, -1.49214583e-01, -6.45888186e-01, -2.76918209e+00,
    6.20342207e-01, -8.10979289e-01, -1.39248186e+00, -5.18750250e-01, 3.77173801e-01, -1.42885977e+00;

  observationsOpenAI[ObservationsType::GenVel].asFlat()
    << -2.53729264e-01, -3.88186416e-01, -1.00383095e-01, 6.04016648e-02, -1.82782000e-01, -5.74378642e+00, 1.84145020e+00, -6.78442011e-01,
    2.03120218e+00, -1.05590268e+00, 4.58207453e+00, -1.45452795e+00, 4.68567227e+00, -2.32580533e+00, 1.83651266e+00, 2.94617598e+00,
    3.17032808e+00, -1.61412040e+00, 4.20168952e+00, -1.78294010e+00, 1.29982670e+00, 3.54138178e+00, -3.44563882e+00;

  observationsOpenAI[ObservationsType::ComInert].asFlat()
    << 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 3.80768914e-01, 1.24800292e+00, 1.55156395e+00, -6.14988355e-01, -1.47813598e-01, -7.03411549e-02,
    -3.03892104e+00, -1.54383217e+00, -4.48015049e-01, 8.32207894e+00, 1.99205699e-02, 4.56364788e-02, 3.68462869e-02, -4.13462466e-03,
    -1.86440022e-02, -4.12394815e-03, -2.40285987e-01, -4.91289435e-02, -1.59479160e-01, 2.03575204e+00, 6.93784656e-02, 5.29803294e-02,
    5.48560851e-02, -3.67612742e-03, 1.17683674e-02, 1.41463802e-02, 1.82285297e-01, 2.62859929e-01, -3.55719010e-01, 5.85278711e+00,
    9.68125500e-02, 2.72336003e-01, 1.96907412e-01, 3.29439031e-02, -1.09051468e-01, 1.63973050e-02, 8.66292183e-01, -1.73047533e-01,
    4.36245380e-01, 4.52555626e+00, 6.30940444e-02, 4.20253872e-01, 3.63457225e-01, 2.11586871e-02, -1.05804722e-01, 9.19118307e-03,
    9.67345614e-01, -5.98274379e-02, 3.10818596e-01, 2.63249442e+00, 1.20876921e-02, 3.53441191e-01, 3.45401030e-01, 3.49236448e-03,
    5.25073423e-02, -5.37142838e-04, 7.76713949e-01, -7.94567610e-03, -1.19462426e-01, 1.76714587e+00, 2.80332271e-01, 1.87853452e-01,
    2.76491541e-01, -1.20544360e-01, -8.82112033e-02, -1.13868560e-01, 5.64712662e-01, 8.90891427e-01, 5.12162151e-01, 4.52555626e+00,
    2.03293901e-01, 1.68468885e-01, 2.51568000e-01, -1.22838215e-01, -6.73976390e-02, -7.52636506e-02, 5.26210719e-01, 6.08841311e-01,
    3.06939895e-01, 2.63249442e+00, 8.10910272e-02, 5.29467633e-02, 1.10931023e-01, -5.15927822e-02, 1.94145733e-02, 2.54191904e-02,
    2.63884416e-01, 3.45499645e-01, -1.30012919e-01, 1.76714587e+00, 1.62533920e-01, 5.93151203e-02, 2.04045021e-01, -7.89401401e-02,
    -1.24459347e-02, -3.17067078e-02, -2.48680787e-01, -4.93262370e-01, -1.04745472e-01, 1.59405984e+00, 6.22430321e-02, 5.99652629e-02,
    9.60889371e-02, 3.66717856e-02, 2.04085709e-02, -2.47958168e-02, 2.10163584e-01, -2.36196974e-01, -1.22945644e-01, 1.19834313e+00,
    2.34431073e-02, 2.62197725e-01, 2.64033023e-01, 2.54857595e-02, -4.98605962e-02, 6.60748519e-03, -6.26168620e-01, 8.05670148e-02,
    -1.27277370e-01, 1.59405984e+00, 1.25339274e-01, 4.21288835e-02, 1.62105663e-01, 5.87275233e-02, -3.03523914e-03, -3.99717919e-03,
    -2.03551985e-01, 3.74581777e-01, 1.49102741e-03, 1.19834313e+00;

  observationsOpenAI[ObservationsType::ComVel].asFlat()
    << 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.92518651e+00, 3.01988776e+00,
    -7.20752301e-01, 3.87319143e-02, -9.10554427e-01, -2.90223918e-01, 3.66348105e+00, 1.54712817e+00, -5.16277265e-01, -1.11293258e-01,
    -7.57001838e-01, -1.09964899e-01, 3.34020027e+00, 1.12044913e+00, 1.47809862e+00, -1.19738521e-01, -6.08183417e-01, -7.94954626e-02,
    -8.67883404e-01, -4.78520189e-01, -4.75000634e-01, -1.18117601e-01, -2.15603601e-01, -4.04386297e-01, -5.51657137e-01, -5.10233899e+00,
    -8.06250035e-01, 1.17021395e+00, -2.69287614e-02, -1.80814428e+00, -5.51657137e-01, -5.10233899e+00, -8.06250035e-01, 1.17021395e+00,
    -2.69287614e-02, -1.80814428e+00, 3.19962008e+00, 3.99067653e+00, 4.01904647e+00, 3.04701481e-01, -6.81954624e-01, 2.73182196e-02,
    4.67479183e+00, 1.21582236e+00, 4.14871159e+00, 1.11070812e+00, -3.02994240e-01, -1.03263089e+00, 4.67479183e+00, 1.21582236e+00,
    4.14871159e+00, 1.11070812e+00, -3.02994240e-01, -1.03263089e+00, 7.16868934e+00, -7.55980417e-01, 2.01566858e-01, -3.43870645e-01,
    -6.92915007e-01, 1.53143149e+00, 6.57584287e+00, 6.85860329e-01, -7.33474063e-01, 9.52536322e-02, -6.15532376e-01, 1.37233701e+00,
    1.94006090e+00, 4.76885503e+00, 1.12582194e+00, 6.54876048e-02, 2.02939780e-01, -1.30160802e+00, -6.65303367e-01, 6.80624952e+00,
    1.70532887e+00, 3.42831970e-01, 5.84868066e-01, -1.39747484e+00;

  observationsOpenAI[ObservationsType::ActForces].asFlat()
    << 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.71549672e+01, -1.74534783e+01, 1.77556708e+01, 1.68877587e+01, 2.02821583e+01,
    -8.36686850e+01, 6.33715630e+01, -1.64733618e+01, 9.93577242e+00, 8.47154886e+01, 6.32216990e+01, 5.14466055e+00, 8.95299539e+00,
    5.69858886e+00, 5.09917401e+00, -6.70049898e-01, -9.97269452e+00;

  observationsOpenAI[ObservationsType::CartExtFrc].asFlat()
    << 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.81639505e+01, 7.19565228e+01,
    7.84076519e+00, 3.41805206e+02, 3.52792342e+01, 4.68061691e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, -9.92995948e+00, -8.03930849e+01, 6.62750821e+00, 3.01462101e+02, 1.12234559e+01, 5.87821933e+02,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00;

  // Tolerance for comparison
  const double tol = 1e-5;

  // Create an environment instance
  size_t control_decimation = 5;
  double discount_factor = 0.995;
  double reset_noise_factor = 0.01;
  bool enable_logging = false;
  bool visualize = false;
  std::string name = "Humanoid";
  std::string scope = "tests";
  bool verbose = false;

  HumanoidEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
                                  name, scope, verbose);

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

  // Convert state to observations
  environment.actions()[0].asFlat() = input.template cast<Scalar>();
  environment.step();
  const auto &observations = environment.observations();

  // Compare openAI output and noesis output
  EXPECT_TRUE(observations[ObservationsType::GenCoord].asFlat().isApprox(
    observationsOpenAI[ObservationsType::GenCoord].asFlat(), tol));
  EXPECT_TRUE(observations[ObservationsType::GenVel].asFlat().isApprox(
    observationsOpenAI[ObservationsType::GenVel].asFlat(), tol));
  EXPECT_TRUE(observations[ObservationsType::ComInert].asFlat().isApprox(
    observationsOpenAI[ObservationsType::ComInert].asFlat(), tol));
  EXPECT_TRUE(observations[ObservationsType::ComVel].asFlat().isApprox(
    observationsOpenAI[ObservationsType::ComVel].asFlat(), tol));
  EXPECT_TRUE(observations[ObservationsType::ActForces].asFlat().isApprox(
    observationsOpenAI[ObservationsType::ActForces].asFlat(), tol));
  EXPECT_TRUE(observations[ObservationsType::CartExtFrc].asFlat().isApprox(
    observationsOpenAI[ObservationsType::CartExtFrc].asFlat(), tol));

  // Disable mujoco
  mujoco::exit();

  // If the test ran until here interpret as a success
  SUCCEED();
}

TEST_F(HumanoidEnvironmentTest, TerminationConditions) {
  /*
   * In this unit test, we run several episodes with a random action agent and check that the average number of steps taken per episode is
   * similar to the behavior of OpenAI gym environments. This is to ensure that the terminal conditions are correctly activated.
   */
  // Enable mujoco
  mujoco::init();

  // Create an environment instance
  size_t control_decimation = 5;
  double discount_factor = 0.995;
  double reset_noise_factor = 0.01;
  bool enable_logging = false;
  bool visualize = false;
  std::string name = "Humanoid";
  std::string scope = "tests";
  bool verbose = false;

  HumanoidEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
                                  name, scope, verbose);

  // Startup environment and visualizer
  environment.configure();

  // Define unit test configuration
  size_t number_of_episodes = 100;
  size_t total_steps_count = 0;
  size_t curr_episode_step_count;

  // Execute simulation
  for (size_t e = 0; e < number_of_episodes; e++) {
    NNOTIFY("Starting new episode...");
    environment.reset();
    curr_episode_step_count = 0;
    while (true) {
      if (environment.terminations().back().type != HumanoidEnvironment::Termination::Type::Unterminated ||
          curr_episode_step_count >= 1000) {
        total_steps_count += curr_episode_step_count;
        break;
      }
      environment.actions().setRandom();
      environment.step();
      curr_episode_step_count++;
    }
  }

  // Over a 100 episodes in Humanoid (OpenAI Gym), number of steps observed (mean +- std): 25.61 +- 7.93
  double mean_episodic_step_count = static_cast<double>(total_steps_count) / number_of_episodes;
  NINFO("Average number of steps taken per episode: " << mean_episodic_step_count);
  ASSERT_NEAR(mean_episodic_step_count, 25.61, 5.0);

  // Disable mujoco
  mujoco::exit();
}

TEST_F(HumanoidEnvironmentTest, Episode) {
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
  double reset_noise_factor = 0.01;
  bool enable_logging = false;
  bool visualize = true;
  std::string name = "Humanoid";
  std::string scope = "tests";
  bool verbose = false;

  HumanoidEnvironment environment(control_decimation, discount_factor, reset_noise_factor, enable_logging, visualize,
                                  name, scope, verbose);

  // Tolerance for comparison
  const double tol = 1e-3;

  // Startup environment and visualizer
  environment.configure();

  // Retrieve package directory
  std::string rootDir = noesis::rootpath() + "/noesis/test/src/gym/mujoco/resources/humanoid_gym_episode/";
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
