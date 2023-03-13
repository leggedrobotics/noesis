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

// noesis_environments
#include <noesis/gym/envs/mujoco/common/simulation.hpp>
#include <noesis/gym/envs/mujoco/common/visualizer.hpp>

namespace noesis {
namespace gym {
namespace tests {


/*
 * Test fixtures
 */

/*!
 * Test class loads the humanoid asset file to check for dynamics simulation.
 */
class MujocoSimulationTest : public ::testing::Test
{
protected:

  // aliases
  static constexpr size_t BasePoseDim = 7;
  static constexpr size_t BaseVelocityDim = 6;
  static constexpr size_t JointsDim = 17;
  static constexpr size_t StateDim = BasePoseDim + JointsDim + BaseVelocityDim + JointsDim;
  static constexpr size_t InputDim = JointsDim;

  MujocoSimulationTest() = default;

  ~MujocoSimulationTest() override = default;

};

/*
 * Tests
 */

TEST_F(MujocoSimulationTest, Basic) {

  // Enable mujoco
  mujoco::init();

  // Simulation
  MujocoSimulationConfig config;
  // Set the default MJCF file to load
  std::string modelFile = mujoco::assets() + "/benchmarks/humanoid.xml";
  NFATAL_IF(!boost::filesystem::exists(modelFile), "MJCF file not found: " << modelFile);
  config.model_file = modelFile;
  MujocoSimulation simulation(config);

  ASSERT_TRUE(StateDim == simulation.state().state_dims());
  ASSERT_TRUE(InputDim == simulation.state().input_dims());
  ASSERT_TRUE(BasePoseDim + JointsDim == simulation.state().q_dims());
  ASSERT_TRUE(BaseVelocityDim + JointsDim == simulation.state().u_dims());

  // Disable mujoco
  mujoco::exit();
}

TEST_F(MujocoSimulationTest, DefaultState) {
  // Enable mujoco
  mujoco::init();

  // Simulation
  MujocoSimulationConfig config;
  // Set the default MJCF file to load
  std::string modelFile = mujoco::assets() + "/benchmarks/humanoid.xml";
  NFATAL_IF(!boost::filesystem::exists(modelFile), "MJCF file not found: " << modelFile);
  config.model_file = modelFile;
  MujocoSimulation simulation(config);

  // Tolerance for comparison
  const double tol = 1e-5;

  // OpenAI-Gym variables
  Eigen::VectorXd defStateOpenAI{Eigen::VectorXd::Zero(StateDim)};
  defStateOpenAI << 0, 0, 1.4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  // Default state read from the file
  Eigen::VectorXd q = defStateOpenAI.head(BasePoseDim + JointsDim);
  Eigen::VectorXd u = defStateOpenAI.tail(BaseVelocityDim + JointsDim);

  // Compare openAI output and noesis output
  EXPECT_TRUE(q.isApprox(simulation.state().getGeneralizedCoordinates(), tol));
  EXPECT_TRUE(q.isApprox(simulation.state().getInitialConfiguration(), tol));
  EXPECT_TRUE(u.isApprox(simulation.state().getGeneralizedVelocities(), tol));

  // Disable mujoco
  mujoco::exit();
}

TEST_F(MujocoSimulationTest, NoActionStep) {
  // Enable mujoco
  mujoco::init();

  // Simulation
  MujocoSimulationConfig config;
  // Set the default MJCF file to load
  std::string modelFile = mujoco::assets() + "/benchmarks/humanoid.xml";
  NFATAL_IF(!boost::filesystem::exists(modelFile), "MJCF file not found: " << modelFile);
  config.model_file = modelFile;
  MujocoSimulation simulation(config);

  // Tolerance for comparison
  const double tol = 1e-5;
  const size_t control_decimation = 5;

  // Set random action as input (same input used in OpenAI-Gym)
  Eigen::VectorXd input{Eigen::VectorXd::Zero(InputDim)};
  input << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  // OpenAI-Gym variables
  Eigen::VectorXd nextStateOpenAI{Eigen::VectorXd::Zero(StateDim)};
  nextStateOpenAI
    << -5.45514261e-05, -2.40032065e-08, 1.39889095e+00, 9.99999976e-01, -3.58512777e-08, 2.20209540e-04, -6.30919082e-07, -3.69245407e-06,
    -3.22127320e-04, 8.10688065e-08, -4.02351021e-08, 1.08240024e-04, -2.64951618e-03, -6.00308102e-03, -1.96449099e-07, 1.05035547e-04,
    -2.64570362e-03, -6.00395841e-03, -2.73702546e-04, 3.71488928e-04, -1.05602657e-04, 2.75429423e-04, -3.68771177e-04, -1.03925304e-04,
    -8.00759934e-03, -1.82814884e-06, -1.47770012e-01, -3.94109156e-06, 5.81915118e-02, -1.27211239e-04, -1.98618736e-04, -8.19594595e-02,
    4.88229304e-06, -1.32034239e-05, 4.10486760e-03, -2.23422798e-01, -6.11352881e-01, -2.24043647e-05, 4.10771776e-03, -2.23151597e-01,
    -6.11459956e-01, -3.51648095e-02, 4.98753793e-02, -1.63077355e-02, 3.53242018e-02, -4.96105904e-02, -1.61558491e-02;

  Eigen::VectorXd qNextOpenAI = nextStateOpenAI.head(BasePoseDim + JointsDim);
  Eigen::VectorXd uNextOpenAI = nextStateOpenAI.tail(BaseVelocityDim + JointsDim);

  // Take steps in the environment
  for (size_t i = 0; i < control_decimation; i++) {
    simulation.step(input);
  }

  // Compare openAI output and noesis output
  EXPECT_TRUE(qNextOpenAI.isApprox(simulation.state().getGeneralizedCoordinates(), tol));
  EXPECT_TRUE(uNextOpenAI.isApprox(simulation.state().getGeneralizedVelocities(), tol));

  // Disable mujoco
  mujoco::exit();
}

TEST_F(MujocoSimulationTest, InputStep) {
  // Enable mujoco
  mujoco::init();

  // Simulation
  MujocoSimulationConfig config;
  // Set the default MJCF file to load
  std::string modelFile = mujoco::assets() + "/benchmarks/humanoid.xml";
  NFATAL_IF(!boost::filesystem::exists(modelFile), "MJCF file not found: " << modelFile);
  config.model_file = modelFile;
  MujocoSimulation simulation(config);

  // Tolerance for comparison
  const double tol = 1e-5;
  const size_t control_decimation = 5;

  // Set initial state (same initial state used in OpenAI-Gym)
  Eigen::VectorXd state{Eigen::VectorXd::Zero(StateDim)};
  state
    << -0.04503779, 0.06686046, 1.19597793, 0.9916029, 0.06669601, -0.10997836, -0.01341969, -0.18035836, -0.26849947, -0.61566317,
    0.0364492, -0.46384983, 0.3572792, -0.39231556, -0.12567007, -0.31351222, 0.36472884, -1.77008361, 0.58696806, -0.6266811,
    0.1540712, -0.36368075, 0.2900712, -0.19629479, -0.21795835, 0.60272346, -0.4329138, 0.45006854, 0.49476189, 0.0260076,
    0.3244291, -3.42480389, 0.03082472, -4.04920659, -0.40398075, -0.03307082, -8.3984227, 2.91677787, -2.45115724, -0.04807611,
    2.4686233, 0.64892197, -2.21360777, -4.93007556, -3.5056676, -0.01073742, -3.08268194;

  // Set random action as input (same input used in OpenAI-Gym)
  Eigen::VectorXd input{Eigen::VectorXd::Zero(InputDim)};
  input
    << 0.00313119, 0.06764893, 0.09859842, -0.2591232, -0.11619686, -0.15038764, -0.37001428, 0.11575765, -0.22126931, -0.3259395,
    -0.18004431, -0.3168321, 0.28332087, -0.01758415, 0.06072755, 0.27163836, 0.05947039;

  // OpenAI-Gym variables
  Eigen::VectorXd nextStateOpenAI{Eigen::VectorXd::Zero(StateDim)};
  nextStateOpenAI
    << -4.62061563e-02, 7.51908843e-02, 1.18972887e+00, 9.90804497e-01, 7.05840815e-02, -1.12560963e-01, -2.55805811e-02, -1.84190860e-01,
    -2.54371185e-01, -5.93437116e-01, -4.92663130e-03, -4.77876239e-01, 2.28672189e-01, -6.38757255e-01, -2.85547442e-02, -3.54415157e-01,
    2.55169794e-01, -1.78082121e+00, 6.00388346e-01, -6.47886157e-01, 7.39856419e-02, -4.08928690e-01, 3.35849902e-01, -2.14574838e-01,
    -1.06241500e-02, 5.15795194e-01, -5.13098823e-01, 1.59086963e-01, -5.25750522e-01, -2.93090069e+00, -5.18334117e-03, 2.75656625e+00,
    2.74215693e+00, -2.64697798e+00, -1.75355199e+00, -1.30789814e+01, -2.14873782e+01, 8.47920004e+00, -2.72381503e+00, -1.13865404e+01,
    -2.89778033e+00, 7.44056020e-01, -6.84105539e-02, -5.33504087e+00, -2.12222450e+00, 4.67094096e+00, 2.54364015e-01;

  // Extract the generalized coorindates and velocites from ground truth
  Eigen::VectorXd q = state.head(BasePoseDim + JointsDim);
  Eigen::VectorXd u = state.tail(BaseVelocityDim + JointsDim);
  Eigen::VectorXd qNextOpenAI = nextStateOpenAI.head(BasePoseDim + JointsDim);
  Eigen::VectorXd uNextOpenAI = nextStateOpenAI.tail(BaseVelocityDim + JointsDim);

  // Set state and input
  simulation.reset(state);
  EXPECT_TRUE(q.isApprox(simulation.state().getGeneralizedCoordinates(), tol));
  EXPECT_TRUE(u.isApprox(simulation.state().getGeneralizedVelocities(), tol));

  // Take steps in the environment
  for (size_t i = 0; i < control_decimation; i++) {
    simulation.step(input);
  }

  // Compare openAI output and noesis output
  EXPECT_TRUE(qNextOpenAI.isApprox(simulation.state().getGeneralizedCoordinates(), tol));
  EXPECT_TRUE(uNextOpenAI.isApprox(simulation.state().getGeneralizedVelocities(), tol));

  // Disable mujoco
  mujoco::exit();
}

} // namespace tests
} // namespace gym
} // namespace noesis

/* EOF */
