/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Noesis
#include <noesis/noesis.hpp>
#include <noesis/framework/core/Graph.hpp>
#include <noesis/gym/core/Vector.hpp>
#include <noesis/rl/agent/PpoAgent.hpp>
#include <noesis/rl/sample/TrajectorySampler.hpp>
#include <noesis/rl/train/Tester.hpp>
#include <noesis/rl/train/Runner.hpp>

// Environment
#include "mdp.hpp"

int main(int argc, char** argv)
{
  // Expose namespace elements for code brevity
  using namespace noesis::utils;
  using namespace noesis::hyperparam;
  
  // Definitions
  using Graph = noesis::core::Graph;
  using Environment = noesis::gym::Kinova3Environment;
  using Scalar = typename Environment::Scalar;
  using Agent = noesis::agent::PpoAgent<Scalar>;
  using Sampler = noesis::algorithm::TrajectorySampler<Scalar>;
  using Runner = noesis::train::Runner<Scalar>;
  using Tester = noesis::train::Tester<Scalar>;
  using Logger = noesis::log::TensorBoardLogger;
  
  /*
   * Experiment options
   */
  
  // Hyper-parameters
  int seed = 0;
  size_t batch_size = 64u;
  size_t iterations = 5000u;
  double time_step = 0.01;
  double time_limit = 2.0;
  double discount_factor = 0.995;
  double goal_noise_factor = 1.0;
  double reset_noise_factor = 1.0;
  double randomization_factor = 1.0;
  double observations_noise_factor = 1.0;
  bool use_pid_controller = true;
  bool use_simulator_pid = false;
  
  // Declare the supported options.
  auto options = noesis::options();
  options.add_options()
    ("seed", noesis::po::value<int>(), "Sets the number of workers used to collect samples for training.")
    ("batch_size", noesis::po::value<size_t>(), "Sets the number of workers used to collect samples for training.")
    ("iterations", noesis::po::value<size_t>(), "Sets the number of training iterations.")
    ("time_step", noesis::po::value<double>(), "Sets the MDP time discretization in seconds.")
    ("time_limit", noesis::po::value<double>(), "Sets the MDP time limit in seconds.")
    ("discount_factor", noesis::po::value<double>(), "Sets the MDP discount factor.")
    ("goal_noise_factor", noesis::po::value<double>(), "Sets the intensity of goal randomization.")
    ("reset_noise_factor", noesis::po::value<double>(), "Sets the intensity of initial state randomization.")
    ("randomization_factor", noesis::po::value<double>(), "Sets the intensity of physical model randomization.")
    ("observations_noise_factor", noesis::po::value<double>(), "Sets the intensity of observation noise.")
    ("use_pid_controller", noesis::po::value<bool>(), "Sets the MDP to use PID joint command actions.")
    ("use_simulator_pid", noesis::po::value<bool>(), "Sets the simulation to use RaiSim's internal PID controller.")
  ;
  
  /*
   * Experiment configuration
   */
  
  auto args = noesis::init(argc, argv, options);
  if (args.count("seed")) { seed = args["seed"].as<int>(); }
  if (args.count("batch_size")) { batch_size = args["batch_size"].as<size_t>(); }
  if (args.count("iterations")) { iterations = args["iterations"].as<size_t>(); }
  if (args.count("time_step")) { time_step = args["time_step"].as<double>(); }
  if (args.count("time_limit")) { time_limit = args["time_limit"].as<double>(); }
  if (args.count("discount_factor")) { discount_factor = args["discount_factor"].as<double>(); }
  if (args.count("goal_noise_factor")) { goal_noise_factor = args["goal_noise_factor"].as<double>(); }
  if (args.count("reset_noise_factor")) { reset_noise_factor = args["reset_noise_factor"].as<double>(); }
  if (args.count("randomization_factor")) { randomization_factor = args["randomization_factor"].as<double>(); }
  if (args.count("observations_noise_factor")) { observations_noise_factor = args["observations_noise_factor"].as<double>(); }
  if (args.count("use_pid_controller")) { use_pid_controller = args["use_pid_controller"].as<bool>(); }
  if (args.count("use_simulator_pid")) { use_simulator_pid = args["use_simulator_pid"].as<bool>(); }
  NINFO("[Training]: seed: " << seed)
  NINFO("[Training]: batch_size: " << batch_size)
  NINFO("[Training]: iterations: " << iterations)
  NINFO("[Training]: time_step: " << time_step)
  NINFO("[Training]: time_limit: " << time_limit)
  NINFO("[Training]: discount_factor: " << discount_factor)
  NINFO("[Training]: goal_noise_factor: " << goal_noise_factor)
  NINFO("[Training]: reset_noise_factor: " << reset_noise_factor)
  NINFO("[Training]: randomization_factor: " << randomization_factor)
  NINFO("[Training]: observations_noise_factor: " << observations_noise_factor)
  NINFO("[Training]: use_pid_controller: " << use_pid_controller)
  NINFO("[Training]: use_simulator_pid: " << use_simulator_pid)
  
  // Configurations
  const std::string expDir = boost::filesystem::path(std::string(__FILE__)).parent_path().string();
  const std::string paramFile = "/ppo.xml";
  const std::string graphFile = "/graph.py";
  const std::string protoFile = "/graph.pb";
  const std::string graphDir = noesis::logpath() + "/graphs";
  const std::string srcDir = noesis::logpath() + "/src";
  const std::string scope = "Kinova3";
  const bool verbose = false;
  
  // 1. Add a TensorBoard logger
  auto logger = std::make_unique<Logger>("", "Kinova3", "Logger", scope, verbose);
  
  // 2. Create a graph
  auto graph = std::make_unique<Graph>(
    noesis::tf::SessionConfig(),
    "Graph",
    scope,
    verbose
  );
  
  // 3. Create a vectorized environment
  auto environment = noesis::gym::make_vectorized<Environment>(
    batch_size,
    time_step,
    time_limit,
    discount_factor,
    goal_noise_factor,
    reset_noise_factor,
    randomization_factor,
    observations_noise_factor,
    use_pid_controller,
    use_simulator_pid,
    nullptr,
    false,
    "Environment",
    scope,
    verbose
  );
  
  // 4. Create an agent
  auto agent = std::make_unique<Agent>(
    graph.get(),
    logger.get(),
    "Agent",
    scope,
    environment->actions_spec(),
    environment->observations_spec(),
    environment->tasks(),
    environment->batch_size(),
    environment->max_steps()
  );
  
  // 5. Define a sampler
  auto sampler = std::make_unique<Sampler>(
    agent.get(),
    &agent->memory(),
    environment.get(),
    logger.get(),
    "Sampler",
    scope
  );
  
  // 6. Add a RL training monitor
  auto tester = std::make_unique<Tester>(
    agent.get(),
    environment.get(),
    logger.get(),
    5, // Episodes per test
    "Tester",
    scope,
    verbose
  );
  
  // 7. Define a callback function to be called after every PPO iteration.
  // Example: This can be used for additional logging and/or curriculum learning.
  auto callback = [&iterations](size_t iter){ NINFO("[Training]: Iteration i=" << iter << " of " << iterations) };
  
  // 8. Add a training runner
  auto runner = std::make_unique<Runner>(
    sampler.get(),
    tester.get(),
    logger.get(),
    graph.get(),
    callback,
    "Trainer",
    scope,
    verbose
  );
  
  // Create the hyper-parameter file if it does not exist.
  noesis::exit_or_load_parameters(expDir + paramFile, false);
  
  /*
   * Experiment execution
   */
  
  // Configure all objects based on the hyper-parameters we just loaded.
  environment->configure();
  agent->configure();
  sampler->configure();
  tester->configure();
  runner->configure();
  
  // Create and store experiment meta-data dirs and files
  boost::filesystem::create_directories(graphDir + "/" + graph->name());
  boost::filesystem::copy(expDir + graphFile, graphDir + "/" + graph->name() + "/graph.py");
  noesis::filesystem::copy_directory(expDir, srcDir);
  
  // Create a runnable graph from the target Python script and initialize all dependents (i.e. the agent).
  graph->generateFrom(graphDir + "/" + graph->name() + "/graph.py");
  graph->loadFrom(graphDir + "/" + graph->name() + protoFile);
  graph->startup();
  agent->initialize();
  
  // Seed the experiment
  environment->seed(seed);
  agent->seed(seed);
  
  // Train the PPO-based agent for a specified number of total sample transitions (steps)
  NNOTIFY("[Training]: Starting training ...")
  runner->reset();
  runner->run(Runner::RunMode::Iterations, iterations);
  NNOTIFY("[Training]: Training completed.")
  
  // Success
  return 0;
}

/* EOF */
