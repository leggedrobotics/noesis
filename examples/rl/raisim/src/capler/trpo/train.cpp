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
#include <noesis/rl/agent/TrpoAgent.hpp>
#include <noesis/rl/sample/TrajectorySampler.hpp>
#include <noesis/rl/train/Tester.hpp>
#include <noesis/rl/train/Runner.hpp>

// Environment
#include <noesis/gym/envs/raisim/capler/CaplerEnvironment.hpp>

int main(int argc, char** argv)
{
  // Expose namespace elements for code brevity
  using namespace noesis::utils;
  using namespace noesis::hyperparam;
  
  // Definitions
  using Graph = noesis::core::Graph;
  using Environment = noesis::gym::CaplerEnvironment;
  using Scalar = typename Environment::Scalar;
  using Agent = noesis::agent::TrpoAgent<Scalar>;
  using Sampler = noesis::algorithm::TrajectorySampler<Scalar>;
  using Tester = noesis::train::Tester<Scalar>;
  using Logger = noesis::log::TensorBoardLogger;
  using Runner = noesis::train::Runner<Scalar>;
  
  /*
   * Experiment options
   */
  
  // Declare the supported options.
  auto options = noesis::options();
  options.add_options()
    ("seed", noesis::po::value<int>(), "Sets the random seed used for the experiment.")
    ("batch_size", noesis::po::value<size_t>(), "Sets the number of workers used to collect samples for training.")
    ("iterations", noesis::po::value<size_t>(), "Sets the number of training iterations used for the experiment.")
    ("terminal_value", noesis::po::value<double>(), "Sets the value assigned for active termination conditions.")
    ("goal_noise_factor", noesis::po::value<double>(), "Sets the magnitude of the goal height randomization.")
    ("reset_noise_factor", noesis::po::value<double>(), "Sets the magnitude of the initial state randomization.")
    ("state_noise_factor", noesis::po::value<double>(), "Sets the magnitude of the state observation randomization.")
    ("randomization_factor", noesis::po::value<double>(), "Sets the magnitude of the dynamics model randomization.")
    ("graph_file", noesis::po::value<std::string>(), "Sets the Python script to use for graph generation.")
  ;
  
  // Define default option values
  int seed = 42;
  size_t batch_size = 16u;
  size_t iterations = 300u;
  double terminal_value = -5.0;
  double goal_noise_factor = 0.0;
  double reset_noise_factor = 1.0;
  double state_noise_factor = 1.0;
  double randomization_factor = 1.0;
  std::string graphFile = "graph.py";
  
  /*
   * Experiment configuration
   */
  
  auto args = noesis::init(argc, argv, options);
  if (args.count("seed")) { seed = args["seed"].as<int>(); }
  if (args.count("batch_size")) { batch_size = args["batch_size"].as<size_t>(); }
  if (args.count("iterations")) { iterations = args["iterations"].as<size_t>(); }
  if (args.count("terminal_value")) { terminal_value = args["terminal_value"].as<double>(); }
  if (args.count("goal_noise_factor")) { goal_noise_factor = args["goal_noise_factor"].as<double>(); }
  if (args.count("reset_noise_factor")) { reset_noise_factor = args["reset_noise_factor"].as<double>(); }
  if (args.count("state_noise_factor")) { state_noise_factor = args["state_noise_factor"].as<double>(); }
  if (args.count("randomization_factor")) { randomization_factor = args["randomization_factor"].as<double>(); }
  if (args.count("graph_file")) { graphFile = args["graph_file"].as<std::string>(); }
  NINFO("[Training]: seed: " << seed)
  NINFO("[Training]: batch_size: " << batch_size)
  NINFO("[Training]: iterations: " << iterations)
  NINFO("[Training]: terminal_value: " << terminal_value)
  NINFO("[Training]: goal_noise_factor: " << goal_noise_factor)
  NINFO("[Training]: reset_noise_factor: " << reset_noise_factor)
  NINFO("[Training]: state_noise_factor: " << state_noise_factor)
  NINFO("[Training]: randomization_factor: " << randomization_factor)
  NINFO("[Training]: graph_file: " << graphFile)
  
  // Configurations
  const std::string expDir = boost::filesystem::path(std::string(__FILE__)).parent_path().string();
  const std::string paramFile = "/parameters.xml";
  const std::string protoFile = "/graph.pb";
  const std::string graphDir = noesis::logpath() + "/graphs";
  const std::string srcDir = noesis::logpath() + "/src";
  const std::string scope = "Example";
  const bool verbose = false;
  
  // Hyper-parameters
  const double time_step = 0.01;
  const double time_limit = 5.0;
  const double discount_factor = 0.995;
  const bool use_pid_controller = true;
  const bool enable_logging = false;
  
  // 1. Create a vectorized environment
  auto environment = noesis::gym::make_vectorized<Environment>(
    batch_size,
    time_step,
    time_limit,
    discount_factor,
    terminal_value,
    randomization_factor,
    reset_noise_factor,
    state_noise_factor,
    goal_noise_factor,
    use_pid_controller,
    enable_logging,
    "Capler",
    scope,
    verbose
  );
  
  // 2. Add a TensorBoard logger
  auto logger = std::make_unique<Logger>("", "Capler", "Logger", scope, verbose);
  
  // 3. Create a graph
  auto graph = std::make_unique<Graph>(
    noesis::tf::SessionConfig(),
    "Capler",
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
  auto callback = [](size_t iter){ NINFO("[Training]: @Iteration: " << iter); };
  
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
  boost::filesystem::copy(expDir + "/" + graphFile, graphDir + "/" + graph->name() + "/" + graphFile);
  noesis::filesystem::copy_directory(expDir, srcDir);
  
  // Create a runnable graph from the target Python script and initialize all dependents (i.e. the agent).
  graph->generateFrom(graphDir + "/" + graph->name() + "/" + graphFile);
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
