/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
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
#include <noesis/gym/envs/classic/pendulum/PendulumEnvironment.hpp>
#include <noesis/gym/envs/classic/pendulum/PendulumVisualizer.hpp>

int main(int argc, char** argv)
{
  // Expose namespace elements for code brevity
  using namespace noesis::utils;
  using namespace noesis::hyperparam;
  
  // Definitions
  using Graph = noesis::core::Graph;
  using Environment = noesis::gym::PendulumEnvironment;
  using Visualizer = noesis::gym::PendulumVisualizer;
  using Scalar = typename Environment::Scalar;
  using Termination = noesis::mdp::Termination<Scalar>;
  using Agent = noesis::agent::PpoAgent<Scalar>;
  using Sampler = noesis::algorithm::TrajectorySampler<Scalar>;
  using Tester = noesis::train::Tester<Scalar>;
  using Logger = noesis::log::TensorBoardLogger;
  using Runner = noesis::train::Runner<Scalar>;
  
  // Initialize log paths
  noesis::init("noesis_rl_train_pendulum_ppo_example");
  
  /*
   * Experiment configuration
   */
  
  // Configurations
  const std::string expDir = boost::filesystem::path(std::string(__FILE__)).parent_path().string();
  const std::string paramFile = "/parameters.xml";
  const std::string graphFile = "/graph.py";
  const std::string protoFile = "/graph.pb";
  const std::string graphDir = noesis::logpath() + "/graphs";
  const std::string srcDir = noesis::logpath() + "/src";
  const std::string videoDir = noesis::logpath() + "/videos";
  const std::string scope = "Example";
  const bool verbose = false;
  
  // Hyper-parameters
  const size_t batch_size = 16u;
  const double reset_noise_factor = 1.0;
  const double randomization_factor = 0.0;
  const double discount_factor = 0.99;
  HyperParameter<int> seed(0, make_namescope({scope, "seed"}), {0, std::numeric_limits<int>::max()});
  HyperParameter<int> samples(0, make_namescope({scope, "samples"}), {0, std::numeric_limits<int>::max()});
  noesis::hyperparam::manager->addParameter(seed);
  noesis::hyperparam::manager->addParameter(samples);
  
  // 1. Create a vectorized environment
  auto environment = noesis::gym::make_vectorized<Environment>(
    batch_size,
    reset_noise_factor,
    randomization_factor,
    discount_factor
  );
  
  // 2. Add a TensorBoard logger
  auto logger = std::make_unique<Logger>("", "Pendulum", "Logger", scope, verbose);
  
  // 3. Create a graph
  auto graph = std::make_unique<Graph>(
    noesis::tf::SessionConfig(),
    "Pendulum",
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
  boost::filesystem::copy(expDir + graphFile, graphDir + "/" + graph->name() + graphFile);
  noesis::filesystem::copy_directory(expDir, srcDir);
  
  // Create a runnable graph from the target Python script and initialize all dependents (i.e. the agent).
  graph->generateFrom(graphDir + "/" + graph->name() + graphFile);
  graph->loadFrom(graphDir + "/" + graph->name() + protoFile);
  graph->startup();
  agent->initialize();
  
  // Seed the experiment
  environment->seed(seed);
  agent->seed(seed);
  
  // Train the PPO-based agent for a specified number of total sample transitions (steps)
  NNOTIFY("[Training]: Starting training ...");
  runner->reset();
  runner->run(Runner::RunMode::Samples, static_cast<size_t>(samples));
  NNOTIFY("[Training]: Training completed.");
  
  // Add a visualizer to visualize the results of training
  auto env = noesis::gym::make_synchronized_wrapper(&environment->front());
  auto visualizer = std::make_unique<Visualizer>(env.get());
  visualizer->launch();
  
  // Execute demo episode
  NNOTIFY("[Training]: Starting test ...");
  env->seed(0);
  env->reset();
  agent->seed(0);
  agent->reset();
  visualizer->startRecording(videoDir, "test");
  for (size_t t = 0; t < 5 * env->max_steps(); ++t) {
    agent->act(env->observations(), env->actions());
    env->step();
    if (env->terminations().back().type != Termination::Type::Unterminated) { env->reset(); }
  }
  visualizer->stopRecording();
  NNOTIFY("[Training]: Test completed.");
  
  // Success
  return 0;
}

/* EOF */
