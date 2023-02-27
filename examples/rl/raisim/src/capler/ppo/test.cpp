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
#include <noesis/gym/core/Vector.hpp>
#include <noesis/rl/agent/PpoAgent.hpp>

// Environment
#include <noesis/gym/envs/raisim/capler/CaplerEnvironment.hpp>
#include "noesis/gym/envs/raisim/capler/CaplerVisualizer.hpp"

int main(int argc, char** argv)
{
  // Expose namespace elements for code brevity
  using namespace noesis::utils;
  using namespace noesis::hyperparam;
  
  // Definitions
  using Graph = noesis::core::Graph;
  using Logger = noesis::log::TensorBoardLogger;
  using Environment = noesis::gym::CaplerEnvironment;
  using Visualizer = noesis::gym::CaplerVisualizer;
  using Agent = noesis::agent::PpoAgent<typename Environment::Scalar>;
  using Termination = noesis::mdp::Termination<typename Environment::Termination>;
  
  // Initialize log paths and other noesis internals
  // NOTE: This creates the process name automatically from argv[0]
  noesis::init(argc, argv);
  
  /*
   * Experiment configuration
   */
  
  // Configurations
  const std::string scope = "Example";
  const bool verbose = false;
  
  // Hyper-parameters
  const double time_step = 0.01;
  const double time_limit = 5.0;
  const double discount_factor = 0.995;
  const double terminal_value = -5.0;
  const double randomization_factor = 0.0;
  const double reset_noise_factor = 0.0;
  const double state_noise_factor = 0.0;
  const double goal_noise_factor = 0.0;
  const bool use_pid_controller = true;
  const bool enable_logging = true;
  
  // 1. Create an environment
  Environment environment(
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
  
  // 2. Create a graph
  auto graph = std::make_unique<Graph>(
    noesis::tf::SessionConfig(),
    "Capler",
    scope,
    verbose
  );
  
  // 3. Create an agent
  Agent agent(
    graph.get(),
    "Agent",
    scope,
    environment.actions_spec(),
    environment.observations_spec(),
    environment.tasks(),
    environment.batch_size(),
    environment.max_steps()
  );
  
  /*
   * Experiment execution
   */
  
  // Configure all objects based on the hyper-parameters we just loaded.
  environment.configure();
  agent.configure();
  
  // Create a runnable graph from the target Python script and initialize all dependents (i.e. the agent).
  const std::string resourceDir = noesis::datapath();
  const std::string experimentDir = "/noesis_rl_train_capler_ppo_example/2023-02-20-05-16-18";
  const std::string graphDir = "/graphs/Capler";
  graph->loadFrom(resourceDir + experimentDir + graphDir + "/graph.pb");
  graph->startup();
  agent.initialize();

  // Restore an existing graph from checkpoints.
  graph->restoreFrom(resourceDir + experimentDir + graphDir + "/checkpoints/graph");
  
  noesis::gym::CaplerVisConfig visConfig;
  visConfig.window_width = 1920;
  visConfig.window_height = 1080;
  visConfig.anti_aliasing = 8;
  visConfig.show_goal = true;
  Visualizer visualizer(visConfig, environment.simulation());
  visualizer.launch();
  
  // Start the test
  NNOTIFY("[Testing]: Starting test ...");
  environment.seed(42);
  environment.reset();
  visualizer.update(environment.goal());
  visualizer.startRecording(noesis::logpath() + "/videos", "test");
  for (int t = 0; t < 10 * environment.max_steps(); ++t) {
    agent.act(environment.observations(), environment.actions());
    environment.step();
    if (static_cast<Termination::Type>(environment.terminations().back().type) != Termination::Type::Unterminated) {
      NWARNING("[Tester]: Resetting environment!");
      environment.reset();
    }
    visualizer.update(environment.goal());
  }
  visualizer.stopRecording();
  NNOTIFY("[Testing]: Test completed.");

  // Success
  return 0;
}

/* EOF */
