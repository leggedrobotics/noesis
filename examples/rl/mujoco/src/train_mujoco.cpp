/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

#include <noesis/noesis.hpp>
#include <noesis/rl/runner/train/batched/BatchedRunner.hpp>
#include <noesis/rl/runner/train/batched/sampler/OpenMpSampler.hpp>
#include "agents.hpp"
#include "environments.hpp"

int main(int argc, char **argv)
{
  // Expose namespace elements for code brevity
  using namespace noesis::utils;
  using namespace noesis::hyperparam;
  using namespace noesis_environments::mujoco;

  // Definitions
  using ScalarType = float;
  using SamplerType = noesis::runner::OpenMpSampler<ScalarType>;
  using RunnerType = noesis::runner::BatchedRunner<SamplerType>;
  using ComputationGraph = noesis::graph::ComputationGraph;
  
  // Find package dir
  // Retrieve package directory
  std::string rootDir = noesis::rootpath();
  NFATAL_IF(!boost::filesystem::exists(rootDir), "'NOESIS_ROOT' has not been set!");
  
  // Set constant configurations
  std::string trainConfig = rootDir + "/noesis_agents/examples/mujoco/resources/hyperparams/train.xml";
  std::string modelsConfig = rootDir + "/noesis_agents/examples/mujoco/resources/hyperparams/agent/model.xml";
  NFATAL_IF(!boost::filesystem::exists(trainConfig), "[Training]: Configuration file not found!");
  
  // Set defaults for modifiable configurations
  bool verbose = false;
  bool visualize = false;
  size_t numberOfInstances = 1;
  std::string graphPath;
  std::string description;
  std::string agentConfig = rootDir + "/noesis_agents/examples/mujoco/resources/hyperparams/agent/";
  std::string envConfig = rootDir + "/noesis_agents/examples/mujoco/resources/hyperparams/environment/";
  
  // Retrieve CLI arguments
  std::vector<std::string> args;
  if (argc > 1) {
    for (int i=0; i<argc; i++) {
      args.emplace_back(argv[i]);
    }
  }

  // Check values
  for (size_t k=0; k<args.size(); ++k) {
    if (args[k] == "--verbose") {
      verbose = true;
    } else if (args[k] == "--visualize") {
      visualize = true;
    } else if (args[k] == "--instances") {
      numberOfInstances = std::stoul(args[k+1]);
    } else if (args[k] == "--graph") {
      graphPath = args[k+1];
    } else if (args[k] == "--description") {
      description = args[k+1];
    }
  }

  // Initialize log paths
  noesis::init("train_mujoco_rl_agent");
  
  // Enable mujoco
  mujoco_physics::init();
  
  // Process configurations
  NINFO("[" << noesis::procname() << "]: Description: " << description);
  NINFO("[" << noesis::procname() << "]: Is verbose: " << std::boolalpha << verbose);
  NINFO("[" << noesis::procname() << "]: Visualize: "  << std::boolalpha << visualize);
  NINFO("[" << noesis::procname() << "]: Number of instances: " << numberOfInstances);
  NINFO_IF(!graphPath.empty(), "[" << noesis::procname() << "]: Source graph path: "  << graphPath);
  
  // Argument checks
  NFATAL_IF(!graphPath.empty() && !boost::filesystem::exists(graphPath), "[" << noesis::procname() << "]: Source graph path does not exist!");
  
  // Static configurations
  constexpr auto scope = "Examples";

  // Hyper-parameters of root scope
  constexpr int IntMax = std::numeric_limits<int>::max();
  HyperParameter<std::string> agentType("ppo", make_namescope({scope, "agent_type"}), {"ppo", "trpo"});
  HyperParameter<std::string> envType("hopper", make_namescope({scope, "environment_type"}), {"hopper", "halfcheetah", "walker2d", "ant", "humanoid"});
  HyperParameter<int> graphSeed(0, make_namescope({scope, "graph_seed"}), {0, IntMax});
  HyperParameter<int> maxStepsPerEpisode(0, make_namescope({scope, "max_steps_per_episode"}), {0, IntMax});
  HyperParameter<int> numberOfTrainingSamples(0, make_namescope({scope, "training_samples"}), {0, IntMax});
  noesis::hyperparam::manager->addParameter(agentType);
  noesis::hyperparam::manager->addParameter(envType);
  noesis::hyperparam::manager->addParameter(graphSeed);
  noesis::hyperparam::manager->addParameter(maxStepsPerEpisode);
  noesis::hyperparam::manager->addParameter(numberOfTrainingSamples);
  
  // Setup runner
  noesis::runner::RunnerConfig runnerConf;
  runnerConf.name = "Trainer";
  runnerConf.scope = scope;
  runnerConf.verbose = verbose;
  RunnerType trainer(runnerConf);
  
  // First load the training configurations
  manager->loadParametersFromXmlFile(trainConfig);
  
  // Setup graph
  noesis::graph::ComputationGraphConfig graphConf;
  graphConf.name = "agent_graph";
  graphConf.scope = scope;
  graphConf.verbose = verbose;
  std::unique_ptr<ComputationGraph> graph = std::make_unique<ComputationGraph>(graphConf);
  graph->setGraphPath(graphPath);
  
  // Setup environment
  MujocoEnvironmentConfig envConf;
  envConf.name = "Environment";
  envConf.scope = scope;
  envConf.number_of_instances = numberOfInstances;
  envConf.verbose = verbose;
  std::unique_ptr<noesis::environment::Environment<ScalarType>> environment;
  std::string envTypeName = envType;
  if (envTypeName == "hopper") {
    environment = std::make_unique<HopperEnvironment<ScalarType>>(envConf);
    auto* env = dynamic_cast<HopperEnvironment<ScalarType>*>(environment.get());
    env->setTask<hopper::BenchmarkTask>();
    envConfig += "hopper.xml";
  } else if (envTypeName == "halfcheetah") {
    environment = std::make_unique<HalfcheetahEnvironment<ScalarType>>(envConf);
    auto* env = dynamic_cast<HalfcheetahEnvironment<ScalarType>*>(environment.get());
    env->setTask<halfcheetah::BenchmarkTask>();
    envConfig += "halfcheetah.xml";
  } else if (envTypeName == "walker2d") {
    environment = std::make_unique<Walker2dEnvironment<ScalarType>>(envConf);
    auto* env = dynamic_cast<Walker2dEnvironment<ScalarType>*>(environment.get());
    env->setTask<walker2d::BenchmarkTask>();
    envConfig += "walker2d.xml";
  } else if (envTypeName == "ant") {
    environment = std::make_unique<AntEnvironment<ScalarType>>(envConf);
    auto* env = dynamic_cast<AntEnvironment<ScalarType>*>(environment.get());
    env->setTask<ant::BenchmarkTask>();
    envConfig += "ant.xml";
  } else if (envTypeName == "humanoid") {
    environment = std::make_unique<HumanoidEnvironment<ScalarType>>(envConf);
    auto* env = dynamic_cast<HumanoidEnvironment<ScalarType>*>(environment.get());
    env->setTask<humanoid::BenchmarkTask>();
    envConfig += "ant.xml";
  } else {
    NFATAL("[Training]: Environment type specified is invalid: " << envTypeName);
  }
  NNOTIFY("[Training]: Creating '" << envTypeName << "' environment ...");
  
  // Setup agent
  noesis::agent::AgentConfig agentConf;
  agentConf.name = "Agent";
  agentConf.scope = scope;
  agentConf.number_of_instances = numberOfInstances;
  agentConf.verbose = verbose;
  std::unique_ptr<noesis::agent::TrajectoryAgent<ScalarType>> agent;
  std::string agentTypeName = agentType;
  if (agentTypeName == "ppo") {
    NNOTIFY("[Training]: Creating PPO-based agent.");
    agent = std::make_unique<noesis::agent::PpoAgent<ScalarType>>(graph.get(), agentConf);
    agentConfig += "ppo.xml";
  } else if (agentTypeName == "trpo") {
    NNOTIFY("[Training]: Creating TRPO-based agent.");
    agent = std::make_unique<noesis::agent::TrpoAgent<ScalarType>>(graph.get(), agentConf);
    agentConfig += "trpo.xml";
  } else {
    NFATAL("[Training]: Agent type specified is invalid: " << agentTypeName);
  }
  
  // Create a visualizer
  MujocoVisualizerConfig visConf;
  visConf.name = "Visualizer";
  visConf.scope = scope;
  visConf.real_time_factor = 1.0;
  visConf.use_render_thread = true;
  visConf.verbose = verbose;
  MujocoVisualizer visualizer(visConf);
  
  // Configure runner
  trainer.setGraph(graph.get());
  trainer.setAgent(agent.get());
  trainer.setEnvironment(environment.get());
  if (visualize) {
    trainer.setVisualizer(&visualizer);
  }
  
  // Create the hyper-parameter file if it does not exist
  manager->loadParametersFromXmlFile(trainConfig); // NOTE: we load this twice for the agent-related HPs to take effect
  manager->loadParametersFromXmlFile(agentConfig);
  manager->loadParametersFromXmlFile(modelsConfig);
  manager->loadParametersFromXmlFile(envConfig);
  
  // Store hyper-parameters used for the experiment before starting
  manager->saveParametersToXmlFile(noesis::logpath() + "/parameters.xml", true);

  // Set additional configurations
  trainer.setMaxStepsPerEpisode(static_cast<size_t>(maxStepsPerEpisode));
  agent->setMaxExperienceTransitions(static_cast<size_t>(maxStepsPerEpisode));
  agent->setSampleTransitionsPerIteration(numberOfInstances * agent->getSampleTransitionsPerIteration());
  
  // Configure the environment and retrieve the agent-environment interface specs
  environment->configure();
  auto obsSpec = environment->getObservationsSpecifications();
  auto actSpec = environment->getActionsSpecifications();
  
  // Configure the agent
  agent->setObservationsSpecifications(obsSpec);
  agent->setActionsSpecifications(actSpec);
  agent->configure();
  
  // Configure the runner
  trainer.configure();
  
  // Training configurations
  NINFO("[Training]: Graph seed: " << static_cast<int>(graphSeed));
  NINFO("[Training]: Max steps per episode: " << static_cast<int>(maxStepsPerEpisode));
  NINFO("[Training]: Total training samples: " << static_cast<int>(numberOfTrainingSamples));
  
  // Set final configurations for the graph and launch it
  graph->setSeed(static_cast<unsigned int>(graphSeed));
  graph->startup(); // NOTE: This either loads an existing graph or generates anew.

  // Startup all elements
  NNOTIFY("[Training]: Starting-up all components ...");
  agent->startup();
  environment->startup();
  trainer.startup();
  if (visualize) {
    noesis_environments::mujoco::MujocoSimulation* simulation{nullptr};
    if (envTypeName == "hopper") {
    } else if (envTypeName == "halfcheetah") {
      auto* env = dynamic_cast<HalfcheetahEnvironment<ScalarType>*>(environment.get());
      simulation = &env->getDynamics(0).getSimulation();
    } else if (envTypeName == "walker2d") {
      auto* env = dynamic_cast<Walker2dEnvironment<ScalarType>*>(environment.get());
      simulation = &env->getDynamics(0).getSimulation();
    } else if (envTypeName == "ant") {
      auto* env = dynamic_cast<AntEnvironment<ScalarType>*>(environment.get());
      simulation = &env->getDynamics(0).getSimulation();
    } else if (envTypeName == "humanoid") {
      auto* env = dynamic_cast<HumanoidEnvironment<ScalarType>*>(environment.get());
      simulation = &env->getDynamics(0).getSimulation();
    } else {
      NFATAL("[Training]: Environment type specified is invalid: " << envTypeName);
    }
    visualizer.setSimulation(simulation);
    visualizer.startup();
    visualizer.setBodyForCameraTracking("torso");
  }
  
  // Train the agent
  NNOTIFY("[Training]: Starting training ...");
  trainer.reset();
  trainer.run(noesis::runner::RunMode::SampleTransitions, static_cast<size_t>(numberOfTrainingSamples));
  NNOTIFY("[Training]: Training completed.");


  // Shutdown all elements
  if (visualize) {
    visualizer.shutdown();
  }
  trainer.shutdown();
  environment->shutdown();
  agent->shutdown();
  graph->shutdown();
  
  // Disable mujoco
  mujoco_physics::exit();
  
  // Success
  return 0;
}

/* EOF */
