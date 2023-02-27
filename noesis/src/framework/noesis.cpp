/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// C/C++
#include <cstdlib>
#include <iostream>

// Noesis
#include <noesis/noesis.hpp>

namespace  noesis {

namespace internal {

/*!
 * @brief Executes shut-down operations
 */
static void shutdown() {
  NNOTIFY("[" << system::internal::getProcessName() << "]:  Shutting down process ...");
  ::noesis::log::message::logger->shutdown();
}

/*!
 * @brief The default signal handler ensures proper termination of the message-logger
 */
static void default_atexit_callback() {
  ::noesis::internal::shutdown();
}

/*!
 * @brief The default signal handler ensures proper termination of the message-logger
 */
static void default_signal_handler(int signum) {
  // The default action is to terminate the process
  // NOTE: this will invoke the atexit callback which will ensure proper shutdown of all global noesis resources
  std::exit(signum);
}

} // namespace internal

/*
 * Noesis Process Operations
 */

Options options() {
  Options options("Arguments");
  options.add_options()
    ("help", "Prints a description of all program arguments.")
    ("log_path", po::value<std::string>(), "Sets the directory in which experiments are logged.")
    ("signal_handlers", po::value<bool>(), "Set to `false` to prevent `noesis::init()` from overriding default signal handlers.")
    ;
  return options;
}

Arguments init(int argc, char** argv, const Options& options, const std::string& name) {
  // Check if program options is default constructed and create opts using the helper if so.
  const auto opts = (options.options().empty()) ? ::noesis::options() : options;
  // Construct initial arguments from parsing command-line inputs.
  Arguments arguments;
  po::store(po::parse_command_line(argc, argv, opts), arguments);
  po::notify(arguments);
  // Determine the process name from the experiment path
  const std::string experiment = (name.empty()) ? boost::filesystem::path(argv[0]).filename().string() : name;
  // Print helpful console output if help is present.
  if (arguments.count("help")) {
    std::cout << opts << "\n";
    std::exit(1);
  }
  // Check if a logging path has been specified
  std::string log_path;
  if (arguments.count("log_path")) {
    log_path = arguments["log_path"].as<std::string>();
    NNOTIFY("[" << experiment << "]: log_path: " << log_path)
  }
  // Determine whether to overwrite signal handlers
  bool signal_handlers = false;
  if (arguments.count("signal_handlers")) {
    signal_handlers = arguments["signal_handlers"].as<bool>();
    NNOTIFY("[" << experiment << "]: signal_handlers: " << signal_handlers)
  }
  // Initialize all process paths
  noesis::init(experiment, log_path, signal_handlers);
  // Return the resulting arguments container.
  return arguments;
}

void init(const std::string& name, const std::string& path, bool install_signal_handlers) {
  // Protect from repeated initialization
  if (::noesis::system::internal::isInitialized()) {
    return;
  }
  // Set a custom process name if specified
  if (!name.empty()) {
    ::noesis::system::internal::setProcessName(name);
  }
  #ifndef NOESIS_DEBUG
  // Set environment variable to suppress TF logging messages
  setenv("TF_CPP_MIN_LOG_LEVEL","1", 1);
  #endif
  if (install_signal_handlers) {
    // Set a persistent atexit callback
    std::atexit(::noesis::internal::default_atexit_callback);
    // Configure the global signal-handler
    ::noesis::system::SignalHandler::bind(&::noesis::internal::default_signal_handler);
  }
  // Set the logging path for the current process
  ::noesis::system::internal::initializePaths(path);
  // Configure the global message logger
  ::noesis::log::message::logger->startup(logpath() + "/logs/" + "messages.log");
  // Print out the root and log path for user checking
  NINFO("[" << ::noesis::system::internal::getProcessName() << "]: Process directory: " << procpath());
  NINFO("[" << ::noesis::system::internal::getProcessName() << "]: Logging directory: " << logpath());
}

std::string exit_or_load_parameters(const std::string& filename, bool is_relative_path) {
  std::string file_path;
  boost::filesystem::path path;
  if (is_relative_path) {
    path = noesis::procpath();
    path /= filename;
  } else {
    path = filename;
  }
  file_path = path.string();
  if (!boost::filesystem::exists(path)) {
    boost::filesystem::create_directories(path.parent_path());
    ::noesis::hyperparam::manager->saveParametersToXmlFile(file_path);
    const auto& name = ::noesis::system::internal::getProcessName();
    NNOTIFY("[" << name << "]: Creating new hyper-parameter file at: " << file_path);
    NNOTIFY("[" << name << "]: Re-run executable once hyper-parameters have been configured.");
    std::exit(0);
  } else {
    // Create the hyper-parameter file if it does not exist
    ::noesis::hyperparam::manager->loadParametersFromXmlFile(file_path);
  }
  return file_path;
}

} // namespace noesis

/* EOF */

