/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_NOESIS_HPP_
#define NOESIS_NOESIS_HPP_

// Boost
#include <boost/program_options.hpp>

// Noesis
#include "noesis/framework/system/version.hpp"
#include "noesis/framework/system/tensorflow.hpp"
#include "noesis/framework/system/filesystem.hpp"
#include "noesis/framework/system/process.hpp"
#include "noesis/framework/system/signal.hpp"
#include "noesis/framework/system/time.hpp"
#include "noesis/framework/log/message.hpp"
#include "noesis/framework/log/timer.hpp"
#include "noesis/framework/log/tensorboard.hpp"
#include "noesis/framework/hyperparam/hyper_parameters.hpp"

namespace  noesis {

//! @brief Local namespace alias for user's convenience
namespace po = boost::program_options;

//! @brief Local type alias for shorter type-names
using Arguments = boost::program_options::variables_map;

//! @brief Local type alias for shorter type-names
using Options = boost::program_options::options_description;

/*!
 * @brief Constructor for Noesis program options.
 *
 * Default options:
 *  - help: Print program help output to console.
 *  - log_path: Root path for experiment logging.
 */
extern Options options();

/*!
 * @brief Initialize the noesis' per-process global resources from program options.
 * @param argc From `main(int argc, char** argv)`.
 * @param argv From `main(int argc, char** argv)`.
 * @param options Program options constructed using the `noesis::options()` helper.
 * @param name Name of the current process.
 * @return Arguments (as key-value) set constructed from the program options.
 */
extern Arguments init(int argc, char** argv, const Options& options=Options(), const std::string& name="");

/*!
 * @brief Initialize the noesis' per-process global resources.
 * @param name Name of the current process.
 * @param path The path to the process' logging directory.
 */
extern void init(const std::string& name="", const std::string& path="", bool install_signal_handlers=true);

/*!
 * @brief Helper function which either generates the process hyper-parameter configuration file, or loads one if it exists.
 * @param parameters_file The path to the file from where parameter values are to be loaded.
 * @param is_relative_path Flag to indicate if the path is relative to the framework sources or absolute in the system.
 * @return
 */
extern std::string exit_or_load_parameters(const std::string& filename, bool is_relative_path=true);

} // namespace noesis

#endif // NOESIS_NOESIS_HPP_

/* EOF */
