/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_SYSTEM_PROCESS_HPP_
#define NOESIS_FRAMEWORK_SYSTEM_PROCESS_HPP_

// C/C++
#include <string>

namespace noesis {

/*!
 * @brief Retrieves the name of the current noesis process instance.
 * @return A string containing the process name.
 */
extern std::string procname();

/*!
 * @brief Retrieves the name of the current noesis process instance.
 * @return A string containing the process name.
 */
extern std::string rootpath();

/*!
 * @brief Retrieves the name of the current noesis process instance.
 * @return A string containing the process name.
 */
extern std::string datapath();

/*!
 * @brief Retrieves the filesystem path to the current root directory set for the application.
 * @return A string containing the absolute path to the current logging directory.
 */
extern std::string procpath();

/*!
 * @brief Retrieves the filesystem path to the current logging directory set for this run.
 * @return A string containing the absolute path to the current logging directory.
 */
extern std::string logpath();

/*
 * Noesis process internal operations
 */
namespace system { // TODO @vt: rename to `process`
namespace internal { // TODO @vt: invert these namespaces

//
// TODO: re-factor the process internals as a class with a single global instance
//

/*!
 * @brief Returns the global name assigned to the current process.
 * @return An std::string with name of the current process.
 */
extern const std::string& getProcessName();

/*!
 * @brief Sets the global name assigned to the current process.
 * @param name An std::string to set as the name of the process.
 * @return False if the process name has already been set.
 */
extern bool setProcessName(const std::string& name);

/*!
 * @brief Returns the path of the root output directory.
 * @return An std::string with the current data path.
 */
extern const std::string& getRootPath();

/*!
 * @brief Returns the path of the global data directory.
 * @return An std::string with the current data path.
 */
extern const std::string& getDataPath();

/*!
 * @brief Returns the path to the root directory of all logs for this process.
 * @return An std::string with the root logging path.
 */
extern const std::string& getProcessPath();

/*!
 * @brief Returns the path of the current logging directory of this process.
 * @return An std::string with the current logging path.
 */
extern const std::string& getLogPath();

/*!
 * @brief Sets the path of the current logging directory of this process.
 * @param name An std::string with the current logging path.
 * @return False if the process log path has already been set.
 */
extern bool initializePaths(const std::string& path);

/*!
 * @brief Indicates if the internals have been configured using `initializePaths()`.
 * @return True if internals have been set, False otherwise
 */
extern bool isInitialized();

} // namespace internal
} // namespace system
} // namespace noesis

#endif // NOESIS_FRAMEWORK_SYSTEM_PROCESS_HPP_

/* EOF */
