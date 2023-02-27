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
#include <iomanip>

// Noesis
#include <noesis/framework/utils/macros.hpp>
#include <noesis/framework/system/filesystem.hpp>
#include "noesis/framework/system/process.hpp"

namespace noesis {

std::string procname() {
  std::string processName = system::internal::getProcessName();
  return processName;
}

std::string rootpath() {
  return std::string(MAKE_STRING(NOESIS_ROOT));
}

std::string datapath() {
  std::string dataPath = system::internal::getDataPath();
  if (!dataPath.empty() && !boost::filesystem::exists(dataPath)) {
    boost::filesystem::create_directories(dataPath);
  }
  return dataPath;
}

std::string procpath() {
  std::string rootPath = system::internal::getProcessPath();
  if (!rootPath.empty() && !boost::filesystem::exists(rootPath)) {
    boost::filesystem::create_directories(rootPath);
  }
  return rootPath;
}

std::string logpath() {
  std::string logPath = system::internal::getLogPath();
  if (!logPath.empty() && !boost::filesystem::exists(logPath)) {
    boost::filesystem::create_directories(logPath);
  }
  return logPath;
}

namespace system {
namespace internal {

//! @brief Helper function used for generating the dated+timed logging directory name strings
static inline std::string getCurrentDataAndTime() {
  std::ostringstream dataAndTime;
  time_t t = time(0);   // get time now
  struct tm *now = localtime(&t);
  dataAndTime << (now->tm_year + 1900) << '-' << std::setfill('0') << std::setw(2) << (now->tm_mon + 1) << '-'
              << std::setw(2) << now->tm_mday << '-' << std::setw(2) << now->tm_hour << '-' << std::setw(2)
              << now->tm_min << '-' << std::setw(2) << now->tm_sec;
  return dataAndTime.str();
}

/*
 * Internal variables of the logging core configuring the noesis process instance.
 */

//! @brief Library internal variable holding the name of the current Noesis process.
static std::string processName_ = "unknown";

//! @brief Library internal variable which holds the current application directory.
static std::string rootPath_ = std::string(secure_getenv("HOME")) + "/.noesis/tmp";

//! @brief Library internal variable which holds the path to the global data directory.
static std::string dataPath_ = std::string(secure_getenv("HOME")) + "/.noesis/tmp/";

//! @brief Library internal variable which holds the current application directory.
static std::string processPath_ = std::string(secure_getenv("HOME")) + "/.noesis/tmp";

//! @brief Library internal variable which holds the current logging directory.
static std::string logPath_ = std::string(secure_getenv("HOME")) + "/.noesis/tmp/" + getCurrentDataAndTime();

//! @brief Library internal variable indicating if the process instance has been configured.
static bool isInitialized_ = false;

/*
 * Logging core operations
 */

const std::string& getProcessName() {
  return processName_;
}

bool setProcessName(const std::string& name) {
  if (processName_ != "unknown") {
    return false;
  }
  processName_ = name;
  return true;
}

const std::string& getRootPath() {
  return rootPath_;
}

const std::string& getDataPath() {
  return dataPath_;
}

const std::string& getProcessPath() {
  return processPath_;
}

const std::string& getLogPath() {
  return logPath_;
}

bool initializePaths(const std::string& path) {
  // First guard against multiple re-settings of the current log path
  if (isInitialized_) {
    return false;
  }
  // Check if a custom logging path has been specified
  if (!path.empty()) {
    boost::filesystem::path rootPath = ::noesis::filesystem::expand_relative_path(path).second;
    rootPath.remove_trailing_separator();
    rootPath_ = rootPath.string();
    dataPath_ = rootPath_;
    processPath_ = dataPath_;
  }
  else {
    // Default to the noesis logging path in the user's home directory
    rootPath_ = std::string(secure_getenv("HOME")) + "/.noesis";
    dataPath_ = rootPath_ + "/proc";
    processPath_ = dataPath_ + "/" + processName_;
  }
  // Set the instance-specific directory with appropriate date-time signature
  logPath_ = processPath_ + "/" + getCurrentDataAndTime();
  // Create and configure current log directories
  boost::filesystem::path logDir(logPath_);
  boost::filesystem::create_directories(logDir);
  // Set data logging path
  boost::filesystem::create_directory(logDir/"logs");
  // Return true to indicate successful initialization
  isInitialized_ = true;
  return true;
}

bool isInitialized() {
  return isInitialized_;
}

} // namespace internal
} // namespace system
} // namespace noesis

/* EOF */
