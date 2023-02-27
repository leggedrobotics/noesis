/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_SYSTEM_FILESYSTEM_HPP_
#define NOESIS_FRAMEWORK_SYSTEM_FILESYSTEM_HPP_

// C/C++
#include <iostream>

// Boost
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>

namespace noesis {
namespace filesystem {

static inline void set_symlink(const std::string& path, const std::string& link) {
  ::boost::filesystem::remove(link);
  ::boost::filesystem::create_symlink(path, link);
}

static inline std::pair<bool, std::string> expand_relative_path(std::string path) {
  bool result = false;
  if (!path.empty()) {
    if (path[0] == '~') {
      path.replace(0, 1, getenv("HOME"));
      result = true;
    } else if (path[0] == '.') {
      path.replace(0, 1, getenv("PWD"));
      result = true;
    }
  }
  return {result, path};
}

static inline std::pair<bool, std::string> filename_from_path(std::string path) {
  bool result = false;
  std::string filename;
  if (!path.empty()) {
    boost::filesystem::path filePath(path);
    filePath = boost::filesystem::change_extension(filePath, "").filename();
    filename = filePath.string();
    result = true;
  }
  return {result, filename};
}

static inline void copy_directory(const boost::filesystem::path& source, const boost::filesystem::path& destination) {
  if (!boost::filesystem::exists(source) || !boost::filesystem::is_directory(source)) {
    throw std::runtime_error("noesis::filesystem::copy: path '" + source.string() + "' does not exist or is not a directory!");
  }
  if (boost::filesystem::exists(destination)) {
    throw std::runtime_error("noesis::filesystem::copy: destination path '" + destination.string() + "' already exists!");
  }
  if (!boost::filesystem::create_directories(destination)) {
    throw std::runtime_error("noesis::filesystem::copy: cannot create destination path '" + destination.string() + "'!");
  }
  for (const auto& dirEnt : boost::filesystem::recursive_directory_iterator{source}) {
    const auto& path = dirEnt.path();
    auto relativePathStr = path.string();
    boost::replace_first(relativePathStr, source.string(), "");
    boost::filesystem::copy(path, destination / relativePathStr);
  }
}

} // filesystem
} // noesis

#endif // NOESIS_FRAMEWORK_SYSTEM_FILESYSTEM_HPP_

/* EOF */
