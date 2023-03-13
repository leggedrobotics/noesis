/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    Mayank Mittal
 * @email     mittalma@student.ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_LOG_TIMING_HPP_
#define NOESIS_FRAMEWORK_LOG_TIMING_HPP_

// C/C++
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <unordered_map>

// Noesis
#include <noesis/framework/system/process.hpp>
#include <noesis/framework/system/time.hpp>
#include <noesis/framework/log/message.hpp>
#include <noesis/framework/core/Object.hpp>

namespace noesis {
namespace log {

/*!
 * @brief The MultiTimer class.
 *
 * This class provides an object which uses an unordered map to store
 * multiple timer objects and perform operations on them like adding the
 * timer to the unordered map, resetting them and measuring time intervals.
 */
class MultiTimer final: core::Object
{
public:
  
  /*
   * Instantiation
   */
  
  /*!
   * @brief Constructor enforcing initialization of the object's name and scope at construction time.
   * @param name The unique object name with the local scope.
   * @param scope The scope within which the object instance exists.
   * @param verbose Set to true to enable verbose output.
   */
  explicit MultiTimer(std::string name, std::string scope, bool verbose=false);

  /*!
   * @brief Default destructor.
   */
  ~MultiTimer() final = default;
  
  /*
   * Configurations
   */
  
  /*!
   * @brief Add a new timer to unordered map by name.
   * @param name The (unique) name of the new timer instance.
   */
  void addTimer(const std::string& name);
  
  /*
   * Properties
   */

  /*!
   * @brief Retrieves the time measurement of when the timer was reset.
   * @param name The (unique) name of the target timer instance.
   * @return Scalar double containing the (relative) time in seconds since system start.
   */
  double getResetTime(const std::string& name);
  
  /*!
   * @brief Retrieves the time measurement of when the timer was started.
   * @param name The (unique) name of the target timer instance.
   * @return Scalar double containing the (relative) time in seconds since system start.
   */
  double getStartTime(const std::string& name);
  
  /*!
   * @brief Retrieves the elapsed time measurement since the last time the timer was started.
   * @param name The (unique) name of the target timer instance.
   * @return Scalar double containing the (relative) time in seconds since the timer was started.
   */
  double getElapsedTime(const std::string& name);
  
  /*!
   * @brief Retrieves the total time since the timer was reset.
   * @param name The (unique) name of the target timer instance.
   * @return Scalar double containing the (relative) time in seconds since the timer was reset.
   */
  double getTotalTime(const std::string& name);
  
  /*
   * Operations
   */
  
  /*!
   * @brief Resets the timer and all internal states.
   * @param name The (unique) name of the target timer instance.
   */
  void reset(const std::string& name);
  
  /*!
   * @brief Starts a new relative measurement interval.
   * @param name The (unique) name of the target timer instance.
   */
  void start(const std::string& name);
  
  /*!
   * @brief Measures the elapsed time since the timer was last started.
   * @param name The (unique) name of the target timer instance.
   * @param restart Set to true if the relative timing measurement is to be reset.
   */
  void measure(const std::string& name, bool restart=false);
  
  /*!
   * @brief Measures the total elapsed time since the timer was reset.
   * @param name The (unique) name of the target timer instance.
   */
  void stop(const std::string& name);
  
  /*!
   * @brief Helper function to perform streaming operations on MultiTimer instances and write the internal states of all timers.
   * @param os The target output stream.
   * @param rhs The source MultiTimer instance.
   * @return The augmented target out stream.
   */
  friend inline std::ostream& operator<<(std::ostream& os, const MultiTimer& rhs) {
    os << "[" << rhs.namescope() << "]: Timers:\n";
    for (const auto& timer: rhs.timers_) {
      os << "[" << timer.first << "]:";
      os << "\n    Reset: " << timer.second.reset;
      os << "\n    Start: " << timer.second.start;
      os << "\n    Elapsed: " << timer.second.elapsed;
      os << "\n    Total: " << timer.second.total;
    }
    return os;
  }
  
private:

  /*!
   * @brief Internal helper function to check if the target name corresponds to an existing timer instance.
   * @param name The (unique) name of the target timer instance.
   * @warning This method throws a fatal error if the target name odes not exist.
   */
  void assertTimerExists(const std::string& name) const;

  /*!
   * @brief Internal type to represent a single timer instance.
   */
  struct Timer {
    Time reset;
    Time start;
    Time elapsed;
    Time total;
  };
  
private:
  //! @brief The set of named timers.
  std::unordered_map<std::string, Timer> timers_;
  //! @brief Internal mutex to synchronize access to the named timers.
  mutable std::mutex mutex_;
};

} // namespace log
} // namespace noesis

#endif // NOESIS_FRAMEWORK_LOG_TIMING_HPP_

/* EOF */
