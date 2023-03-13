/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    Philipp Leemann
 * @email     pleeman@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_SYSTEM_TIME_HPP_
#define NOESIS_FRAMEWORK_SYSTEM_TIME_HPP_

// C/C++
#include <time.h>
#include <sys/time.h>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <vector>

namespace noesis {

class Time
{
public:

  /*!
   * @brief Default constructor
   */
  Time() = default;
  
  /*!
   * @brief Factory method for creating new time measurement instances set to the current time since system start (i.e. boot).
   * @return Time object set to the absolute time of the method call.
   */
  static auto Now() {
    Time time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &time.time_);
    return time;
  }
  
  /*
   * Operations
   */
  
  /*!
   * @brief Resets the internal state to zero.
   */
  void clear();
  
  /*!
   * @brief Resets the time measurement instance to the current time.
   * @return Reference to the current instance.
   */
  Time& reset();
  
  /*!
   * @brief Measures the time elapsed since the last (re)setting of the current time measurement instance.
   * @return A new time measurement object containing the measured time difference.
   */
  Time elapsed() const;
  
  /*!
   * @brief Operator for computing time-differences.
   * @param other Another time measurement instance to measure relative time difference.
   * @return Time object measuring a relative time difference between time instances.
   */
  Time operator-(const Time& other) const;
  
  /*!
   * @brief Operator for computing time-accumulation.
   * @param other Another time measurement instance to measure relative time accumulation.
   * @return Time object measuring a relative time accumulation amongst time instances.
   */
  Time operator+(const Time& other) const;
  
  /*
   * Conversions
   */

  /*!
   * @brief Converts the latest time measurement to seconds (s).
   * @return Measurement of the latest time measurement in seconds
   */
  double toSeconds() const;
  
  /*!
   * @brief Converts the latest time measurement to mili-seconds (ms).
   * @return Measurement of the latest time measurement in mili-seconds
   */
  double toMiliSeconds() const;
  
  /*!
   * @brief Converts the latest time measurement to micro-seconds (us).
   * @return Measurement of the latest time measurement in micro-seconds
   */
  double toMicroSeconds() const;
  
  /*!
   * @brief Converts the latest time measurement to nano-seconds (ns).
   * @return Measurement of the latest time measurement in nano-seconds
   */
  double toNanoSeconds() const;
  
  /*!
  * @brief Converts the latest time measurement to time format string `H:M:S`.
  * @return Measurement of the latest time measurement as a time format string `H:M:S`.
  */
  std::string toString() const;

  /*
   * Helper functions
   */

  /*!
   * @brief Helper streaming operation.
   * @param os The target output stream object.
   * @param time The source time measurement instance.
   * @return The augmented output stream object.
   */
  friend inline std::ostream& operator<<(std::ostream& os, const Time& time) {
    return os << std::fixed << std::setprecision(9) << time.toSeconds() << " (s)";
  }
  
  /*!
   * @brief Helper function for retrieving the measured integer seconds.
   * @return The integer seconds from timespec.
   */
  inline int64_t seconds() const {
    return static_cast<int64_t>(time_.tv_sec);
  }
  
  /*!
   * @brief Helper function for retrieving the measured integer nanoseconds.
   * @return The integer nanoseconds from timespec.
   */
  inline int64_t nanoseconds() const {
    return static_cast<int64_t>(time_.tv_nsec);
  }
  
private:
  
  /*!
   * @brief Internal constructor used for creating objects using time-spec-compatible fields.
   * @param seconds
   * @param nanoseconds
   */
  Time(int64_t seconds, int64_t nanoseconds);
  
private:
  //! @brief The internal representation of time using the POSIX.1b timespec struct object.
  timespec time_{0, 0};
};

} // namespace noesis

#endif // NOESIS_FRAMEWORK_SYSTEM_TIME_HPP_

/* EOF */
