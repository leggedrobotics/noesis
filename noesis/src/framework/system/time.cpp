/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    Philipp Leemann
 * @email     pleeman@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// C/C++
#include <sys/time.h>
#include <cstddef>
#include <vector>
#include <string>
#include <algorithm>

// Noesis
#include "noesis/framework/system/time.hpp"

namespace noesis {

void Time::clear() {
  time_.tv_sec = 0;
  time_.tv_nsec = 0;
}

Time& Time::reset() {
  *this = Now();
  return *this;
}

Time Time::elapsed() const {
  return (Now() - *this);
}

Time Time::operator-(const Time& other) const {
  int64_t sec = seconds() - other.seconds();
  int64_t nsec = nanoseconds() - other.nanoseconds();
  if (nsec < 0) {
    nsec += static_cast<int64_t>(1e9);
    sec--;
  }
  return {sec, nsec};
}

Time Time::operator+(const Time& other) const {
  auto sec = static_cast<uint64_t>(seconds() + other.seconds());
  auto nsec = static_cast<uint64_t>(nanoseconds() + other.nanoseconds());
  if (nsec > static_cast<uint64_t>(1e9)) {
    nsec -= static_cast<uint64_t>(1e9);
    sec++;
  }
  return {static_cast<int64_t>(sec), static_cast<int64_t>(nsec)};
}

double Time::toSeconds() const {
  return static_cast<double>(seconds()) + 1e-9 * static_cast<double>(nanoseconds());
}

double Time::toMiliSeconds() const {
  return 1e+3 * static_cast<double>(seconds()) + 1e-6 * static_cast<double>(nanoseconds());
}

double Time::toMicroSeconds() const {
  return 1e+6 * static_cast<double>(seconds()) + 1e-3 * static_cast<double>(nanoseconds());
}

double Time::toNanoSeconds() const {
  return 1e+9 * static_cast<double>(seconds()) + static_cast<double>(nanoseconds());
}

std::string Time::toString() const {
  std::time_t epoch = seconds();
  std::stringstream ss;
  ss << std::put_time(std::gmtime(&epoch), "%H:%M:%S");
  return ss.str();
}

Time::Time(int64_t seconds, int64_t nanoseconds):
  time_{seconds, nanoseconds}
{
}

} // namespace noesis

/* EOF */
