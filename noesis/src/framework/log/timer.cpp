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

// Noesis
#include "noesis/framework/log/timer.hpp"

namespace noesis {
namespace log {

MultiTimer::MultiTimer(std::string name, std::string scope, bool verbose):
  core::Object(name, scope, verbose),
  timers_(),
  mutex_()
{
}

void MultiTimer::addTimer(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto elem = timers_.find(name);
  NFATAL_IF(elem != timers_.end(), "[" << namescope() << "]: Timer '" << name  << "' already exists!");
  timers_.insert({name, Timer()});
  NINFO_IF(isVerbose(), "[" << namescope() << "]: Adding new timer '" << name  << "'.");
}

double MultiTimer::getResetTime(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  assertTimerExists(name);
  return timers_[name].reset.toSeconds();
}

double MultiTimer::getStartTime(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  assertTimerExists(name);
  return timers_[name].start.toSeconds();
}

double MultiTimer::getElapsedTime(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  assertTimerExists(name);
  return timers_[name].elapsed.toSeconds();
}

double MultiTimer::getTotalTime(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  assertTimerExists(name);
  return timers_[name].total.toSeconds();
}

void MultiTimer::reset(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  assertTimerExists(name);
  timers_[name].reset.reset();
  timers_[name].start = timers_[name].reset;
  timers_[name].elapsed.clear();
  timers_[name].total.clear();
}

void MultiTimer::start(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  assertTimerExists(name);
  timers_[name].start.reset();
}

void MultiTimer::measure(const std::string& name, bool restart) {
  std::lock_guard<std::mutex> lock(mutex_);
  assertTimerExists(name);
  auto now = Time::Now();
  timers_[name].elapsed = now - timers_[name].start;
  if (restart) {
    timers_[name].start = now;
  }
}

void MultiTimer::stop(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  assertTimerExists(name);
  timers_[name].total = Time::Now() - timers_[name].reset;
}

void MultiTimer::assertTimerExists(const std::string& name) const {
  auto elem = timers_.find(name);
  NFATAL_IF(elem == timers_.end(), "[" << namescope() << "]: Timer '" << name  << "' does not exist!");
}

} // namespace log
} // namespace noesis

/* EOF */
