/*!
 * @author    Philipp Leemann
 * @email     pleeman@ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#include "noesis/framework/system/signal.hpp"

namespace noesis {
namespace system {

void SignalHandler::bind(int signum, const Handler& handler) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = handlers_.find(signum);
  if (it == handlers_.end()) {
    it = handlers_.emplace(std::make_pair(signum, std::list<Handler>())).first;
    std::signal(signum, &SignalHandler::signaled);
  }
  for (std::list<Handler>::const_iterator jt = it->second.begin(); jt != it->second.end(); ++jt) {
    if (jt->target_type().name() == handler.target_type().name()) {
      return;
    }
  }
  it->second.push_back(handler);
}

void SignalHandler::bind(int signum, void(*fp)(int)) {
  SignalHandler::bind(signum, std::bind(fp, std::placeholders::_1));
}

void SignalHandler::bind(const Handler& handler) {
  SignalHandler::bind(SIGINT, handler);  // shell Ctrl+C
  SignalHandler::bind(SIGTERM, handler); // shell command kill
  SignalHandler::bind(SIGABRT, handler); // invoked by abort();
  SignalHandler::bind(SIGFPE, handler);  // floating-point exceptions (e.g. divide by zero)
  SignalHandler::bind(SIGILL, handler);  // Illegal operation
  SignalHandler::bind(SIGQUIT, handler); // the QUIT character, usually C-'\'
  SignalHandler::bind(SIGHUP, handler);  // hang-up” signal is used to report that the user’s terminal is disconnected
  // SIGKILL cannot be handled
}

void SignalHandler::bind(void(*fp)(int)) {
  const Handler handler = std::bind(fp, std::placeholders::_1);
  SignalHandler::bind(SIGINT, handler);  // shell Ctrl+C
  SignalHandler::bind(SIGTERM, handler); // shell command kill
  SignalHandler::bind(SIGABRT, handler); // invoked by abort();
  SignalHandler::bind(SIGFPE, handler);  // floating-point exceptions (e.g. divide by zero)
  SignalHandler::bind(SIGILL, handler);  // Illegal operation
  SignalHandler::bind(SIGQUIT, handler); // the QUIT character, usually C-'\'
  SignalHandler::bind(SIGHUP, handler);  // hang-up” signal is used to report that the user’s terminal is disconnected
  // SIGKILL cannot be handled
}

void SignalHandler::unbind(int signum, const Handler& handler) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = handlers_.find(signum);
  if (it == handlers_.end()) {
    return;
  }
  for (auto jt = it->second.begin();jt != it->second.end(); ++jt) {
    if (jt->target_type().name() == handler.target_type().name()) { //jt->target
      it->second.erase(jt);
      if (it->second.empty()) {
        std::signal(signum, SIG_DFL);
      }
      return;
    }
  }
}

void SignalHandler::unbind(int signum, void(*fp)(int)) {
  SignalHandler::unbind(signum, std::bind(fp, std::placeholders::_1));
}

void SignalHandler::unbind(const Handler& handler) {
  SignalHandler::unbind(SIGINT, handler);
  SignalHandler::unbind(SIGTERM, handler);
  SignalHandler::unbind(SIGABRT, handler);
  SignalHandler::unbind(SIGFPE, handler);
  SignalHandler::unbind(SIGILL, handler);
  SignalHandler::unbind(SIGQUIT, handler);
  SignalHandler::unbind(SIGHUP, handler);
}

void SignalHandler::unbind(void(*fp)(int)) {
  const Handler handler = std::bind(fp, std::placeholders::_1);
  SignalHandler::unbind(SIGINT, handler);
  SignalHandler::unbind(SIGTERM, handler);
  SignalHandler::unbind(SIGABRT, handler);
  SignalHandler::unbind(SIGFPE, handler);
  SignalHandler::unbind(SIGILL, handler);
  SignalHandler::unbind(SIGQUIT, handler);
  SignalHandler::unbind(SIGHUP, handler);
}

void SignalHandler::signaled(int signum) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = handlers_.find(signum);
  if (it == handlers_.end()) {
    return;
  }
  for (auto& handler: it->second) {
    handler(signum);
  }
}

/*
 * Definition and allocation of static members
 */
std::map<int, std::list<SignalHandler::Handler>> SignalHandler::handlers_;
std::mutex SignalHandler::mutex_;

} // namespace system
} // namespace noesis

/* EOF */
