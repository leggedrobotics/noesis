/*!
 * @author    Philipp Leemann
 * @email     pleeman@ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_SYSTEM_SIGNAL_HPP_
#define NOESIS_FRAMEWORK_SYSTEM_SIGNAL_HPP_

#include <csignal>
#include <functional>
#include <list>
#include <map>
#include <mutex>

namespace noesis {
namespace system {

/*!
 * @brief Singleton class for handling system signals using installable event callbacks.
 */
class SignalHandler
{
public:
  //! @brief Signal handlers_ are provided by STL functionals
  using Handler = std::function<void(int)>;

  //! @brief The default constructor is deleted to make this class a singleton
  SignalHandler() = delete;

  /*
   * Interface for registration of external signal-handler call-backs
   */

  static void bind(int signum, const Handler& handler);
  static void unbind(int signum, const Handler& handler);

  static void bind(int signum, void(*fp)(int));
  static void unbind(int signum, void(*fp)(int));

  static void bind(const Handler& handler);
  static void unbind(const Handler& handler);

  static void bind(void(*fp)(int));
  static void unbind(void(*fp)(int));

  /*
   * Templated signal-handler registration - used to declare object function members as signal handlers
   */

  template <typename T>
  static void bind(int signum, void(T::*fp)(int), T* object) {
    SignalHandler::bind(signum, std::bind(fp, object, std::placeholders::_1));
  }

  template <typename T>
  static void bind(void(T::*fp)(int), T* object) {
    const Handler handler = std::bind(fp, object, std::placeholders::_1);
    SignalHandler::bind(SIGINT, handler);
    SignalHandler::bind(SIGTERM, handler); // shell command kill
    SignalHandler::bind(SIGABRT, handler); // invoked by abort();
    SignalHandler::bind(SIGFPE, handler);
    SignalHandler::bind(SIGILL, handler);
    SignalHandler::bind(SIGQUIT, handler); // the QUIT character, usually C-'\'
    SignalHandler::bind(SIGHUP, handler);  // hang-up” signal is used to report that the user’s terminal is disconnected
    // SIGKILL cannot be handled
  }

  template <typename T>
  static void unbind(int signum, void(T::*fp)(int), T* object) {
    SignalHandler::unbind(signum, std::bind(fp, object, std::placeholders::_1));
  }

  template <typename T>
  static void unbind(void(T::*fp)(int), T* object) {
    const Handler handler = std::bind(fp, object, std::placeholders::_1);
    SignalHandler::unbind(SIGINT, handler);
    SignalHandler::unbind(SIGTERM, handler);
    SignalHandler::unbind(SIGABRT, handler);
    SignalHandler::unbind(SIGFPE, handler);
    SignalHandler::unbind(SIGILL, handler);
    SignalHandler::unbind(SIGQUIT, handler);
    SignalHandler::unbind(SIGHUP, handler);
  }

protected:
  /*!
   * @brief The global signal handler which captures the configured signals and calls all configured handler callbacks.
   * @param signum The SIGx signal number.
   */
  static void signaled(int signum);

private:
  static std::map<int, std::list<Handler>> handlers_;
  static std::mutex mutex_;
};

} // namespace system
} // namespace noesis

#endif // NOESIS_FRAMEWORK_SYSTEM_SIGNAL_HPP_

/* EOF */
