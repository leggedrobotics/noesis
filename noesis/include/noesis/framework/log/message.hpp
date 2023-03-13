/*!
 * @author    HaoChih Lin
 * @email     hlin@ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_LOG_MESSAGE_HPP_
#define NOESIS_FRAMEWORK_LOG_MESSAGE_HPP_

// C/C++
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace noesis {
namespace log {
namespace message {

/*!
 * @brief The MessageLogger class.
 *
 * This class provides an object which can pass std::cout msg to logfile.
 * In order to support multi-thread application, this class satisfy
 * re-entrant and thread safe characteristics. The path of log file
 * should be specified by user through noesis::init() function.
 */
class MessageLogger
{
public:
  
  /*!
   * @brief Constructs an instance of the MessageLogger with configured buffer size and auto-saving.
   * @param max_messages The maximum number of messages bufferd before the logger will flush to file.
   * @param auto_save_enabled Set to true so that the message logger can periodically save messages to the respective log file.
   */
  explicit MessageLogger(size_t max_messages = 100, bool auto_save_enabled = true);
  
  /*!
   * @brief Default destructor
   * @note Calls `shutdown()` in order to ensure proper releasing of any held
   * file descriptors and other resources.
   */
  ~MessageLogger();
  
  /*!
   * @brief Create logfile dir (if needed), path and open it.
   * @note If file dir or path is not specified, then uses default values.
   * @return True is the the file specified by `filename` was opened successfully, otherwise False.
   */
  bool startup(const std::string& filename);
  
  /*!
   * @brief Check buffer vector then close the ofstream obj.
   */
  void shutdown();
  
  /*!
   * @brief Pushes message into buffer vector.
   * @note This function is thread safe.
   * @param message The message which will be stored in buffer vector.
   */
  void append(const std::string& message);
  
  /*!
   * @brief Write data stored in buffer vector then clear it.
   * @note This function is thread safe.
   */
  void write();

private:
  std::ofstream ofs_;
  std::vector<std::string> messages_;
  std::string logFileName_ = "";
  std::mutex appendMutex_;
  std::mutex writerMutex_;
  bool isActive_ = false;
  bool autoSave_ = true;
};

/*!
 * @brief enum list for different types of message printout.
 */
enum class MessageLevel : unsigned int
{
  Info = 0,
  Notify,
  Warning,
  Error,
  Fatal,
};

/*!
 * @brief Helper function for writing formatted, timed and logged messages to std::out
 * @note Message format: [<MSG_LEVEL>][<TIME_STAMP>][<CALLER_FILE>:<LINE_NUM>]: <MESSAGE>
 * @note This function is re-entrant
 * @param file The name of the file calling this helper - __FILE__ is recommended.
 * @param line_number The line number of the location at which the helper is called - __LINE__ is recommended.
 * @param ss The std::stringstream containing the message to be written.
 * @param level The severity (level) indicated by the message.
 */
extern void write(const char* file, size_t line_number, std::stringstream& ss, MessageLevel level);

/*!
 * @brief Declaration of the global messageLogger instance.
 */
extern std::shared_ptr<noesis::log::message::MessageLogger> logger;

} // namespace message
} // namespace log
} // namespace noesis

/*!
 * @brief Define common message streaming macro
 */
#define NMESSAGE(content, level) \
{ \
  ::std::stringstream __nmessage; \
  __nmessage << content; \
  ::noesis::log::message::write(__FILE__, __LINE__, __nmessage, level); \
}

// Direct message output
#define NINFO(content) NMESSAGE(content, ::noesis::log::message::MessageLevel::Info)
#define NNOTIFY(content) NMESSAGE(content, ::noesis::log::message::MessageLevel::Notify)
#define NWARNING(content) NMESSAGE(content, ::noesis::log::message::MessageLevel::Warning)
#define NERROR(content) NMESSAGE(content, ::noesis::log::message::MessageLevel::Error)
#define NFATAL(content) NMESSAGE(content, ::noesis::log::message::MessageLevel::Fatal)

// Checks and assertions
#define NINFO_IF(condition, content) if (condition) { NMESSAGE(content, ::noesis::log::message::MessageLevel::Info); }
#define NNOTIFY_IF(condition, content) if (condition) { NMESSAGE(content, ::noesis::log::message::MessageLevel::Notify); }
#define NWARNING_IF(condition, content) if (condition) { NMESSAGE(content, ::noesis::log::message::MessageLevel::Warning); }
#define NERROR_IF(condition, content) if (condition) { NMESSAGE(content, ::noesis::log::message::MessageLevel::Error); }
#define NFATAL_IF(condition, content) if (condition) { NMESSAGE(content, ::noesis::log::message::MessageLevel::Fatal); }

// Debug versions
#ifdef NOESIS_DEBUG
  // Direct message output
  #define DNINFO(content) NINFO(content)
  #define DNNOTIFY(content) NNOTIFY(content)
  #define DNWARNING(content) NWARNING(content)
  #define DNERROR(content) NERROR(content)
  #define DNFATAL(content) NFATAL(content)
  // Checks and assertions
  #define DNINFO_IF(condition, content) NINFO_IF(condition, content)
  #define DNNOTIFY_IF(condition, content) NNOTIFY_IF(condition, content)
  #define DNWARNING_IF(condition, content) NWARNING_IF(condition, content)
  #define DNERROR_IF(condition, content) NERROR_IF(condition, content)
  #define DNFATAL_IF(condition, content) NFATAL_IF(condition, content)
// Release versions
#else
  // Direct message output
  #define DNINFO(content)
  #define DNNOTIFY(content)
  #define DNWARNING(content)
  #define DNERROR(content)
  #define DNFATAL(content)
  // Checks and assertions
  #define DNINFO_IF(condition, content)
  #define DNNOTIFY_IF(condition, content)
  #define DNWARNING_IF(condition, content)
  #define DNERROR_IF(condition, content)
  #define DNFATAL_IF(condition, content)
#endif

#endif // NOESIS_FRAMEWORK_LOG_MESSAGE_HPP_

/* EOF */
