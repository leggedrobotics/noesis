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

// C/C++
#include <cstdlib>
#include <csignal>
#include <sys/stat.h>
#include <time.h>
#include <iostream>
#include <iomanip>

#include <noesis/framework/log/message.hpp>

namespace noesis {
namespace log {
namespace message {

/*
 * Function member definitions for the MessageLogger class.
 */

MessageLogger::MessageLogger(size_t max_messages, bool auto_save_enabled):
  autoSave_(auto_save_enabled)
{
  // Configure the buffer
  messages_.clear();
  messages_.reserve(max_messages);
}

MessageLogger::~MessageLogger() {
  shutdown();
}

bool MessageLogger::startup(const std::string& filename) {
  if (!isActive_) {
    isActive_ = true;
    logFileName_ = filename;
    ofs_.open(logFileName_);
    if (ofs_.is_open()) {
      return true;
    }
  }
  // Fall-through signals that the logger is already active
  // NOTE: this means no action is taken
  return false;
}

void MessageLogger::shutdown() {
  if (isActive_) {
    if (!messages_.empty()) {
      write(); // final check
    }
    if (ofs_.is_open()) {
      ofs_.close();
    }
    isActive_ = false;
  }
}

void MessageLogger::append(const std::string& message) {
  if (isActive_) {
    std::unique_lock<std::mutex> lockAppend(appendMutex_);
    messages_.emplace_back(message);
    if ( (messages_.size() == messages_.capacity()) && autoSave_) {
      // release mutex in order to execute write func
      // NOTE: write() will also clear the buffer
      lockAppend.unlock();
      write();
    }
  }
}

void MessageLogger::write() {
  if (isActive_) {
    std::lock_guard<std::mutex> lockWriter(writerMutex_);
    std::lock_guard<std::mutex> lockAppend(appendMutex_);
    if (ofs_.is_open() && !messages_.empty()) {
      for (const auto& row : messages_) {
        ofs_ << row;
        ofs_.flush();
      }
      messages_.clear();
    }
    if (!ofs_.is_open() && !messages_.empty()) {
      std::cerr << "\nLog file not opened !" << std::endl;
    }
  }
}


/*
 * A global mutex to synchronize access to std::cout
 */
namespace internal {
static std::mutex mutex;
} // namespace internal

/*
 * Definition of the write_message helper function
 */
void write(const char *file, size_t line_number, std::stringstream& ss, MessageLevel level) {
  std::stringstream header;
  // Set type-identifier and color based on level
  std::string color;
  switch(level) {
    case MessageLevel::Info:
      header << "[INFO]";
      color = "";
      break;
    case MessageLevel::Notify:
      header << "[NOTIF]";
      color = "\033[34m";
      break;
    case MessageLevel::Warning:
      header << "[WARNING]";
      color = "\033[33m";
      break;
    case MessageLevel::Error:
      header << "[ERROR]";
      color = "\033[31m";
      break;
    case MessageLevel::Fatal:
      header << "[FATAL]";
      color = "\033[1;31m";
      break;
  }
  // Get the current time and convert the local time
  time_t t_now;
  tm tm_now;
  time( &t_now );
  localtime_r(&t_now, &tm_now);
  // Construct the message time-stamp and append it to the message header
  header << " [" << std::setfill('0')
    << std::setw(2) << 1 + tm_now.tm_mon << ':'
    << std::setw(2) << tm_now.tm_mday << ':'
    << std::setw(2) << tm_now.tm_hour << ':'
    << std::setw(2) << tm_now.tm_min << ':'
    << std::setw(2) << tm_now.tm_sec << "]";
  // If debug-mode is enabled, append the caller's info (filename and line-number) to the message header
  #ifdef NOESIS_DEBUG
  // Process the filename - strip the path and retain only the filename and extension
  const char* filename_start = file;
  const char* filename = filename_start;
  while (*filename != '\0') { filename++; }
  while ((filename != filename_start) && (*(filename - 1) != '/')) { filename--; }
  // Write the caller info
  header << " [" << std::setfill(' ') << filename << ':' << line_number << "]: ";
  #else
  header << ": ";
  #endif
  // Construct the output messages
  std::string message = header.str() + ss.str();
  // Write the message to the console
  {
    std::lock_guard<std::mutex> lock(internal::mutex);
    std::cout << color << message << "\033[0m" << std::endl;
  }
  // Write the message to log file (through global messageLogger instance)
  ::noesis::log::message::logger->append(message + "\n");
  // If the message is fatal, throw to terminate either the process or handle the error
  if (level == MessageLevel::Fatal) {
    ::noesis::log::message::logger->shutdown(); // shutdown the message logger
    throw;
  }
}

/*
 * Definition and initialization of the global MessageLogger instance.
 */
std::shared_ptr<noesis::log::message::MessageLogger> logger = std::make_shared<noesis::log::message::MessageLogger>();

} // namespace message
} // namespace log
} // namespace noesis

/* EOF */
