/*!
 * @author    Mayank Mittal
 * @email     mittalma@student.ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    David Hoeller
 * @email     dhoeller@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Module header
#include "noesis/framework/log/tensorboard.hpp"

// C/C++
#include <unistd.h>
#include <sys/time.h>

// Noesis
#include "noesis/framework/system/process.hpp"
#include "noesis/framework/log/message.hpp"

namespace noesis {
namespace log {

/*
 * TensorBoardSignal
 */

namespace internal {

TensorBoardSignal::TensorBoardSignal(size_t buffer_capacity, size_t starting_step):
  stepCounter_(starting_step)
{
  setCapacity(buffer_capacity);
}

void TensorBoardSignal::setCapacity(size_t buffer_capacity) {
  events_.clear();
  events_.reserve(buffer_capacity);
}

size_t TensorBoardSignal::getStep() const {
  return stepCounter_;
}

size_t TensorBoardSignal::getSize() const {
  return events_.size();
}

size_t TensorBoardSignal::getCapacity() const {
  return events_.capacity();
}

bool TensorBoardSignal::isFull() const {
  return (events_.size() == events_.capacity());
}

bool TensorBoardSignal::isEmpty() const {
  return events_.empty();
}

void TensorBoardSignal::reset() {
  events_.clear();
  stepCounter_ = 0;
}

bool TensorBoardSignal::append(const tensorflow::Event& event) {
  bool result = false;
  if (events_.size() < events_.capacity()) {
    stepCounter_++;
    events_.emplace_back(event);
    result = true;
  }
  return result;
}

bool TensorBoardSignal::write(tensorflow::EventsWriter& writer) {
  tensorflow::Status status;
  if (!isEmpty()) {
    for (const auto& event : events_) {
      writer.WriteEvent(event);
      status = writer.Flush();
      if (!status.ok()) {
        NERROR(status.ToString());
        return false;
      }
    }
    events_.clear();
  }
  return true;
}

} // namespace internal

/*
 * TensorBoardLogger
 */

TensorBoardLogger::TensorBoardLogger(const TensorBoardLoggerConfig& config):
  TensorBoardLogger(config.path_prefix, config.log_prefix, config.name, config.scope, config.verbose)
{
}

TensorBoardLogger::TensorBoardLogger(
    const std::string& path_prefix,
    const std::string& log_prefix,
    const std::string& name,
    const std::string& scope,
    bool verbose):
  core::Object(name, scope, verbose),
  writer_(utils::make_namescope({logpath(), "logs", path_prefix, log_prefix})),
  signals_(),
  writerMutex_(),
  signalsMutex_()
{
  if (!path_prefix.empty()) {
    boost::filesystem::create_directories(utils::make_namescope({logpath(), "logs", path_prefix}));
  } else {
    boost::filesystem::create_directories(utils::make_namescope({logpath(), "logs"}));
  }
  filename_ = writer_.FileName();
}

TensorBoardLogger::~TensorBoardLogger() {
  if (isActive_) {
    shutdown();
  }
}

void TensorBoardLogger::setAutosaveEnabled(bool enable) {
  autoSave_ = enable;
}

void TensorBoardLogger::addLoggingSignal(const std::string& name, size_t buffer_capacity, size_t starting_step) {
  std::lock_guard<std::mutex> lock(signalsMutex_);
  auto elem = signals_.find(name);
  NFATAL_IF(elem != signals_.end(), "[" << namescope() << "]: Signal '" << name  << "' already exists!");
  signals_.insert(std::make_pair(name, internal::TensorBoardSignal(buffer_capacity, starting_step)));
}

std::vector<std::string> TensorBoardLogger::getSignalNames() const {
  std::lock_guard<std::mutex> lock(signalsMutex_);
  std::vector<std::string> names;
  names.reserve(signals_.size());
  for (const auto& signal: signals_) {
    names.emplace_back(signal.first);
  }
  return names;
}

size_t TensorBoardLogger::getNumberOfSignals() const {
  std::lock_guard<std::mutex> lock(signalsMutex_);
  return signals_.size();
}

size_t TensorBoardLogger::getSignalStep(const std::string& name) {
  std::lock_guard<std::mutex> lock(signalsMutex_);
  assertSignalExists(name);
  return signals_[name].getStep();
}

size_t TensorBoardLogger::getSignalSize(const std::string& name) {
  std::lock_guard<std::mutex> lock(signalsMutex_);
  assertSignalExists(name);
  return signals_[name].getSize();
}

size_t TensorBoardLogger::getSignalCapacity(const std::string& name) {
  std::lock_guard<std::mutex> lock(signalsMutex_);
  assertSignalExists(name);
  return signals_[name].getCapacity();
}

void TensorBoardLogger::startup() {
  std::lock_guard<std::mutex> lockSignals(signalsMutex_);
  std::lock_guard<std::mutex> lockWriter(writerMutex_);
  NINFO("[" << namescope() << "]: Starting-up logger ...");
  isActive_ = true;
  auto status = writer_.Init();
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString());
  NWARNING_IF(isVerbose() & !autoSave_, "[" << namescope() << "]: Autosave is disabled. Flushing must be called manually.");
}

void TensorBoardLogger::shutdown() {
  NINFO("[" << namescope() << "]: Shutting-down logger ...");
  flush();
  std::lock_guard<std::mutex> lockSignals(signalsMutex_);
  std::lock_guard<std::mutex> lockWriter(writerMutex_);
  signals_.clear();
  auto status = writer_.Close();
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString());
  isActive_ = false;
}

void TensorBoardLogger::reset() {
  NINFO("[" << namescope() << "]: Resetting logger ...");
  flush();
  std::lock_guard<std::mutex> lockSignals(signalsMutex_);
  std::lock_guard<std::mutex> lockWriter(writerMutex_);
  signals_.clear();
  auto status = writer_.Flush();
  NFATAL_IF(!status.ok(), "[" << namescope() << "]: " << status.ToString());
}

bool TensorBoardLogger::appendImageString(const std::string& name, const std::string& image_string) {
  std::unique_lock<std::mutex> lock(signalsMutex_);
  assertSignalExists(name);
  bool result = true;
  auto& signal = signals_[name];
  // Create the tensorflow::Event
  tensorflow::Event event;
  event.set_wall_time(getWallClockTime());
  event.set_step(signal.getStep());
  auto* summaryValue = event.mutable_summary()->add_value();
  summaryValue->set_tag(name);
  // Add an image string to the summary value
  auto* summaryImage = summaryValue->mutable_image();
  summaryImage->set_encoded_image_string(image_string);
  // Write to events file
  result &= signal.append(event);
  NWARNING_IF(isVerbose() & !result,
              "[" << namescope() << "]: Failed to append datum: Buffer of signal '" << name << "' is full.");
  lock.unlock();
  result &= writeSignal(signal);
  return result;
}

bool TensorBoardLogger::flush(const std::string& name) {
  std::unique_lock<std::mutex> lock(signalsMutex_);
  assertSignalExists(name);
  lock.unlock();
  return flushSignal(signals_[name]);
}

bool TensorBoardLogger::flush() {
  std::unique_lock<std::mutex> lock(signalsMutex_);
  bool result = true;
  for (auto& signal : signals_) {
    if (!signal.second.isEmpty()) {
      lock.unlock();
      if (!flushSignal(signal.second)) {
        result = false;
        break;
      }
      lock.lock();
    }
  }
  return result;
}

void TensorBoardLogger::assertSignalExists(const std::string& name) const {
  auto elem = signals_.find(name);
  NFATAL_IF(elem == signals_.end(), "[" << namescope() << "]: Signal '" << name  << "' does not exist!");
}

void TensorBoardLogger::checkSignalAppendResult(const std::string& name, bool result) const {
  NWARNING_IF(isVerbose() & !result, "[" << namescope() << "]: Failed to append datum: Buffer of signal '" << name << "' is full.");
}

double TensorBoardLogger::getWallClockTime() const {
  timeval time {0,0};
  double result = 0.0;
  if (!gettimeofday(&time, nullptr)) {
    result = static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec) * 1e-6;
  }
  return result;
}

bool TensorBoardLogger::flushSignal(internal::TensorBoardSignal& signal) {
  std::lock_guard<std::mutex> lockWriter(writerMutex_);
  std::lock_guard<std::mutex> lockSignal(signalsMutex_);
  return signal.write(writer_);
}

bool TensorBoardLogger::writeSignal(internal::TensorBoardSignal& signal) {
  std::lock_guard<std::mutex> lockWriter(writerMutex_);
  std::lock_guard<std::mutex> lockSignal(signalsMutex_);
  bool result = true;
  if (signal.isFull() && autoSave_) {
    result = signal.write(writer_);
  }
  return result;
}

} // namespace log
} // namespace noesis

/* EOF */
