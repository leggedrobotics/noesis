/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_TRAIN_RUNNER_HPP_
#define NOESIS_RL_TRAIN_RUNNER_HPP_

// Noesis
#include "noesis/framework/system/time.hpp"
#include "noesis/framework/system/process.hpp"
#include "noesis/framework/log/metric.hpp"
#include "noesis/framework/log/tensorboard.hpp"
#include "noesis/framework/hyperparam/hyper_parameters.hpp"
#include "noesis/gym/train/RunnerInterface.hpp"

namespace noesis {
namespace train {

template <class ScalarType_>
class Runner final:
  public ::noesis::core::Object,
  public ::noesis::gym::RunnerInterface<ScalarType_>
{
public:
  
  // Aliases
  using Object = ::noesis::core::Object;
  using Interface = ::noesis::gym::RunnerInterface<ScalarType_>;
  using RunMode = typename Interface::RunMode;
  using Scalar = typename Interface::Scalar;
  using Metrics = typename Interface::Metrics;
  using SamplerPtr = typename Interface::SamplerPtr;
  using MonitorPtr = typename Interface::MonitorPtr;
  using LoggerPtr = typename Interface::LoggerPtr;
  using GraphPtr = typename Interface::GraphPtr;
  using IterationCallback = std::function<void(size_t)>;
  
  // Constants
  static constexpr int MaxInt = std::numeric_limits<int>::max();
  
  // Configuration helper
  struct Config {
    SamplerPtr sampler{nullptr};
    MonitorPtr monitor{nullptr};
    LoggerPtr logger{nullptr};
    GraphPtr graph{nullptr};
    IterationCallback callback;
    std::string name{"Runner"};
    std::string scope{"/"};
    bool verbose{false};
  };
  
  /*
   * Instantiation
   */

  Runner(Runner&& other) noexcept = default;
  Runner& operator=(Runner&& other) noexcept = default;
  
  Runner(const Runner& other) = delete;
  Runner& operator=(const Runner& other) = delete;
  
  explicit Runner(const Config& config=Config()):
    Runner(config.sampler, config.monitor, config.logger, config.graph,
     config.callback, config.name, config.scope, config.verbose)
  {
  }
  
  explicit Runner(
    SamplerPtr sampler,
    MonitorPtr monitor=nullptr,
    LoggerPtr logger=nullptr,
    GraphPtr graph=nullptr,
    IterationCallback callback=IterationCallback(),
    const std::string& name="Runner",
    const std::string& scope="/",
    bool verbose=false):
    Object(name, scope, verbose),
    Interface(),
    iterationsPerCheckpoint_(0, utils::make_namescope({scope, name,"/iterations_per_checkpoint"}), {0, MaxInt}),
    iterationCallback_(std::move(callback)),
    sampler_(sampler),
    monitor_(monitor),
    logger_(logger),
    graph_(graph)
  {
    // We add the local HPs to the global manager and will only remove them in the destructor.
    hyperparam::manager->addParameter(iterationsPerCheckpoint_);
  }
  
  ~Runner() override {
    // NOTE: We finally remove the HPs from the global manager to ensure
    // that they are not used while `run()` is active.
    hyperparam::manager->removeParameter(iterationsPerCheckpoint_);
  }
  
  /*
   * Configurations
   */

  void setSampler(SamplerPtr sampler) override {
    NFATAL_IF(!sampler, "[" << namescope() << "]: 'sampler' pointer argument is invalid (nullptr).");
    sampler_ = sampler;
  }
  
  void setGraph(GraphPtr graph) override {
    NFATAL_IF(!graph, "[" << namescope() << "]: 'graph' pointer argument nullptr!");
    graph_ = graph;
  }
  
  void setMonitor(MonitorPtr monitor) override {
    NFATAL_IF(!monitor, "[" << namescope() << "]: 'monitor' pointer argument is nullptr!");
    monitor_ = monitor;
  }
  
  void setLogger(LoggerPtr logger) override {
    NFATAL_IF(!logger, "[" << namescope() << "]: 'logger' pointer argument is nullptr!");
    logger_ = logger;
    logger_->setAutosaveEnabled(false);
  }
  
  void setIterationsPerCheckpoint(size_t iterations) {
    iterationsPerCheckpoint_ = static_cast<int>(iterations);
  }
  
  void setIterationCallback(const IterationCallback& callback) {
    iterationCallback_ = callback;
  }
  
  /*
   * Properties
   */
  
  const std::vector<size_t>& getSampleCounters() const override {
    return sampler_->getSampleCounters();
  }
  
  const std::vector<size_t>& getBatchCounters() const override {
    return sampler_->getBatchCounters();
  }

  size_t getTotalSamples() const override {
    return sampler_->getTotalSamples();
  }
  
  size_t getTotalBatches() const override {
    return sampler_->getTotalBatches();
  }
  
  size_t getTotalIterations() const override {
    return iterationsCounter_;
  }
  
  std::string info() const override {
    std::stringstream out;
    if (monitor_ && isVerbose()) { out << monitor_->info(); }
    out << sampler_->info();
    out << "\n  Iterations          : " << iterationsCounter_;
    out << "\n  Checkpoints         : " << checkpointsCounter_;
    out << "\n  Time Elapsed        : " << runElapsedTime_.toString() << " (h:m:s);";
    return out.str();
  }
  
  /*
   * Operations
   */
  
  void configure() override {
    const auto ns = namescope();
    NINFO("[" << ns << "]: Configuring runner ...");
    NINFO("[" << ns << "]: Iterations-per-checkpoint: " << static_cast<int>(iterationsPerCheckpoint_));
    NINFO("[" << ns << "]: Is verbose: " << std::boolalpha << isVerbose());
    NFATAL_IF(!sampler_, "[" << ns << "]: A sampler has not been set!");
    NWARNING_IF(!graph_, "[" << ns << "]: 'graph' is nullptr! No graph checkpoint-ing will be performed.");
    NWARNING_IF(!monitor_, "[" << ns << "]: 'monitor' is nullptr! No monitoring will be performed.");
    NWARNING_IF(!logger_, "[" << ns << "]: 'logger' is nullptr! No logging will be performed.");
    metrics_.clear();
    metrics_.push_back("Time/Runner/Sampling");
    metrics_.push_back("Time/Runner/Training");
    metrics_.push_back("Time/Runner/Elapsed");
    if (logger_) { metrics_.add_to(logger_); }
  }
  
  void reset() override {
    iterationsCounter_ = 0;
    checkpointsCounter_ = 0;
    sampler_->reset();
    if (monitor_) { monitor_->reset(); }
  }

  void run(RunMode mode, size_t duration) override {
    const auto name = namescope();
    NFATAL_IF(duration == 0, "[" << name << "]: Called `run()` with zero duration!")
    runStartTime_.reset();
    iterationStartTime_.reset();
    switch (mode) {
      case RunMode::Samples: {
        NNOTIFY("[" << name << "]: Starting run for '" << duration << "' samples.");
        train( [this, duration](){return hasSamples(duration);} );
      } break;
      case RunMode::Batches: {
        NNOTIFY("[" << name << "]: Starting run for '" << duration << "' batches.");
        train( [this, duration](){return hasBatches(duration);} );
      } break;
      case RunMode::Iterations: {
        NNOTIFY("[" << name << "]: Starting run for '" << duration << "' iterations.");
        train( [this, duration](){return hasIterations(duration);} );
      } break;
    }
    if (iterationsCounter_ > 0) { save(".final"); }
    NNOTIFY("[" << name << "]: Run completed.");
    NINFO(*this);
  }
  
  friend inline std::ostream& operator<<(std::ostream &out, const Runner& runner) {
    out << "[" << runner.namescope() << "]: Status:" << runner.info();
    return out;
  }

private:
  
  /*
   * Internal functions
   */
  
  inline bool hasSamples(size_t target) const { return sampler_->getTotalSamples() < target; }
  
  inline bool hasBatches(size_t target) const { return sampler_->getTotalBatches() < target; }
  
  inline bool hasIterations(size_t target) const { return iterationsCounter_ < target; }
  
  inline void save(const std::string& tag="") {
    if (graph_) {
      checkpointsCounter_++;
      const std::string file = "graph" + tag;
      const std::string path = noesis::logpath() + "/graphs/" + graph_->name() + "/checkpoints/graph" + tag;
      const std::string link = noesis::logpath() + "/graphs/" + graph_->name() + "/checkpoints/graph";
      graph_->saveTo(path);
      noesis::filesystem::set_symlink(file + ".data-00000-of-00001", link + ".data-00000-of-00001");
      noesis::filesystem::set_symlink(file + ".index", link + ".index");
      NINFO("[" << namescope() << "]: Saving graph checkpoint: Version '"
        << checkpointsCounter_ << "' at iteration '" << iterationsCounter_ << "'");
    }
  }
  
  template <typename ConditionType_>
  inline void train(ConditionType_&& condition) {
    bool running = true;
    // NOTE: We will continue to execute until one of the following conditions are met:
    // a) The `condition()` for termination is met (i.e. achieved target iterations, samples, etc)
    // b) A call to `shutdown()` is called from another thread while the calling thread is active in `run()`
    // c) The destructor is called, which, actually then calls `shutdown()`, so b) applies again.
    while (running) {
      // Step 1: Collect samples.
      const bool done = sampler_->sample();
      // Step 2: Check condition for termination of training loop.
      // NOTE: This is called after sampling because the conditions
      // must be evaluated using the latest state of the sampler.
      running &= condition();
      // Step 3: Process the collected samples once sample data criteria have been met.
      if (done) {
        // Step 3a: Process the samples.
        // NOTE: This is application specific and determined by
        // the implementation of the attached sampler.
        const auto samplingEndTime = Time::Now();
        sampler_->process();
        iterationsCounter_++;
        const int savingInterval = iterationsPerCheckpoint_;
        if (savingInterval != 0 && (iterationsCounter_) % savingInterval == 0) { save("." + std::to_string(iterationsCounter_)); }
        const auto trainingEndTime = Time::Now();
        // Step 3.1: Update the training monitor to collect performance statistics for this iteration.
        if (monitor_) { monitor_->update(); }
        // Step 3.2: Buffer the current elapsed run-time.
        // NOTE: This will also be reused in calls to `info()`.
        runElapsedTime_ = (trainingEndTime - runStartTime_);
        // Step 3.3: Store metrics to be later saved using the tensorboard logger.
        metrics_[Metric::SamplingDuration] = (samplingEndTime - iterationStartTime_).toSeconds();
        metrics_[Metric::TrainingDuration] = (trainingEndTime - samplingEndTime).toSeconds();
        metrics_[Metric::TimeElapsed] = runElapsedTime_.toSeconds();
        // Step 3.4: Optionally call the end-of-iteration callback for additional functionality.
        if (iterationCallback_) { iterationCallback_(iterationsCounter_); }
        // Step 3.5: Print the current training progress to console.
        NINFO("[" << namescope() << "]: Iteration no. '" << iterationsCounter_ << "':" << info());
        // Step 3.6: Collect all registered metrics metrics and update logging signals.
        if (logger_) {
          metrics_.append_to(logger_);
          logger_->flush();
        }
        // Step 3.7: If still running, prepare all components for another iteration.
        if (running) {
          sampler_->reset();
          if (monitor_) { monitor_->reset(); }
        }
        // Step 3.8: Reset clock on iteration duration timer.
        iterationStartTime_.reset();
      }
    }
  }
  
private:
  //! Defines indices for training performance metrics
  enum Metric {
    SamplingDuration = 0,
    TrainingDuration,
    TimeElapsed
  };
  //! @brief Container for metrics aggregated over all attached modules.
  Metrics metrics_;
  //! @brief HP defining the interval at which to generate graph checkpoints.
  hyperparam::HyperParameter<int> iterationsPerCheckpoint_;
  //! @brief Timer recording the start time of each call to `run()`.
  Time runStartTime_;
  //! @brief Timer recording intermediate time-stamps within each call of `run()`.
  Time runElapsedTime_;
  //! @brief Timer recording time-stamps and durations of run iterations.
  Time iterationStartTime_;
  //! @brief Auxiliary (optional) configurable callback called at the end of each iteration.
  IterationCallback iterationCallback_;
  //! @brief The sampler module used for collecting data.
  SamplerPtr sampler_{nullptr};
  //! @brief The monitor module used for application-specific logging.
  MonitorPtr monitor_{nullptr};
  //! @brief The TensorBoard logger used by all modules to record logging signals.
  LoggerPtr logger_{nullptr};
  //! @brief The graph operated throughout all calls to `run()`.
  GraphPtr graph_{nullptr};
  //! @brief Internal counter recording the number of executed sampling iterations.
  size_t iterationsCounter_{0};
  //! @brief Internal counter recording the number of graph checkpoints generated during each call to `run()`.
  size_t checkpointsCounter_{0};
};

} // namespace train
} // namespace noesis

#endif // NOESIS_RL_TRAIN_RUNNER_HPP_

/* EOF */
