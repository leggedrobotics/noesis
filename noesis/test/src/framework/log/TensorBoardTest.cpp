/*!
 * @author    Mayank Mittal
 * @email     mittalma@student.ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// C/C++
#include <random>

// google test
#include <gtest/gtest.h>

// Noesis
#include <noesis/framework/log/tensorboard.hpp>
#include <noesis/framework/log/message.hpp>
#include <noesis/framework/log/timer.hpp>

namespace noesis {
namespace tests {

/*
 * Test Fixture
 */

class TensorBoardSignalTest : public ::testing::Test
{
protected:
  TensorBoardSignalTest() = default;
  ~TensorBoardSignalTest() = default;
};

/*
 * Tests
 */

TEST_F(TensorBoardSignalTest, Creation) {
  log::internal::TensorBoardSignal signal(10, 0);
  EXPECT_EQ(0, signal.getSize());
  EXPECT_EQ(10, signal.getCapacity());
  EXPECT_EQ(0, signal.getStep());
  EXPECT_TRUE(signal.isEmpty());
  EXPECT_FALSE(signal.isFull());
  NINFO(signal);
}

TEST_F(TensorBoardSignalTest, AppendEvent) {
  log::internal::TensorBoardSignal signal(10, 0);
  tensorflow::Event event;
  signal.append(event);
  signal.append(event);
  signal.append(event);
  EXPECT_EQ(3, signal.getSize());
  EXPECT_EQ(10, signal.getCapacity());
  EXPECT_EQ(3, signal.getStep());
  EXPECT_FALSE(signal.isEmpty());
  EXPECT_FALSE(signal.isFull());
  NINFO(signal);
}

TEST_F(TensorBoardSignalTest, AppendEventTooManyEvents) {
  log::internal::TensorBoardSignal signal(5, 0);
  tensorflow::Event event;
  signal.append(event);
  signal.append(event);
  signal.append(event);
  signal.append(event);
  signal.append(event);
  EXPECT_FALSE(signal.append(event));
  EXPECT_EQ(5, signal.getSize());
  EXPECT_EQ(5, signal.getCapacity());
  EXPECT_EQ(5, signal.getStep());
  EXPECT_FALSE(signal.isEmpty());
  EXPECT_TRUE(signal.isFull());
  NINFO(signal);
}

/*
 * Test Fixture
 */

class TensorBoardLoggerTest : public ::testing::Test
{
protected:
  // Declare fixture aliases
  TensorBoardLoggerTest() = default;
  ~TensorBoardLoggerTest() = default;
};

/*
 * Tests
 */

TEST_F(TensorBoardLoggerTest, AddSignals) {
  // create logger object
  noesis::log::TensorBoardLogger logger("test", "test", "logger", "test", true);
  // add logging signals
  logger.addLoggingSignal("test/signal1", 10);
  logger.addLoggingSignal("test/signal2", 10);
  logger.addLoggingSignal("test/signal3", 10);
  // check number of signals being managed
  EXPECT_EQ(logger.getNumberOfSignals(), 3);
  // print status of all signals
  NINFO(logger);
}

TEST_F(TensorBoardLoggerTest, StartupAndShutdown) {
  // create logger object
  noesis::log::TensorBoardLogger logger("test", "test", "logger", "test", true);
  // add logging signals
  logger.addLoggingSignal("test/signal1", 10);
  logger.addLoggingSignal("test/signal2", 10);
  logger.addLoggingSignal("test/signal3", 10);
  // Startup
  logger.startup();
  usleep(10000);
  logger.shutdown();
  // print status of all signals
  NINFO(logger);
}

TEST_F(TensorBoardLoggerTest, LoggingSignalDoesNotExist) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  // create logger object
  noesis::log::TensorBoardLogger logger("test", "test", "logger", "test", true);
  // Append data to the signal that has not been added
  ASSERT_DEATH(logger.appendScalar("noexist", 0.0f), "");
  ASSERT_DEATH(logger.appendHistogram("noexist", std::vector<float>{0, 1, 2, 3}), "");
  ASSERT_DEATH(logger.appendImageString("noexist", "random_buffer_string_for_PNG"), "");
}

TEST_F(TensorBoardLoggerTest, DuplicateLoggingSignal) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  // create logger object
  noesis::log::TensorBoardLogger logger("test", "test", "logger", "test", true);
  // add logging signal
  logger.addLoggingSignal("duplicate", 10);
  logger.startup();
  // append data to the signal that has not been added
  ASSERT_DEATH({logger.addLoggingSignal("duplicate", 5);}, "");
}

TEST_F(TensorBoardLoggerTest, AppendScalarData) {
  // create logger object with autosave mode is turned off.
  // This means the flush needs to be called manually to write
  // events into file
  noesis::log::TensorBoardLogger logger("test", "scalars", "logger", "test", true);
  // add logging signal
  logger.addLoggingSignal("test/scalar", 10);
  logger.startup();
  int N = 8;
  for (int i = 0; i<N; i++) {
    ASSERT_TRUE(logger.appendScalar("test/scalar", i));
  }
  ASSERT_EQ(N, logger.getSignalSize("test/scalar"));
  ASSERT_TRUE(logger.flush("test/scalar"));
  ASSERT_EQ(0, logger.getSignalSize("test/scalar"));
}

TEST_F(TensorBoardLoggerTest, AppendHistogramData) {
  // create logger object with autosave mode is turned true
  noesis::log::TensorBoardLogger logger("test", "histograms", "logger", "test", true);
  // add logging signal
  logger.addLoggingSignal("test/histogram", 25);
  logger.startup();
  std::default_random_engine generator;
  std::normal_distribution<double> default_distribution(0, 1.0);

  int N = 401;
  for (int i = 0; i<N; ++i) {
    std::vector<float> values;
    // moving average distribution
    float mean = i * 5.0f / N;
    std::normal_distribution<float> distribution(mean, 1.0);
    // sample 1000 random samples from the distribution
    for (int j = 0; j < 1000; ++j) {
      values.push_back(distribution(generator));
    }
    EXPECT_TRUE(logger.appendHistogram("test/histogram", values));
  }
  EXPECT_EQ(1, logger.getSignalSize("test/histogram")); // Since autosave is enabled: 1 = 401 mod 25
  EXPECT_TRUE(logger.flush("test/histogram"));
  EXPECT_EQ(0, logger.getSignalSize("test/histogram"));
}

TEST_F(TensorBoardLoggerTest, AppendImageData) {
  // create logger object with autosave mode is turned on
  noesis::log::TensorBoardLogger logger("test", "image_matrix", "logger", "test", true);
  logger.addLoggingSignal("test/image", 3);
  logger.startup();
  Eigen::MatrixXf mat = Eigen::MatrixXf::Random(300, 150);
  EXPECT_TRUE(logger.appendImageMatrix("test/image", mat));
  EXPECT_TRUE(logger.appendImageMatrix("test/image", mat));
  EXPECT_EQ(2, logger.getSignalSize("test/image"));
}

TEST_F(TensorBoardLoggerTest, AppendImageString) {
  // create logger object with autosave mode is turned on
  noesis::log::TensorBoardLogger logger("test", "image_string", "logger", "test", true);
  // add logging signal with max buffer size 1
  // if autosave is on then data is added at each append call
  logger.addLoggingSignal("test/image", 1);
  logger.startup();
  // Reading an image file
  std::string filename = noesis::rootpath() + "/docs/images/noesis-logo.png";
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  std::ostringstream oss;
  oss << fin.rdbuf();
  std::string encoded_string(oss.str());
  // Add matrix image to logger
  EXPECT_TRUE(logger.appendImageString("test/image", encoded_string));
  // signal's buffer should now be empty
  EXPECT_EQ(0, logger.getSignalSize("test/image"));
}

TEST_F(TensorBoardLoggerTest, AppendTimerScalarData) {
  // create logger object with autosave mode is turned on
  noesis::log::TensorBoardLogger logger("test", "timings", "logger", "test", true);
  // create timer object
  noesis::log::MultiTimer timers("test", "timers", true);
  // add scalar signal for timer
  logger.addLoggingSignal("test/timer/loop", 10);
  logger.startup();
  // add timer
  timers.addTimer("loop");
  for (int j = 0; j < 500; ++j) {
    // start the timer
    timers.start("loop");
    usleep(1000);
    // measure the time taken by the timer
    timers.measure("loop");
    // Add data values to log
    EXPECT_TRUE(logger.appendScalar("test/timer/loop", timers.getElapsedTime("loop")));
  }
}

TEST_F(TensorBoardLoggerTest, AppendExcessDataWithoutAutoSave) {
  // create logger object with autosave mode is turned off.
  // This means the flush needs to be called manually to write
  // events into file
  noesis::log::TensorBoardLogger logger("test", "autosave", "logger", "test", true);
  logger.setAutosaveEnabled(false);
  // add logging signal
  size_t max_buffer = 10;
  logger.addLoggingSignal("autosave/scalar", max_buffer);
  logger.startup();
  EXPECT_EQ(max_buffer, logger.getSignalCapacity("autosave/scalar"));
  size_t N = 15;
  for (size_t i = 0; i<N; i++) {
    // Add data values to log
    if (i<max_buffer) {
      EXPECT_TRUE(logger.appendScalar("autosave/scalar", i));
    } else {
      // when buffer is full, data cannot be appended
      EXPECT_FALSE(logger.appendScalar("autosave/scalar", i));
    }
  }
  // write event buffer to file
  EXPECT_TRUE(logger.flush("autosave/scalar"));
}

TEST_F(TensorBoardLoggerTest, FlushAll) {
  noesis::log::TensorBoardLogger logger("test", "flushing", "logger", "test", true);
  logger.setAutosaveEnabled(false);
  // add logging signal
  logger.addLoggingSignal("scalar/x", 1);
  logger.addLoggingSignal("scalar/x^2", 1);
  logger.addLoggingSignal("scalar/x^3", 1);
  logger.startup();
  NINFO(logger);
  size_t N = 100;
  for (size_t i = 0; i<N; i++) {
    logger.appendScalar("scalar/x", i);
    logger.appendScalar("scalar/x^2", i*i);
    logger.appendScalar("scalar/x^3", i*i*i);
    EXPECT_TRUE(logger.flush());
  }
}

/*
 * Test Fixture
 */

class TensorBoardLauncherTest : public ::testing::Test
{
protected:
  // Declare fixture aliases
  TensorBoardLauncherTest() = default;
  ~TensorBoardLauncherTest() = default;
};

/*
 * Tests
 */

TEST_F(TensorBoardLauncherTest, DISABLED_Creation) {
  // TODO
}

TEST_F(TensorBoardLauncherTest, DISABLED_Launch) {
  // TODO
}

} // namespace tests
} // namespace noesis

/* EOF */
