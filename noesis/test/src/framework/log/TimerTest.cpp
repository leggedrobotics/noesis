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

// google test
#include <gtest/gtest.h>

// Noesis
#include <noesis/framework/log/message.hpp>
#include <noesis/framework/log/timer.hpp>

namespace noesis {
namespace tests {

/*
 * Tests
 */

TEST(MultiTimerTest, Creation) {
  noesis::log::MultiTimer timers("timers", "test", true);
  NINFO(timers);
}

TEST(MultiTimerTest, AddTimers) {
  noesis::log::MultiTimer timers("timers", "test", true);
  timers.addTimer("test/timer0");
  timers.addTimer("test/timer1");
  timers.addTimer("test/timer2");
  NINFO(timers);
}

TEST(MultiTimerTest, AddExistingTimer) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  noesis::log::MultiTimer timers("timers", "test", true);
  timers.addTimer("test/timer0");
  timers.addTimer("test/timer1");
  ASSERT_DEATH({timers.addTimer("test/timer0");}, "");
  NINFO(timers);
}

TEST(MultiTimerTest, CheckInitialTimes) {
  noesis::log::MultiTimer timers("timers", "test", true);
  timers.addTimer("test/timer0");
  EXPECT_EQ(0.0, timers.getResetTime("test/timer0"));
  EXPECT_EQ(0.0, timers.getStartTime("test/timer0"));
  EXPECT_EQ(0.0, timers.getElapsedTime("test/timer0"));
  EXPECT_EQ(0.0, timers.getTotalTime("test/timer0"));
  NINFO(timers);
}

TEST(MultiTimerTest, Resetting) {
  noesis::log::MultiTimer timers("timers", "test", true);
  timers.addTimer("test/timer0");
  timers.reset("test/timer0");
  auto now = Time::Now().toSeconds();
  EXPECT_EQ(timers.getResetTime("test/timer0"), timers.getStartTime("test/timer0"));
  EXPECT_NEAR(now, timers.getResetTime("test/timer0"), 1e-3);
  EXPECT_NEAR(now, timers.getStartTime("test/timer0"), 1e-3);
  EXPECT_EQ(0.0, timers.getElapsedTime("test/timer0"));
  EXPECT_EQ(0.0, timers.getTotalTime("test/timer0"));
  NINFO(timers);
  NINFO("Time now (s): " << std::fixed << std::setprecision(9) << now);
}

TEST(MultiTimerTest, ComapareResetAndStartTime) {
  noesis::log::MultiTimer timers("timers", "test", true);
  timers.addTimer("test/timer0");
  timers.reset("test/timer0");
  auto now = Time::Now();
  usleep(10000);
  timers.start("test/timer0");
  auto dt = now.elapsed().toSeconds();
  auto resetToStartTime = timers.getStartTime("test/timer0") - timers.getResetTime("test/timer0");
  EXPECT_NEAR(dt, resetToStartTime, 1e-3);
  NINFO(timers);
  NINFO("Time difference (s): " << std::fixed << std::setprecision(9) << dt);
  NINFO("Reset-to-Start (s): " << std::fixed << std::setprecision(9) << resetToStartTime);
}

TEST(MultiTimerTest, MeasuringElapsedTime) {
  noesis::log::MultiTimer timers("timers", "test", true);
  timers.addTimer("test/timer0");
  timers.reset("test/timer0");
  auto now = Time::Now();
  timers.start("test/timer0");
  usleep(10000);
  timers.measure("test/timer0");
  auto dt = now.elapsed().toSeconds();
  EXPECT_NEAR(dt, timers.getElapsedTime("test/timer0"), 1e-3);
  NINFO(timers);
  NINFO("Time elapsed (s): " << std::fixed << std::setprecision(9) << dt);
}

TEST(MultiTimerTest, Stopping) {
  noesis::log::MultiTimer timers("timers", "test", true);
  Time time;
  timers.addTimer("test/timer0");
  timers.reset("test/timer0");
  auto t_reset = time.reset().toSeconds();
  usleep(10000);
  timers.start("test/timer0");
  auto t_start = time.reset().toSeconds();
  usleep(10000);
  timers.measure("test/timer0");
  auto t_measure = time.reset().toSeconds();
  usleep(10000);
  timers.stop("test/timer0");
  auto t_stop = time.reset().toSeconds();
  EXPECT_NEAR(t_reset, timers.getResetTime("test/timer0"), 1e-3);
  EXPECT_NEAR(t_start, timers.getStartTime("test/timer0"), 1e-3);
  EXPECT_NEAR(t_measure-t_start, timers.getElapsedTime("test/timer0"), 1e-3);
  EXPECT_NEAR(t_stop-t_reset, timers.getTotalTime("test/timer0"), 1e-3);
  NINFO(timers);
  NINFO("Time reset (s): " << std::fixed << std::setprecision(9) << t_reset);
  NINFO("Time started (s): " << std::fixed << std::setprecision(9) << t_start);
  NINFO("Time stopped (s): " << std::fixed << std::setprecision(9) << t_stop);
  NINFO("Time elapsed (s): " << std::fixed << std::setprecision(9) << (t_measure-t_start));
  NINFO("Time total (s): " << std::fixed << std::setprecision(9) << (t_stop-t_reset));
}

TEST(MultiTimerTest, NonExistingTimer) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  noesis::log::MultiTimer timers("timers", "test", true);
  ASSERT_DEATH({timers.reset("test/timer1");}, "");
  ASSERT_DEATH({timers.start("test/timer1");}, "");
  ASSERT_DEATH({timers.measure("test/timer1");}, "");
  ASSERT_DEATH({timers.getResetTime("test/timer1");}, "");
  ASSERT_DEATH({timers.getStartTime("test/timer1");}, "");
  ASSERT_DEATH({timers.getElapsedTime("test/timer1");}, "");
}

} // namespace tests
} // namespace noesis

/* EOF */
