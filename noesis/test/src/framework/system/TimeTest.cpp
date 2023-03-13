/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// C/C++
#include <stdlib.h>

// google test
#include <gtest/gtest.h>

// Noesis
#include <noesis/framework/log/message.hpp>
#include <noesis/framework/system/time.hpp>

namespace noesis {
namespace tests {

/*
 * Tests
 */

TEST(TimeTest, Creation) {
  Time time;
  NINFO("Time now is: " << time);
  EXPECT_EQ(0.0, time.toSeconds());
}

TEST(TimeTest, NowAndResettingToNow) {
  Time now = Time::Now();
  Time time;
  time.reset();
  NINFO("Time is: " << time);
  NINFO("Now is: " << now);
  EXPECT_NEAR(now.toSeconds(), time.toSeconds(), 1.0e-5);
}

TEST(TimeTest, MeasureElapsed) {
  Time now = Time::Now();
  usleep(10000);
  auto dt = now.elapsed().toSeconds();
  NINFO("dt is: " << dt);
  EXPECT_NEAR(0.01, dt, 1.0e-3);
}

TEST(TimeTest, CovertToMiliSeconds) {
  Time now = Time::Now();
  usleep(10000);
  auto dt_s = now.elapsed().toSeconds();
  auto dt_ms = now.elapsed().toMiliSeconds();
  NINFO("dt in seconds is: " << dt_s);
  NINFO("dt in mili-seconds is: " << dt_ms);
  EXPECT_NEAR(dt_s*1000.0, dt_ms, 1.0e-2);
}

TEST(TimeTest, CovertToMicroSeconds) {
  Time now = Time::Now();
  usleep(10000);
  auto dt_s = now.elapsed().toSeconds();
  auto dt_us = now.elapsed().toMicroSeconds();
  NINFO("dt in seconds is: " << dt_s);
  NINFO("dt in micro-seconds is: " << dt_us);
  EXPECT_NEAR(dt_s*1.0e+6, dt_us, 10.0);
}

TEST(TimeTest, CovertToNanoSeconds) {
  Time now = Time::Now();
  usleep(10000);
  auto dt_s = now.elapsed().toSeconds();
  auto dt_ns = now.elapsed().toNanoSeconds();
  NINFO("dt in seconds is: " << dt_s);
  NINFO("dt in nano-seconds is: " << dt_ns);
  EXPECT_NEAR(dt_s*1.0e+9, dt_ns, 10000.0);
}

TEST(TimeTest, CovertToString) {
  Time now = Time::Now();
  NINFO("Time now is: " << now.toString());
}

} // namespace tests
} // namespace noesis

/* EOF */
