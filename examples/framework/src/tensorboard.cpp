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

// Noesis
#include <noesis/noesis.hpp>
#include <noesis/framework/log/timer.hpp>
#include <noesis/framework/log/tensorboard.hpp>
#include <noesis/framework/utils/png.hpp>

// function to create an encoded PNG string corresponding to given rgb
void rgbToEncodedPNG(const int& r, const int& g, const int& b,
                     const size_t& height, const size_t& width,
                     std::string& encoded_string) {
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> red;
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> green;
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> blue;
  auto ones = Eigen::MatrixXi::Ones(width, height);
  red = (ones * r).cast<unsigned char>();
  green = (ones * g).cast<unsigned char>();
  blue = (ones * b).cast<unsigned char>();
  noesis::utils::png_image_to_string(red, green, blue, encoded_string);
}

int main(int argc, char** argv) {
  
  // Initialize noesis process thread
  noesis::init("noesis_tensorboard_example");
  NINFO("Starting the tensorboard logging and visualization demo ...");
  
  // Setup logging and timing object
  noesis::log::TensorBoardLogger logger("", "example", "logger", "example", true);
  noesis::log::MultiTimer timers("example", "timers", true);
  
  // Add signals to be logged
  logger.addLoggingSignal("example/iterator", 10);
  logger.addLoggingSignal("example/hyperbola", 10);
  logger.addLoggingSignal("example/moving_average", 10);
  logger.addLoggingSignal("example/logo", 1);
  logger.addLoggingSignal("example/eigen_rgb", 1);
  logger.addLoggingSignal("example/eigen_gray", 1);
  logger.addLoggingSignal("example/timer", 10);
  logger.startup();
  
  // Add timer
  timers.addTimer("example/timer");
  
  // Print all the signals and timers added
  NINFO(logger);
  NINFO(timers);
  
  // -------------------------------------------------
  // Scalar Data Example
  // -------------------------------------------------
  for (int i = 0; i < 100; i++) {
    timers.start("example/timer");
    // Add data values to log
    logger.appendScalar("example/iterator", i);
    logger.appendScalar("example/hyperbola", 100.0f / i);
    // add intermediate timer data to log
    timers.measure("example/timer");
    logger.appendScalar("example/timer", static_cast<float>(timers.getElapsedTime("example/timer")));
  }
  
  // -------------------------------------------------
  // Histogram Data Example
  // -------------------------------------------------
  
  // reset timer clock
  timers.reset("example/timer");
  
  std::default_random_engine generator;
  std::normal_distribution<double> default_distribution(0, 1.0);
  
  int N = 400;
  for (int i = 0; i < N; ++i) {
    std::vector<float> values;
    
    // start the timer
    timers.start("example/timer");
    usleep(1000);
    
    // moving average distribution
    double mean = i * 5.0 / N;
    std::normal_distribution<float> distribution(mean, 1.0);
    // sample 1000 random samples from the distribution
    for (int j = 0; j < 1000; ++j) {
      values.push_back(distribution(generator));
    }
    // measure the time taken by the timer
    timers.measure("example/timer");
    
    // Add data values to log
    logger.appendHistogram("example/moving_average", values);
    logger.appendScalar("example/timer", static_cast<float>(timers.getElapsedTime("example/timer")));
  }
  
  // -------------------------------------------------
  // Image Data Examples
  // -------------------------------------------------
  
  // ------------- Reading an image file -------------
  
  std::string filename = noesis::rootpath() + "/docs/images/noesis-logo.png";
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  std::ostringstream oss;
  oss << fin.rdbuf();
  std::string data(oss.str());
  // Add encoded image string to log
  logger.appendImageString("example/logo", data);
  
  // ---------- Using an eigen matrix: RGB -----------
  
  std::string encoded_rgb_string;
  int rgb[3] = {50, 50, 100};
  rgbToEncodedPNG(rgb[0], rgb[1], rgb[2], 150, 300, encoded_rgb_string);
  // Add encoded image string to log
  logger.appendImageString("example/eigen_rgb", encoded_rgb_string);
  
  // ----- Using a float eigen matrix: greyscale ------
  
  Eigen::MatrixXf random_grey_matrix = Eigen::MatrixXf::Random(300, 150);
  // Add matrix image to logger
  logger.appendImageMatrix("example/eigen_gray", random_grey_matrix);
  
  // Manually shut-down the logger
  logger.shutdown();
  
  return 0;
}

/* EOF */
