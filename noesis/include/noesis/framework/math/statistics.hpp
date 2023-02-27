/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_MATH_STATISTICS_HPP_
#define NOESIS_FRAMEWORK_MATH_STATISTICS_HPP_

#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "noesis/framework/core/Tensor.hpp"

namespace noesis {
namespace math {

/*
 * Matrix Coefficient-Wise Operations
 */

template<typename Scalar_, int Rows_, int Cols_>
static inline Scalar_ cwise_mean(const Eigen::Matrix<Scalar_,Rows_,Cols_>& matrix) {
  return matrix.mean();
}

template<typename Scalar_, int Rows_, int Cols_>
static inline Scalar_ cwise_variance(const Eigen::Matrix<Scalar_,Rows_,Cols_>& matrix) {
  Scalar_ mean = matrix.mean(), variance = 0;
  if (matrix.size() > 1) {
    auto N = static_cast<Scalar_>(matrix.size());
    auto squares = matrix.array();
    squares = squares - mean;
    squares = squares.square();
    variance = squares.sum()/(N-1);
  }
  return variance;
}

template<typename Scalar_, int Rows_, int Cols_>
static inline std::pair<Scalar_,Scalar_> cwise_moments(Eigen::Matrix<Scalar_,Rows_,Cols_>& matrix) {
  Scalar_ mean = matrix.mean(), variance = 0;
  if (matrix.size() > 1) {
    auto N = static_cast<Scalar_>(matrix.size());
    auto squares = matrix.array();
    squares = squares - mean;
    squares = squares.square();
    variance = squares.sum()/(N-1);
  }
  return std::make_pair(mean,variance);
}

template<typename Scalar_, int Rows_, int Cols_>
static inline std::pair<Scalar_,Scalar_> cwise_normalize(Eigen::Matrix<Scalar_,Rows_,Cols_>& matrix) {
  constexpr auto eps = std::numeric_limits<Scalar_>::epsilon();
  Scalar_ mean = matrix.mean(), stddev = 0;
  if (matrix.size() > 1) {
    auto N = static_cast<Scalar_>(matrix.size());
    auto squares = matrix.array();
    squares = squares - mean;
    squares = squares.square();
    stddev = std::sqrt(squares.sum()/(N-1));
    matrix.array() -= mean;
    matrix.array() /= (stddev + eps);
  }
  DNFATAL_IF(!matrix.allFinite(), "Normalized matrix contains NaN/Inf elements!");
  return std::make_pair(mean,stddev);
}

/*
 * Tensor Coefficient-Wise Operations
 */

// NOTE: internal helper functions w/o checks
namespace internal {

template<typename Scalar_>
static inline auto cwise_sum(const Tensor<Scalar_>& tensor) {
  // TODO: get sum of valid data only across all batches - efficiently!!!???
  return tensor.asFlat().sum();
}

} // namespace internal

template<typename Scalar_>
static inline Scalar_ cwise_mean(const Tensor<Scalar_>& tensor) {
  NFATAL_IF(tensor.empty(), "[" << tensor.name() << "]: Cannot compute coefficient-wise mean: Tensor is empty!");
  Scalar_ mean = 0;
  Scalar_ N;
  if (tensor.isBatched()) {
    N = static_cast<Scalar_>(tensor.totalTimeSteps() * tensor.sizeOfDatum());
  } else {
    N = static_cast<Scalar_>(tensor.size());
  }
  mean = internal::cwise_sum(tensor)/N;
  return mean;
}

template<typename Scalar_>
static inline Scalar_ cwise_variance(const Tensor<Scalar_>& tensor) {
  NFATAL_IF(tensor.empty(), "[" << tensor.name() << "]: Cannot compute coefficient-wise variance: Tensor is empty!");
  Scalar_ variance = 0;
  if (tensor.size() > 1) {
    Scalar_ mean = 0;
    Scalar_ N;
    Tensor<Scalar_> squares; squares.copy(tensor);
    if (tensor.isBatched()) {
      N = static_cast<Scalar_>(tensor.totalTimeSteps() * tensor.sizeOfDatum());
    } else {
      N = static_cast<Scalar_>(tensor.size());
    }
    mean = internal::cwise_sum(tensor)/N;
    squares -= mean;
    squares *= squares;
    variance = internal::cwise_sum(squares)/(N-1);
  }
  return variance;
}

template<typename Scalar_>
static inline std::pair<Scalar_,Scalar_> cwise_moments(const Tensor<Scalar_>& tensor) {
  NFATAL_IF(tensor.empty(), "[" << tensor.name() << "]: Cannot compute coefficient-wise moments: Tensor is empty!");
  Scalar_ mean = 0;
  Scalar_ variance = 0;
  Scalar_ N;
  if (tensor.isBatched()) {
    N = static_cast<Scalar_>(tensor.totalTimeSteps() * tensor.sizeOfDatum());
  } else {
    N = static_cast<Scalar_>(tensor.size());
  }
  mean = internal::cwise_sum(tensor)/N;
  if (tensor.size() > 1) {
    Tensor<Scalar_> squares; squares.copy(tensor);
    squares -= mean;
    squares *= squares;
    variance = internal::cwise_sum(squares)/(N-1);
  }
  return std::make_pair(mean, variance);
}

template<typename Scalar_>
static inline std::pair<Scalar_,Scalar_> cwise_normalize(Tensor<Scalar_>& tensor) {
  auto moments = cwise_moments(tensor);
  moments.second = std::sqrt(moments.second);
  if (tensor.size() > 1) {
    tensor -= moments.first;
    tensor /= moments.second;
    DNFATAL_IF(!tensor.allFinite(), "Normalized tensor contains NaN/Inf elements!");
  }
  return moments;
}

template<typename Scalar_>
static inline Scalar_ cwise_explained_variance(const Tensor<Scalar_>& predictions, const Tensor<Scalar_>& observations) {
  constexpr auto eps = std::numeric_limits<Scalar_>::epsilon();
  auto mean_observation = math::cwise_mean(observations);
  auto squares_total = observations - mean_observation;
  squares_total.asEigenMatrix() = squares_total.asEigenMatrix().array().square();
  auto ss_total = squares_total.asEigenMatrix().array().sum();
  auto squares_residual = predictions - observations;
  squares_residual.asEigenMatrix() = squares_residual.asEigenMatrix().array().square();
  auto ss_residual = squares_residual.asEigenMatrix().array().sum();
  return (1.0 - ss_residual / (ss_total + eps));
}

/*
 * Tensor Sequence-Wise Operations
 */

template<typename Scalar_>
static inline Tensor<Scalar_> sequence_mean(Tensor<Scalar_>& tensor) {
  NFATAL_IF(!tensor.isBatched(), "[" << tensor.name() << "]: Cannot compute sequence mean: Tensor is not batched!");
  NFATAL_IF(tensor.empty(), "[" << tensor.name() << "]: Cannot compute sequence mean: Tensor is empty!");
  std::vector<size_t> dims(tensor.dimensions()); dims.end()[-2] = 1;
  Tensor<Scalar_> mean(dims, true);
  auto B = tensor.batches();
  for (size_t b = 0; b < B; ++b) {
    auto T = tensor.timesteps()[b];
    auto map = mean(0,b);
    for (size_t t = 0; t < T; ++t) {
      map += tensor(t,b);
    }
    map /= static_cast<Scalar_>(T);
  }
  return mean;
}

template<typename Scalar_>
static inline Tensor<Scalar_> sequence_variance(Tensor<Scalar_>& tensor) {
  NFATAL_IF(!tensor.isBatched(), "[" << tensor.name() << "]: Cannot compute sequence variance: Tensor is not batched!");
  NFATAL_IF(tensor.empty(), "[" << tensor.name() << "]: Cannot compute sequence variance: Tensor is empty!");
  std::vector<size_t> dims(tensor.dimensions()); dims.end()[-2] = 1;
  Tensor<Scalar_> mean(dims, true);
  Tensor<Scalar_> variance(dims, true);
  Tensor<Scalar_> squares(dims, true);
  auto B = tensor.batches();
  for (size_t b = 0; b < B; ++b) {
    auto T = tensor.timesteps()[b];
    auto meanMap = mean(0,b);
    auto varianceMap = variance(0,b);
    auto squaresMap = squares(0,b);
    for (size_t t = 0; t < T; ++t) {
      meanMap += tensor(t,b);
    }
    meanMap /= static_cast<Scalar_>(T);
    for (size_t t = 0; t < T; ++t) {
      squaresMap = tensor(t,b) - meanMap;
      squaresMap *= squaresMap;
      varianceMap += squaresMap;
    }
    varianceMap /= static_cast<Scalar_>(T-1);
  }
  return variance;
}

template<typename Scalar_>
static inline std::pair<Tensor<Scalar_>,Tensor<Scalar_>> sequence_moments(Tensor<Scalar_>& tensor) {
  NFATAL_IF(!tensor.isBatched(), "[" << tensor.name() << "]: Cannot compute sequence moments: Tensor is not batched!");
  NFATAL_IF(tensor.empty(), "[" << tensor.name() << "]: Cannot compute sequence moments: Tensor is empty!");
  std::vector<size_t> dims(tensor.dimensions()); dims.end()[-2] = 1;
  Tensor<Scalar_> mean(dims, true);
  Tensor<Scalar_> variance(dims, true);
  Tensor<Scalar_> squares(dims, true);
  auto B = tensor.batches();
  for (size_t b = 0; b < B; ++b) {
    auto T = tensor.timesteps()[b];
    auto meanMap = mean(0,b);
    auto varianceMap = variance(0,b);
    auto squaresMap = squares(0,b);
    for (size_t t = 0; t < T; ++t) {
      meanMap += tensor(t,b);
    }
    meanMap /= static_cast<Scalar_>(T);
    for (size_t t = 0; t < T; ++t) {
      squaresMap = tensor(t,b) - meanMap;
      squaresMap *= squaresMap;
      varianceMap += squaresMap;
    }
    varianceMap /= static_cast<Scalar_>(T-1);
  }
  return std::make_pair(std::move(mean), std::move(variance));
}

/*
 * Tensor Batch-Wise Operations
 */

template<typename Scalar_>
static inline Tensor<Scalar_> batch_mean(const Tensor<Scalar_>& tensor) {
  NFATAL_IF(!tensor.isBatched(), "[" << tensor.name() << "]: Cannot compute batch mean: Tensor is not batched!");
  NFATAL_IF(!tensor.isFull(), "[" << tensor.name() << "]: Cannot compute batch mean: Tensor is not full!");
  std::vector<size_t> dims(tensor.dimensions()); dims.back() = 1;
  Tensor<Scalar_> mean(dims, true);
  if (tensor.batches() > 1) {
    auto B = tensor.batches();
    auto map = mean(0);
    for (size_t b = 0; b < B; ++b) {
      map += tensor(b);
    }
    map /= static_cast<Scalar_>(B);
  } else {
    mean = tensor(0);
  }
  return mean;
}

template<typename Scalar_>
static inline Tensor<Scalar_> batch_variance(const Tensor<Scalar_>& tensor) {
  NFATAL_IF(!tensor.isBatched(), "[" << tensor.name() << "]: Cannot compute batch variance: Tensor is not batched!");
  NFATAL_IF(!tensor.isFull(), "[" << tensor.name() << "]: Cannot compute batch variance: Tensor is not full!");
  std::vector<size_t> dims(tensor.dimensions()); dims.back() = 1;
  Tensor<Scalar_> mean(dims, true);
  Tensor<Scalar_> variance(dims, true);
  Tensor<Scalar_> squares(dims, true);
  if (tensor.batches() > 1) {
    auto B = tensor.batches();
    auto meanMap = mean(0);
    auto varianceMap = variance(0);
    auto squaresMap = squares(0);
    for (size_t b = 0; b < B; ++b) {
      meanMap += tensor(b);
    }
    meanMap /= static_cast<Scalar_>(B);
    for (size_t b = 0; b < B; ++b) {
      squaresMap = tensor(b) - meanMap;
      squaresMap *= squaresMap;
      varianceMap += squaresMap;
    }
    varianceMap /= static_cast<Scalar_>(B-1);
  }
  return variance;
}

template<typename Scalar_>
static inline std::pair<Tensor<Scalar_>,Tensor<Scalar_>> batch_moments(const Tensor<Scalar_>& tensor) {
  NFATAL_IF(!tensor.isBatched(), "[" << tensor.name() << "]: Cannot compute batch moments: Tensor is not batched!");
  NFATAL_IF(!tensor.isFull(), "[" << tensor.name() << "]: Cannot compute batch moments: Tensor is not full!");
  std::vector<size_t> dims(tensor.dimensions()); dims.back() = 1;
  Tensor<Scalar_> mean(dims, true);
  Tensor<Scalar_> variance(dims, true);
  Tensor<Scalar_> squares(dims, true);
  if (tensor.batches() > 1) {
    auto B = tensor.batches();
    auto meanMap = mean(0);
    auto varianceMap = variance(0);
    auto squaresMap = squares(0);
    for (size_t b = 0; b < B; ++b) {
      meanMap += tensor(b);
    }
    meanMap /= static_cast<Scalar_>(B);
    for (size_t b = 0; b < B; ++b) {
      squaresMap = tensor(b) - meanMap;
      squaresMap *= squaresMap;
      varianceMap += squaresMap;
    }
    varianceMap /= static_cast<Scalar_>(B-1);
  } else {
    mean(0) = tensor(0);
  }
  return std::make_pair(std::move(mean), std::move(variance));
}

/*
 * Tensor Operations
 */

template<typename Scalar_>
static inline Tensor<Scalar_> mean(const Tensor<Scalar_>& tensor) {
  NFATAL_IF(tensor.empty(), "[" << tensor.name() << "]: Cannot compute mean: Tensor is empty!");
  Tensor<Scalar_> mean(tensor.datumDimensions(), false);
  if (tensor.isBatched()) {
    auto N = tensor.totalTimeSteps();
    auto B = tensor.batches();
    for (size_t b = 0; b < B; ++b) {
      for (size_t t = 0; t < tensor.timesteps()[b]; ++t) {
        mean += tensor(t,b);
      }
    }
    mean /= static_cast<Scalar_>(N);
  } else {
    mean.copy(tensor);
  }
  return mean;
}

template<typename Scalar_>
static inline Tensor<Scalar_> variance(const Tensor<Scalar_>& tensor) {
  NFATAL_IF(tensor.empty(), "[" << tensor.name() << "]: Cannot compute variance: Tensor is empty!");
  Tensor<Scalar_> mean(tensor.datumDimensions(), false);
  Tensor<Scalar_> variance(tensor.datumDimensions(), false);
  Tensor<Scalar_> squares(tensor.datumDimensions(), false);
  if (tensor.isBatched()) {
    auto N = tensor.totalTimeSteps();
    auto B = tensor.batches();
    for (size_t b = 0; b < B; ++b) {
      for (size_t t = 0; t < tensor.timesteps()[b]; ++t) {
        mean += tensor(t,b);
      }
    }
    mean /= static_cast<Scalar_>(N);
    for (size_t b = 0; b < B; ++b) {
      for (size_t t = 0; t < tensor.timesteps()[b]; ++t) {
        squares = tensor(t,b);
        squares -= mean;
        squares *= squares;
        variance += squares;
      }
    }
    variance /= static_cast<Scalar_>(N-1);
  }
  return variance;
}

template<typename Scalar_>
static inline std::pair<Tensor<Scalar_>,Tensor<Scalar_>> moments(const Tensor<Scalar_>& tensor) {
  NFATAL_IF(tensor.empty(), "[" << tensor.name() << "]: Cannot compute moments: Tensor is empty!");
  Tensor<Scalar_> mean(tensor.datumDimensions(), false);
  Tensor<Scalar_> variance(tensor.datumDimensions(), false);
  Tensor<Scalar_> squares(tensor.datumDimensions(), false);
  if (tensor.isBatched()) {
    auto N = tensor.totalTimeSteps();
    auto B = tensor.batches();
    for (size_t b = 0; b < B; ++b) {
      for (size_t t = 0; t < tensor.timesteps()[b]; ++t) {
        mean += tensor(t,b);
      }
    }
    mean /= static_cast<Scalar_>(N);
    for (size_t b = 0; b < B; ++b) {
      for (size_t t = 0; t < tensor.timesteps()[b]; ++t) {
        squares = tensor(t,b);
        squares -= mean;
        squares *= squares;
        variance += squares;
      }
    }
    variance /= static_cast<Scalar_>(N-1);
  } else {
    mean.copy(tensor);
  }
  return std::make_pair(std::move(mean), std::move(variance));
}

} // namespace math
} // namespace noesis

#endif // NOESIS_FRAMEWORK_MATH_STATISTICS_HPP_

/* EOF */
