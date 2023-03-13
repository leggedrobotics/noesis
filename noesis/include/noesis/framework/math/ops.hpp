/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_MATH_OPS_HPP_
#define NOESIS_RL_MATH_OPS_HPP_

// Eigen
#include <Eigen/Geometry>

// Noesis
#include "noesis/framework/core/Tensor.hpp"
#include "noesis/framework/core/TensorTuple.hpp"

namespace noesis {
namespace math {

/*
 * Scalar Operations
 */

template<typename Scalar_>
static inline Scalar_ floor(Scalar_ x) {
  int y = int(x);
  if (y > x) y--;
  return static_cast<Scalar_>(y);
}

template<typename Scalar_>
static inline Scalar_ lgsk(Scalar_ x) {
  return static_cast<Scalar_>(4) / (std::exp(x) + std::exp(-x) + static_cast<Scalar_>(2));
}

template<typename Scalar_>
static inline Scalar_ sech(Scalar_ x) {
  return static_cast<Scalar_>(2) / (std::exp(x) + std::exp(-x));
}

template<typename Scalar_>
static inline Scalar_ clip(Scalar_ input, const Scalar_& minimum, const Scalar_& maximum) {
  input = (input > minimum) * input + (input <= minimum) * minimum;
  return (input < maximum) * input + (input >= maximum) * maximum;
}

template<typename Scalar_>
static inline Scalar_ wrap(Scalar_ input, const Scalar_& minimum, const Scalar_& maximum) {
  Scalar_ diff = maximum - minimum;
  while (input > maximum) { input = input - diff; }
  while (input < minimum) { input = input + diff; }
  return input;
}

template<typename Scalar_>
static inline Scalar_ safe_acos(Scalar_ x) {
  if (x < -static_cast<Scalar_>(1)) x = -static_cast<Scalar_>(1) ;
  else if (x > static_cast<Scalar_>(1)) x = static_cast<Scalar_>(1) ;
  return std::acos(x);
}

template<typename Scalar_>
static inline Scalar_ angle_wrap(Scalar_ theta) {
  return theta - (2.0 * M_PI) * floor(theta / (2.0 * M_PI));
}

template<typename Scalar_>
static inline Scalar_ angle_modulo(Scalar_ theta) {
  return angle_wrap((theta + M_PI)) - M_PI;
}

template<typename Scalar_>
static inline Scalar_ angle_diff(Scalar_ theta, Scalar_ phi) {
  return angle_modulo(theta - phi);
}

/*
 * Matrix Operations
 */

template<typename Scalar_, int Rows_, int Cols_>
static inline Eigen::Matrix<Scalar_,Rows_,Cols_>
clip(
    const Eigen::Matrix<Scalar_,Rows_,Cols_>& matrix,
    const Eigen::Matrix<Scalar_,Rows_,Cols_>& min,
    const Eigen::Matrix<Scalar_,Rows_,Cols_>& max) {
  return matrix.cwiseMin(max).cwiseMax(min);
}

template<typename Scalar_, int Rows_, int Cols_>
static inline Eigen::Matrix<Scalar_,Rows_,Cols_>
clip(const Eigen::Matrix<Scalar_,Rows_,Cols_>& matrix, const Eigen::Matrix<Scalar_,Rows_,Cols_>& limits) {
  return matrix.cwiseMin(limits).cwiseMax(-limits);
}

template<typename Scalar_, int Rows_, int Cols_>
static inline void
clip(Eigen::Matrix<Scalar_,Rows_,Cols_>& matrix, Scalar_ min, Scalar_ max) {
  matrix = matrix.cwiseMin(max).cwiseMax(min);
}

template <typename Scalar_, int Rows_, int Cols_>
static inline Eigen::Matrix<Scalar_, Rows_, Cols_>
clip(const Eigen::Matrix<Scalar_, Rows_, Cols_>& input, Scalar_ threshold) {
  return input.array().min(threshold).max(-threshold);
}

template <typename Scalar_, int Rows_, int Cols_>
static inline Eigen::Matrix<Scalar_, Rows_, Cols_>
clip(const Eigen::Matrix<Scalar_, Rows_, Cols_>& matrix, Scalar_ min, Scalar_ max) {
  return matrix.array().min(max).max(min);
}

template<typename Scalar_, int Rows_, int Cols_>
static inline Eigen::Matrix<Scalar_, Rows_, Cols_>
deadzone(const Eigen::Matrix<Scalar_, Rows_, Cols_>& input, Scalar_ threshold) {
  using Matrix = Eigen::Matrix<Scalar_, Rows_, Cols_>;
  const auto zeros = Matrix::Zero(input.rows(), input.cols());
  return (input.array() > threshold || input.array() < -threshold).select(input, zeros);
}

template<typename Scalar_, int Rows_, int Cols_>
static inline Scalar_
select_nearest(Eigen::Matrix<Scalar_,Rows_,Cols_>& values, Scalar_ input) {
  using Matrix = Eigen::Matrix<Scalar_,Rows_,Cols_>;
  Matrix diffs = values - input*Matrix::Ones();
  diffs.array() = diffs.array().square();
  typename Matrix::Index minimizer;
  diffs.minCoeff(&minimizer);
  return values(minimizer);
}

template<typename Scalar_, int Rows_, int Cols_>
static inline typename std::enable_if< Rows_ == Eigen::Dynamic && Cols_ == Eigen::Dynamic, void >::type
matrix_pseudo_inverse(const Eigen::Matrix<Scalar_, Rows_, Cols_>& input, Eigen::Matrix<Scalar_, Cols_, Rows_>& output) {
  using MatrixType = Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>;
  constexpr auto epsilon = std::numeric_limits<Scalar_>::epsilon();
  Eigen::JacobiSVD<MatrixType> svd;
  Scalar_ tolerance;
  if (input.rows() == input.cols()) {
    svd = Eigen::JacobiSVD<MatrixType>(input, Eigen::ComputeFullU | Eigen::ComputeFullV);
    tolerance = epsilon * std::max(input.cols(), input.rows()) * svd.singularValues().array().abs().maxCoeff();
  } else {
    svd = Eigen::JacobiSVD<MatrixType>(input, Eigen::ComputeThinU | Eigen::ComputeThinV);
    tolerance = epsilon * std::max(input.cols(), input.rows()) * svd.singularValues().array().abs()(0);
  }
  auto reciprocals = (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0);
  MatrixType SigmaPlus = reciprocals.matrix().asDiagonal();
  output = svd.matrixV() * SigmaPlus * svd.matrixU().adjoint();
}

template<typename Scalar_, int Rows_, int Cols_>
static inline typename std::enable_if< Rows_ == Cols_ && Rows_ != Eigen::Dynamic , void >::type
matrix_pseudo_inverse(const Eigen::Matrix<Scalar_, Rows_, Cols_>& input, Eigen::Matrix<Scalar_, Cols_, Rows_>& output) {
  using MatrixType = Eigen::Matrix<Scalar_, Rows_, Cols_>;
  constexpr auto epsilon = std::numeric_limits<Scalar_>::epsilon();
  Eigen::JacobiSVD<MatrixType> svd(input, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Scalar_ tolerance = epsilon * std::max(input.cols(), input.rows()) * svd.singularValues().array().abs().maxCoeff();
  auto reciprocals = (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0);
  auto SigmaPlus = reciprocals.matrix().asDiagonal();
  output = svd.matrixV() * SigmaPlus * svd.matrixU().adjoint();
}

template<typename Scalar_, int Rows_, int Cols_>
static inline typename std::enable_if< Rows_ != Cols_ && Rows_ != Eigen::Dynamic && Cols_ != Eigen::Dynamic, void >::type
matrix_pseudo_inverse(const Eigen::Matrix<Scalar_, Rows_, Cols_>& input, Eigen::Matrix<Scalar_, Cols_, Rows_>& output) {
  using BufferType = Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>;
  BufferType other;
  matrix_pseudo_inverse(BufferType(input), other);
  output = other;
}

template<typename Scalar_, int Rows_, int Cols_>
static inline Eigen::Matrix<Scalar_, Cols_, Rows_>
matrix_pseudo_inverse(const Eigen::Matrix<Scalar_, Rows_, Cols_>& input) {
  Eigen::Matrix<Scalar_, Cols_, Rows_> output;
  matrix_pseudo_inverse(input, output);
  return output;
}

/*
 * Tensor Operations
 */

template<typename Scalar_>
static inline void clip(Tensor<Scalar_>& tensor, Scalar_ min, Scalar_ max) {
  auto map = tensor.asEigenMatrix();
  map = map.cwiseMin(max).cwiseMax(min);
}

template<typename Scalar_>
static inline void clip(TensorTuple<Scalar_>& tuple, Scalar_ min, Scalar_ max) {
  for (size_t k = 0; k < tuple.size(); ++k) {
    auto map = tuple[k].asEigenMatrix();
    map = map.cwiseMin(max).cwiseMax(min);
  }
}

} // namespace math
} // namespace noesis

#endif // NOESIS_RL_MATH_OPS_HPP_

/* EOF */
