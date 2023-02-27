/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_COMMON_MATH_HPP_
#define NOESIS_GYM_ENVS_RAISIM_COMMON_MATH_HPP_

// C/C++
#include <random>
#include <iostream>

// Noesis
#include "noesis/gym/envs/raisim/common/types.hpp"

namespace noesis {
namespace gym {
namespace math {

/*
 * Scalar Operations
 */

static inline int floor(double x) {
  int i = int(x);
  if (i > x) i--;
  return i;
}

template<typename T>
static inline T safe_acos(T x) {
  if (x < -static_cast<T>(1)) x = -static_cast<T>(1) ;
  else if (x > static_cast<T>(1)) x = static_cast<T>(1) ;
  return std::acos(x);
}

static inline double angle_wrap(double theta) {
  return theta - (2.0 * M_PI) * floor(theta / (2.0 * M_PI));
}

static inline double angle_modulo(double theta) {
  return angle_wrap((theta + M_PI)) - M_PI;
}

static inline double angle_diff(double theta, double phi) {
  return angle_modulo(theta - phi);
}

static inline double modulo(double x, double y) {
  if (y == 0.0) return x;
  double m = x - y * std::floor(x/y);
  // handle boundary cases resulted from floating-point cut off:
  if (y > 0) { // modulo range: [0..y)
    if (m >= y) { return 0; } // mod(-1e-16, 360.0): m = 360.0
    if (m < 0) {
      if (y+m == y) { return 0; } // just in case...
      else { return y+m; } // mod(106.81415022205296 , 2*M_PI): m = -1.421e-14
    }
  } else { // modulo range: (y..0]
    if (m <= y) { return 0; } // mod(1e-16, -360.0): m = -360.0
    if ( m>0) {
      if (y+m == y) { return 0; } // just in case...
      else { return y+m; } // mod(-106.81415022205296, -2*M_PI): m = 1.421e-14
    }
  }
  return m;
}

static inline double wrap_pi(double angle) {
  return modulo(angle + M_PI, 2.0*M_PI) - M_PI;
}

static inline double wrap_two_pi(double angle) {
  return modulo(angle, 2.0*M_PI);
}

static inline AngleAxis unique_angle_axis(const AngleAxis& phi) {
  AngleAxis aa(wrap_pi(phi.angle()), phi.axis()); // first wraps angle into [-pi,pi)
  if(aa.angle() > 0.0) {
    return aa;
  } else if(aa.angle() < 0.0) {
    if(aa.angle() != -M_PI) {
      return AngleAxis(-aa.angle(), -aa.axis());
    } else { // angle == -pi, so axis must be viewed further, because -pi,axis does the same as -pi,-axis
      if(aa.axis()[0] < 0.0) {
        return AngleAxis(-aa.angle(), -aa.axis());
      } else if(aa.axis()[0] > 0) {
        return AngleAxis(-aa.angle() ,aa.axis());
      } else { // v1 == 0
        if(aa.axis()[1] < 0.0) {
          return AngleAxis(-aa.angle(), -aa.axis());
        } else if(aa.axis()[1] > 0) {
          return AngleAxis(-aa.angle(), aa.axis());
        } else { // v2 == 0
          if(aa.axis()[2] < 0.0) { // v3 must be -1 or 1
            return AngleAxis(-aa.angle(), -aa.axis());
          } else {
            return AngleAxis(-aa.angle(), aa.axis());
          }
        }
      }
    }
  } else { // angle == 0
    return AngleAxis(0.0, Vector3::Zero());
  }
}

/*
 * Algebraic Operations
 */

static inline Matrix3 skew(const Vector3& v) {
  Matrix3 S;
  S <<
    0,    -v(2),  v(1),
    v(2),  0,    -v(0),
    -v(1),  v(0),  0;
  return S;
}

static inline Vector3 unskew(const Matrix3& S) {
  return Vector3(S(2, 1), S(0, 2), S(1, 0));
}


/*
 * Rotation Primitives
 */

static inline RotationMatrix rotation_x(double phi) {
  RotationMatrix R;
  R << 1, 0, 0,
       0, std::cos(phi), -std::sin(phi),
       0, std::sin(phi), std::cos(phi);
  return R;
}

static inline RotationMatrix rotation_y(double phi) {
  RotationMatrix R;
  R << std::cos(phi), 0, std::sin(phi),
       0, 1, -0,
       -std::sin(phi), 0, std::cos(phi);
  return R;
}

static inline RotationMatrix rotation_z(double phi) {
  RotationMatrix R;
  R << std::cos(phi), -std::sin(phi), 0,
       std::sin(phi), std::cos(phi), 0,
       0, 0, 1;
  return R;
}

/*
 * From Unit Quaternions
 */

static inline RotationMatrix quaternion_to_rotation_matrix(const Quaternion &q) {
  RotationMatrix R;
  R << q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3), 2 * q(1) * q(2) - 2 * q(0) * q(3), 2 * q(0) * q(2) + 2 * q(1) * q(3),
       2 * q(0) * q(3) + 2 * q(1) * q(2), q(0) * q(0) - q(1) * q(1) + q(2) * q(2) - q(3) * q(3), 2 * q(2) * q(3) - 2 * q(0) * q(1),
       2 * q(1) * q(3) - 2 * q(0) * q(2), 2 * q(0) * q(1) + 2 * q(2) * q(3), q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3);
  return R;
}

static inline EulerRpy quaternion_to_euler_angles(const Quaternion& q) {
  EulerRpy rpy;
  double ysqr = q[2] * q[2];
  // roll (x-axis rotation)
  double t0 = +2.0 * (q[0] * q[1] + q[2] * q[3]);
  double t1 = +1.0 - 2.0 * (q[1] * q[1] + ysqr);
  rpy(0) = std::atan2(t0, t1);
  // pitch (y-axis rotation)
  double t2 = +2.0 * (q[0] * q[2] - q[3] * q[1]);
  t2 = t2 > 1.0 ? 1.0 : t2;
  t2 = t2 < -1.0 ? -1.0 : t2;
  rpy(1) = std::asin(t2);
  // yaw (z-axis rotation)
  double t3 = +2.0 * (q[0] * q[3] + q[1] * q[2]);
  double t4 = +1.0 - 2.0 * (ysqr + q[3] * q[3]);
  rpy(2) = std::atan2(t3, t4);
  return rpy;
}

/*
 * From Rotation Matrices
 */

static inline Quaternion rotation_matrix_to_quaternion(const RotationMatrix& R) {
  Quaternion q(1.0, 0.0, 0.0, 0.0);
  double tr = R.trace();
  if (tr > 0.0) {
    double S = sqrt(tr + 1.0) * 2.0; // S=4*qw
    q(0) = 0.25 * S;
    q(1) = (R(2, 1) - R(1, 2)) / S;
    q(2) = (R(0, 2) - R(2, 0)) / S;
    q(3) = (R(1, 0) - R(0, 1)) / S;
  } else if ((R(0, 0) > R(1, 1)) & (R(0, 0) > R(2, 2))) {
    double S = sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0; // S=4*qx
    q(0) = (R(2, 1) - R(1, 2)) / S;
    q(1) = 0.25 * S;
    q(2) = (R(0, 1) + R(1, 0)) / S;
    q(3) = (R(0, 2) + R(2, 0)) / S;
  } else if (R(1, 1) > R(2, 2)) {
    double S = sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0; // S=4*qy
    q(0) = (R(0, 2) - R(2, 0)) / S;
    q(1) = (R(0, 1) + R(1, 0)) / S;
    q(2) = 0.25 * S;
    q(3) = (R(1, 2) + R(2, 1)) / S;
  } else {
    double S = sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0; // S=4*qz
    q(0) = (R(1, 0) - R(0, 1)) / S;
    q(1) = (R(0, 2) + R(2, 0)) / S;
    q(2) = (R(1, 2) + R(2, 1)) / S;
    q(3) = 0.25 * S;
  }
  return q;
}

static inline EulerRpy rotation_matrix_to_euler_angles(const RotationMatrix& R) {
  Vector3 rpy;
  rpy(0) = std::atan2(R(2,1), R(2,2));
  rpy(1) = std::atan2(-R(2,0), std::sqrt(R(2,1)*R(2,1) + R(2,2)*R(2,2)));
  rpy(2) = std::atan2(R(1,0), R(0,0));
  return rpy;
}

/*
 * From Euler-Angles
 */

static inline RotationMatrix euler_angles_to_rotation_matrix(const EulerRpy&  rpy) {
  using AngleAxis = Eigen::AngleAxis<double>;
  return (AngleAxis(rpy[2], EulerRpy::UnitZ()) * AngleAxis(rpy[1], EulerRpy::UnitY()) * AngleAxis(rpy[0], EulerRpy::UnitX())).matrix();
}

static inline Quaternion euler_angles_to_quaternion(const EulerRpy& rpy) {
  auto R = euler_angles_to_rotation_matrix(rpy);
  return rotation_matrix_to_quaternion(R);
}

/*
 * From Angle-Axis
 */

static inline Quaternion angle_axis_to_quaternion(double angle, const Vector3& axis) {
  Quaternion quat;
  quat << cos(angle / 2.0), axis * sin(angle / 2.0);
  return quat;
}

/*
 * Rotation Operations
 */

//! @brief Implements Rodrigues' formula: R = I + sin(θ)*K (1-cos(θ)*K^2)
static inline RotationMatrix exponential_map(const Vector3& v) {
  const double angle = v.norm();
  const Vector3 axis = v.normalized();
  const Matrix3 K = skew(axis);
  const Matrix3 K2 = K*K;
  return RotationMatrix::Identity() + std::sin(angle) * K + (1.0 - std::cos(angle)) * K2;
}

//! @brief Computes R [+] v == exp(v) * R
static inline RotationMatrix box_plus(const RotationMatrix& R, const Vector3& v) {
  return exponential_map(v) * R;
}

//! @brief Computes R0 [-] R1 == log(R0 * inv(R1))
static inline Vector3 box_minus(const RotationMatrix& R0, const RotationMatrix& R1) {
  const AngleAxis aa = unique_angle_axis(AngleAxis(R0 * R1.transpose()));
  Vector3 v = aa.angle() * aa.axis();
  return v;
}

static inline Vector3 quaternion_distance(const Quaternion& q0, const Quaternion& q1) {
  return q0(0) * q1.tail<3>() - q1(0) * q0.tail<3>() + q0.tail<3>().cross(q1.tail<3>());
}

static inline Vector3 quaternion_distance(const RotationMatrix& R0, const RotationMatrix& R1) {
  const auto q0 = rotation_matrix_to_quaternion(R0);
  const auto q1 = rotation_matrix_to_quaternion(R1);
  return quaternion_distance(q0, q1);
}

/*
 * From vector
 */

static inline RotationMatrix vector_to_rotation_matrix(const Eigen::Vector3d& v) {
  using AngleAxis = Eigen::AngleAxis<double>;
  RotationMatrix R = RotationMatrix::Identity();
  if (v.norm() > 0.0) {
    Eigen::Vector3d e(1.0, 0.0, 0.0);
    Eigen::Vector3d n = v;
    n.normalize();
    Eigen::Vector3d axis = e.cross(n);
    axis.normalize();
    double angle = std::acos(e.dot(n));
    R = AngleAxis(angle, axis).matrix();
  }
  return R;
}

/*
 * Algebraic computations
 */

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
static inline Eigen::Matrix<Scalar_, Cols_, Rows_> matrix_pseudo_inverse(const Eigen::Matrix<Scalar_, Rows_, Cols_>& input) {
  Eigen::Matrix<Scalar_, Cols_, Rows_> output;
  matrix_pseudo_inverse(input, output);
  return output;
}

/*
 * Statistical Operations
 */

template<typename ScalarType_, int Rows_, int Cols_>
static inline ScalarType_ cwise_mean(const Eigen::Matrix<ScalarType_,Rows_,Cols_>& matrix) {
  return matrix.mean();
}

template<typename ScalarType_, int Rows_, int Cols_>
static inline ScalarType_ cwise_variance(const Eigen::Matrix<ScalarType_,Rows_,Cols_>& matrix) {
  ScalarType_ mean = matrix.mean(), variance = 0;
  if (matrix.size() > 1) {
    auto N = static_cast<ScalarType_>(matrix.size());
    auto squares = matrix.array();
    squares = squares - mean;
    squares = squares.square();
    variance = squares.sum()/(N-1);
  }
  return variance;
}

template<typename ScalarType_, int Rows_, int Cols_>
static inline std::pair<ScalarType_,ScalarType_> cwise_moments(Eigen::Matrix<ScalarType_,Rows_,Cols_>& matrix) {
  ScalarType_ mean = matrix.mean(), variance = 0;
  if (matrix.size() > 1) {
    auto N = static_cast<ScalarType_>(matrix.size());
    auto squares = matrix.array();
    squares = squares - mean;
    squares = squares.square();
    variance = squares.sum()/(N-1);
  }
  return std::make_pair(mean,variance);
}

template<typename ScalarType_, int Rows_, int Cols_>
static inline std::pair<ScalarType_,ScalarType_> cwise_normalize(Eigen::Matrix<ScalarType_,Rows_,Cols_>& matrix) {
  constexpr auto eps = std::numeric_limits<ScalarType_>::epsilon();
  ScalarType_ mean = matrix.mean(), stddev = 0;
  if (matrix.size() > 1) {
    auto N = static_cast<ScalarType_>(matrix.size());
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

} // namespace math
} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_COMMON_MATH_HPP_

/* EOF */
