/*!
 * @author    Joonho Lee
 * @email     junja94@gmail.com
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_TEST_FRAMEWORK_CORE_TEST_TENSOR_HPP_
#define NOESIS_TEST_FRAMEWORK_CORE_TEST_TENSOR_HPP_

// Noesis
#include <noesis/framework/core/Tensor.hpp>

namespace noesis {
namespace tests {

template<typename ScalarType_>
static bool isSameData(const Tensor<ScalarType_>& T1, const Tensor<ScalarType_>& T2) {
  if (T1.size() != T2.size()) {
    DNINFO("size mismatch!");
    return false;
  }
  if ((T1.asFlat() - T2.asFlat()).norm() != 0.0f) {
    DNINFO("data mismatch!");
    return false;
  }
  return true;
}

template<typename ScalarType_>
static bool isSameData(const noesis::TensorMap<ScalarType_>& T1, const noesis::TensorMap<ScalarType_>& T2) {
  if (T1.size() != T2.size()) {
    DNINFO("size mismatch!");
    return false;
  }
  if ((T1.asFlat() - T2.asFlat()).norm() != 0.0f) {
    DNINFO("data mismatch!");
    return false;
  }
  return true;
}

template<typename ScalarType_>
static bool isSameData(const Eigen::Matrix<ScalarType_, -1, -1> M1, const Eigen::Matrix<ScalarType_, -1, -1> M2) {
  if (M1.size() != M2.size()) {
    DNINFO("size mismatch!");
    return false;
  }
  for (int i = 0; i < M1.size(); i++) {
    if (M1.data()[i] != M2.data()[i]) {
      DNINFO("data mismatch!");
      return false;
    }
  }
  return true;
}

template<typename ScalarType_, int NDim>
static bool isSameData(const Eigen::Tensor<ScalarType_, NDim> T1, const Eigen::Tensor<ScalarType_, NDim> T2) {
  if (T1.dimensions().size() != T2.dimensions().size()) {
    DNINFO("size mismatch!");
    return false;
  }
  for (size_t i = 0; i < T1.size(); i++) {
    if (T1.data()[i] - T2.data()[i] != 0) {
      DNINFO("data mismatch!");
      return false;
    }
  }
  return true;
}

template<typename ScalarType_>
static bool isEqual(const Tensor<ScalarType_>& T1, const Tensor<ScalarType_>& T2) {
  for (size_t i = 0; i < T1.dimensions().size(); i++) {
    if (T1.dimensions(i) != T2.dimensions(i)) {
      DNINFO("dimension mismatch!");
      return false;
    }
  }
  return isSameData(T1, T2);
}

template<typename ScalarType_>
static bool isEqual(const noesis::TensorMap<ScalarType_>& T1, const noesis::TensorMap<ScalarType_>& T2) {
  for (size_t i = 0; i < T1.dimensions().size(); i++) {
    if (T1.dimensions(i) != T2.dimensions(i)) {
      DNINFO("dimension mismatch!");
      return false;
    }
  }
  return isSameData(T1, T2);
}

template<typename ScalarType_>
static bool isEqual(const Eigen::Matrix<ScalarType_, -1, -1> M1, const Eigen::Matrix<ScalarType_, -1, -1> M2) {
  if (M1.cols() != M2.cols() || M1.rows() != M2.rows()) {
    DNINFO("dimension mismatch!");
    return false;
  }
  return isSameData(M1, M2);
}

template<typename ScalarType_, int NDim>
static bool isEqual(const Eigen::Tensor<ScalarType_, NDim> T1, const Eigen::Tensor<ScalarType_, NDim> T2) {
  for (size_t i = 0; i < T1.dimensions().size(); i++) {
    if (T1.dimensions(i) != T2.dimensions(i)) {
      DNINFO("dimension mismatch!");
      return false;
    }
  }
  return isSameData(T1, T2);
}

template<typename ScalarType_>
static bool testCastToEigenMatrix(Eigen::Matrix<ScalarType_, -1, -1> eig_matrix, Tensor<ScalarType_>& ns_tensor) {
  DNINFO("Casting to Eigen::Matrix:\n" << eig_matrix)
  return isEqual(ns_tensor.asEigenMatrix(), eig_matrix);
}

template<typename ScalarType_>
static bool testCastToTensorflowTensor(std::pair<std::string, tensorflow::Tensor> tf_tensor, Tensor<ScalarType_>& ns_tensor) {
  DNINFO("Casting to tensorflow::Tensor:\n" << (tf_tensor.second.DebugString()));
  bool checkData, checkShape, checkName;
  checkData = true;
  for (size_t i = 0; i < ns_tensor.size(); i++) {
    if (ns_tensor[i] != tf_tensor.second.template flat<ScalarType_>().data()[i]) {
      checkData = false;
    }
  }
  checkShape = (tf_tensor.second.shape() == ns_tensor.get().shape());
  checkName = (ns_tensor.getName() == tf_tensor.first);
  return (checkData && checkShape && checkName);
}

} // namespace tests
} // namespace noesis

#endif // NOESIS_TEST_FRAMEWORK_CORE_TEST_TENSOR_HPP_

/* EOF */
