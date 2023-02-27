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
#ifndef NOESIS_FRAMEWORK_CORE_TENSOR_MAP_HPP_
#define NOESIS_FRAMEWORK_CORE_TENSOR_MAP_HPP_

// C/C++
#include <algorithm>
#include <string>
#include <vector>

// Eigen3
#include <Eigen/Core>

// TensorFlow
#include <tensorflow/core/framework/tensor.h>

// Noesis
#include "noesis/framework/log/message.hpp"
#include "noesis/framework/math/random.hpp"
#include "noesis/framework/utils/string.hpp"

namespace noesis {

// Forward declaration of the TensorMap type
template<typename ScalarType_>
class Tensor;

/*!
 * @brief Enables access to subsets or slices of noesis::Tensor objects by providing mappings to
 *  the underlying storage via Eigen:Map and Eigen::TensorMap types.
 *
 * @note If batchSize == 0: Un-batched
 *       If batchSize == 1: Single Batch. dimensions_.back() = time axi
 *       If batchSize >= 2: Multiple Batch. dimensions_.back() = batch axis
 *
 * @tparam ScalarType_ Fundamental arithmetic type used for the scalar values in the Tensor object.
 * @tparam IndexType_ The datum index type.
 */
template<typename ScalarType_, typename IndexType_ = size_t>
struct TensorMap
{
  // Aliases
  using IndexType = IndexType_;
  using ScalarType = ScalarType_;
  using ScalarBaseType = typename std::remove_const<ScalarType>::type;
  using IndexBaseType = typename std::remove_const<IndexType>::type;
  using TensorType = Tensor<ScalarBaseType>;
  using TensorMapType = TensorMap<ScalarBaseType, IndexBaseType>;
  using TensorMapConstType = TensorMap<const ScalarBaseType, const IndexBaseType>;
  using MatrixMap = Eigen::Map<Eigen::Matrix<ScalarBaseType, -1, -1>>;
  using ConstMatrixMap = const Eigen::Map<const Eigen::Matrix<ScalarBaseType, -1, -1>>;
  using VectorMap = Eigen::Map<Eigen::Matrix<ScalarBaseType, -1, 1>>;
  using ConstVectorMap = const Eigen::Map<const Eigen::Matrix<ScalarBaseType, -1, 1>>;
  
  /*!
   * @brief Full explicit constructor for creating a TensorMap instance to map Tensor element subsets.
   * @param data
   * @param dimensions
   * @param name
   * @param timesteps
   * @param batches
   */
  explicit TensorMap(ScalarType *data, const std::vector<size_t>& dimensions, const std::string* name, IndexType* timesteps, size_t batches):
    dimensions_(dimensions),
    name_(name),
    timesteps_(timesteps),
    data_(data),
    batches_(batches)
  {
    if (dimensions.empty()) {
      size_ = 1;
    } else {
      size_ = std::accumulate(std::begin(dimensions), std::end(dimensions), 1u, std::multiplies<ScalarType>());
    }
  }

  /*!
   * @brief Reduced constructor for creating non-batched mappings to Tensor objects.
   * @param data
   * @param dimensions
   */
  explicit TensorMap(ScalarType *data, const std::vector<size_t>& dimensions, const std::string* name):
    TensorMap(data, dimensions, name, nullptr, 0)
  {
  }
  
  /*!
   * @brief Reduced constructor for creating minimal mappings to Tensor objects.
   * @param data
   * @param dimensions
   */
  explicit TensorMap(ScalarType *data, const std::vector<size_t>& dimensions):
    TensorMap(data, dimensions, nullptr, nullptr, 0)
  {
  }
  
  /*!
   * @brief Default destructor.
   */
  ~TensorMap() = default;
  
  /*
   * Properties
   */

  const std::vector<size_t>& dimensions() const {
    return dimensions_;
  }
  
  std::string name() const {
    std::string name = (name_) ? *name_ : "unknown";
    return name;
  }
  
  const IndexType* timesteps() const {
    return timesteps_;
  }
  
  size_t batches() const {
    return batches_;
  }
  
  size_t size() const {
    return size_;
  }
  
  const ScalarType* data() const {
    return data_;
  }
  
  ScalarType* data() {
    return data_;
  }
  
  size_t totalTimeSteps() const {
    size_t sum = 0;
    for (size_t i = 0; i < batches_; i++) {
      sum += *(timesteps_ + i);
    }
    return sum;
  }
  
  bool isBatched() const {
    return (bool)batches_;
  }
  
  /*
   * Access operations
   */

  const ScalarType& operator[](size_t i) const {
    return data()[i];
  }
  
  ScalarType& operator[](size_t i) {
    return data()[i];
  }
  
  auto asFlat() const {
    return ConstVectorMap(data(), size_, 1);
  }
  
  auto asFlat() {
    return VectorMap(const_cast<ScalarBaseType*>(data()), size_, 1);
  }
  
  auto asEigenMatrix() const {
    size_t dim = (dimensions_.empty()) ? 1 : dimensions_[0];
    return ConstMatrixMap(data(), dim, size_ / dim);
  }
  
  auto asEigenMatrix() {
    size_t dim = (dimensions_.empty()) ? 1 : dimensions_[0];
    return MatrixMap(const_cast<ScalarBaseType*>(data()), dim, size_ / dim);
  }
  
  template<int NumDims>
  auto asEigenTensor() {
    Eigen::DSizes<Eigen::DenseIndex, NumDims> DSizes;
    for (int d = 0; d < NumDims; d++) { DSizes[d] = dimensions_[d]; }
    return Eigen::TensorMap<Eigen::Tensor<ScalarBaseType, NumDims>, Eigen::Aligned>(data(), DSizes);
  }
  
  /*
   * Occupancy operations
   */

  void clearTimeSteps() {
    for (size_t i=0; i < batches_; i++) {
      timesteps_[i] = 0;
    }
  }
  
  void fillTimeSteps() {
    if (batches_ == 1) {
      timesteps_[0] = dimensions_.back();
    } else if (batches_ > 1) {
      for (size_t i=0; i < batches_; i++) {
        timesteps_[i] = dimensions_[dimensions_.size()-2];
      }
    }
  }
  
  /*
   * Value setting operations
   */
  
  void setConstant(const ScalarBaseType constant) {
    asFlat().setConstant(constant);
    fillTimeSteps();
  }
  
  void setZero() {
    asFlat().setZero();
    fillTimeSteps();
  }
  
  void setRandomUnitUniform(math::RandomNumberGenerator<ScalarType>& generator) {
    for (size_t k = 0; k < size_; k++) {
      data()[k] = generator.sampleUnitUniform();
    }
    fillTimeSteps();
  }
  
  void setRandomUnitUniform(math::RandomNumberGenerator<ScalarType>* generator) {
    for (size_t k = 0; k < size_; k++) {
      data()[k] = generator->sampleUnitUniform();
    }
    fillTimeSteps();
  }
  
  void setRandomUniform(math::RandomNumberGenerator<ScalarType>& generator, const TensorType& min, const TensorType& max) {
    NFATAL_IF(min.dimensions() != max.dimensions(), "[" << name() << "]: Min-Max dimensions are not compatible!");
    NFATAL_IF(min.dimensions() != dimensions_, "[" << name() << "]: Min-Max dimensions are not compatible with current!");
    for (size_t k = 0; k < size_; k++) {
      data()[k] = generator.sampleUniform(min[k], max[k]);
    }
    fillTimeSteps();
  }
  
  void setRandomUniform(math::RandomNumberGenerator<ScalarType>* generator, const TensorType& min, const TensorType& max) {
    NFATAL_IF(min.dimensions() != max.dimensions(), "[" << name() << "]: Min-Max dimensions are not compatible!");
    NFATAL_IF(min.dimensions() != dimensions_, "[" << name() << "]: Min-Max dimensions are not compatible with current!");
    for (size_t k = 0; k < size_; k++) {
      data()[k] = generator->sampleUniform(min[k], max[k]);
    }
    fillTimeSteps();
  }
  
  void setRandomStandardNormal(math::RandomNumberGenerator<ScalarType>& generator) {
    for (size_t k = 0; k < size_; k++) {
      data()[k] = generator.sampleStandardNormal();
    }
    fillTimeSteps();
  }
  
  void setRandomStandardNormal(math::RandomNumberGenerator<ScalarType>* generator) {
    for (size_t k = 0; k < size_; k++) {
      data()[k] = generator->sampleStandardNormal();
    }
    fillTimeSteps();
  }
  
  void setRandomNormal(math::RandomNumberGenerator<ScalarType>& generator, const TensorType& min, const TensorType& max) {
    NFATAL_IF(min.dimensions() != max.dimensions(), "[" << name() << "]: Min-Max dimensions are not compatible!");
    NFATAL_IF(min.dimensions() != dimensions_, "[" << name() << "]: Min-Max dimensions are not compatible with current!");
    for (size_t k = 0; k < size_; k++) {
      auto delta = max[k] - min[k];
      auto center = 0.5 * delta + min[k];
      data()[k] = (0.166666667 * delta) * generator.sampleStandardNormal() + center; // NOTE: 3*sigma ~= delta/2
    }
    fillTimeSteps();
  }
  
  void setRandomNormal(math::RandomNumberGenerator<ScalarType>* generator, const TensorType& min, const TensorType& max) {
    NFATAL_IF(min.dimensions() != max.dimensions(), "[" << name() << "]: Min-Max dimensions are not compatible!");
    NFATAL_IF(min.dimensions() != dimensions_, "[" << name() << "]: Min-Max dimensions are not compatible with current!");
    for (size_t k = 0; k < size_; k++) {
      auto delta = max[k] - min[k];
      auto center = 0.5 * delta + min[k];
      data()[k] = (0.166666667 * delta) * generator->sampleStandardNormal() + center; // NOTE: 3*sigma ~= delta/2
    }
    fillTimeSteps();
  }
  
  /*
   * Assignment operations
   */
  
  TensorMap& operator=(const TensorMapConstType& rhs) {
    NFATAL_IF(rhs.dimensions() != dimensions_, "[" << name() << "]: `rhs` dimensions are not compatible with current!");
    if (batches_ == rhs.batches()){
      for (size_t i = 0; i < batches_; i++) {
        *(timesteps_ + i) = rhs.timesteps()[i];
      }
    } else {
      fillTimeSteps();
    }
    memcpy(data_, rhs.data(), sizeof(ScalarBaseType) * size_);
    return *this;
  }
  
  TensorMap& operator=(const TensorMapType& rhs) {
    NFATAL_IF(rhs.dimensions() != dimensions_, "[" << name() << "]: `rhs` dimensions are not compatible with current!");
    if (batches_ == rhs.batches()){
      for (size_t i = 0; i < batches_; i++) {
        *(timesteps_ + i) = rhs.timesteps()[i];
      }
    } else {
      fillTimeSteps();
    }
    memcpy(data_, rhs.data(), sizeof(ScalarType) * size_);
    return *this;
  }
  
  TensorMap& operator=(const TensorType& rhs) {
    NFATAL_IF(rhs.dimensions() != dimensions_, "[" << name() << "]: `rhs` dimensions are not compatible with current!");
    if ( rhs.isBatched() && (batches_ == rhs.batches()) ) {
      for (size_t i = 0; i < dimensions_.back(); i++) {
        *(timesteps_ + i) = rhs.timesteps()[i];
      }
    } else {
      fillTimeSteps();
    }
    memcpy(data_, rhs.data(), sizeof(ScalarBaseType) * size_);
    return *this;
  }
  
  TensorMap& operator=(const tensorflow::Tensor& rhs) {
    int j = 0;
    if (!dimensions_.empty()) {
      NFATAL_IF(rhs.dims() != dimensions_.size(), "[" << name() << "]: `rhs` dimensions are not compatible with current!");
      for (int i = (int) dimensions_.size() - 1; i > -1; i--) {
        NFATAL_IF(rhs.dim_size(i) != dimensions_[j++], "[" << name() << "]: `rhs` dimensions are not compatible with current!");
      }
    } else {
      NFATAL_IF(rhs.dims() != 0, "[" << name() << "]: `rhs` should be scalar");
    }
    fillTimeSteps();
    std::memcpy(data_, rhs.flat<ScalarBaseType>().data(), sizeof(ScalarBaseType) * size_);
    return *this;
  }
  
  /*
   * Arithmetic operations
   */

  template<typename DerivedScalarType_, typename DerivedIndexType_>
  TensorMap& operator+=(const TensorMap<DerivedScalarType_,DerivedIndexType_>& rhs) {
    NFATAL_IF(rhs.dimensions() != dimensions_, "[" << name() << "]: `rhs` dimensions are not compatible with current!");
    asEigenMatrix() += rhs.asEigenMatrix();
    return *this;
  }
  
  template<typename DerivedScalarType_, typename DerivedIndexType_>
  TensorMap& operator-=(const TensorMap<DerivedScalarType_,DerivedIndexType_>& rhs) {
    NFATAL_IF(rhs.dimensions() != dimensions_, "[" << name() << "]: `rhs` dimensions are not compatible with current!");
    asEigenMatrix() -= rhs.asEigenMatrix();
    return *this;
  }
  
  template<typename DerivedScalarType_, typename DerivedIndexType_>
  TensorMap& operator*=(const TensorMap<DerivedScalarType_,DerivedIndexType_>& rhs) {
    NFATAL_IF(rhs.dimensions() != dimensions_, "[" << name() << "]: `rhs` dimensions are not compatible with current!");
    asEigenMatrix() = asEigenMatrix().cwiseProduct(rhs.asEigenMatrix());
    return *this;
  }
  
  template<typename DerivedScalarType_, typename DerivedIndexType_>
  TensorMap& operator/=(const TensorMap<DerivedScalarType_,DerivedIndexType_>& rhs) {
    NFATAL_IF(rhs.dimensions() != dimensions_, "[" << name() << "]: `rhs` dimensions are not compatible with current!");
    asEigenMatrix() = asEigenMatrix().cwiseQuotient(rhs.asEigenMatrix());
    return *this;
  }
  
  TensorMap& operator+=(const TensorType& rhs) {
    NFATAL_IF(rhs.dimensions() != dimensions_, "[" << name() << "]: `rhs` dimensions are not compatible with current!");
    asEigenMatrix() += rhs.asEigenMatrix();
    return *this;
  }
  
  TensorMap& operator-=(const TensorType& rhs) {
    NFATAL_IF(rhs.dimensions() != dimensions_, "[" << name() << "]: `rhs` dimensions are not compatible with current!");
    asEigenMatrix() -= rhs.asEigenMatrix();
    return *this;
  }
  
  TensorMap& operator*=(const TensorType& rhs) {
    NFATAL_IF(rhs.dimensions() != dimensions_, "[" << name() << "]: `rhs` dimensions are not compatible with current!");
    asEigenMatrix() = asEigenMatrix().cwiseProduct(rhs.asEigenMatrix());
    return *this;
  }
  
  TensorMap& operator/=(const TensorType& rhs) {
    NFATAL_IF(rhs.dimensions() != dimensions_, "[" << name() << "]: `rhs` dimensions are not compatible with current!");
    asEigenMatrix() = asEigenMatrix().cwiseQuotient(rhs.asEigenMatrix());
    return *this;
  }
  
  TensorMap& operator+=(ScalarType rhs) {
    asEigenMatrix().array() += rhs;
    return *this;
  }
  
  TensorMap& operator-=(ScalarType rhs) {
    asEigenMatrix().array() -= rhs;
    return *this;
  }
  
  TensorMap& operator*=(ScalarType rhs) {
    asEigenMatrix().array() *= rhs;
    return *this;
  }
  
  TensorMap& operator/=(ScalarType rhs) {
    auto eps = std::numeric_limits<ScalarType>::epsilon();
    asEigenMatrix().array() /= (rhs + eps);
    return *this;
  }
  
  /*
   *  Helper operations
   */
  
  std::string info() const {
    std::string info = "\n<TensorMap>";
    info += "\nName: '" + name() + "'";
    info += "\nDimensions: " + utils::vector_to_string(dimensions_, "[]");
    info += "\nSize : " + std::to_string(size_);
    return info;
  }
  
  friend std::ostream& operator<<(std::ostream& os, const TensorMap& map) {
    os << map.info() << "\nData:\n" << map.asEigenMatrix() << "\n";
    return os;
  }

private:
  std::vector<size_t> dimensions_;
  const std::string* name_ = nullptr;
  IndexType* timesteps_ = nullptr;
  ScalarType *data_ = nullptr;
  size_t batches_ = 0;
  size_t size_ = 0;
};

} // namespace noesis

#endif // NOESIS_FRAMEWORK_CORE_TENSOR_MAP_HPP_

/* EOF */
