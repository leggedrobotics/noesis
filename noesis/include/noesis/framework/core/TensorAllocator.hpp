/*!
 * @author    Joonho Lee
 * @email     junja94@gmail.com
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_CORE_TENSOR_ALLOCATOR_HPP_
#define NOESIS_FRAMEWORK_CORE_TENSOR_ALLOCATOR_HPP_

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
#include "noesis/framework/utils/string.hpp"

namespace noesis {

template<typename ScalarType_>
class TensorAllocator
{
public:
  // Alias
  using ScalarType = typename std::remove_const<ScalarType_>::type;
  
  /*!
   * @brief Default constructor.
   */
  TensorAllocator() = default;

  /*!
   * @brief Constructor which sets the pointer to the target tensor.
   * @param tensor The target tensor for which allocation operations will be performed.
   */
  explicit TensorAllocator(tensorflow::Tensor *tensor):
    tensorPtr_(tensor)
  {
  }
  
  /*!
   * @brief Copy constructor for allocators to ensure owning Tensor objects can also copy allocation internals.
   * @param other The source allocator instance whose internal state is to be copied.
   */
  TensorAllocator(const TensorAllocator& other):
    dimensions_(other.dimensions_),
    datumDimensions_(other.datumDimensions_),
    timesteps_(other.timesteps_),
    type_(other.type_),
    size_(other.size_),
    batchSize_(other.batchSize_),
    datumSize_(other.datumSize_),
    batched_(other.batched_)
  {
    tensorPtr_ = other.tensorPtr_();
  }
  
  /*!
   * @brief Move constructor for allocators to ensure owning Tensor objects can also move allocation internals.
   * @param other The source allocator instance whose internal state is to be moved.
   */
  TensorAllocator(const TensorAllocator&& other) noexcept:
    dimensions_(std::move(other.dimensions_)),
    datumDimensions_(std::move(other.datumDimensions_)),
    timesteps_(std::move(other.timesteps_)),
    type_(other.type_),
    size_(other.size_),
    batchSize_(other.batchSize_),
    datumSize_(other.datumSize_),
    batched_(other.batched_)
  {
  }
  
  /*
   * Configurations
   */

  void setTensor(tensorflow::Tensor& tensor) {
    tensorPtr_ = &tensor;
  }
  
  void setAllocation(const std::vector<size_t>& dimensions, const std::vector<size_t>& timesteps, const bool batched) {
    dimensions_ = dimensions;
    timesteps_ = timesteps;
    batched_ = batched;
    configureSizes(dimensions_);
  }

  /*
   * Properties
   */

  const std::vector<size_t>& dimensions() const {
    return dimensions_;
  }
  
  std::vector<size_t>& dimensions() {
    return dimensions_;
  }
  
  const std::vector<size_t>& timesteps() const {
    return timesteps_;
  }
  
  std::vector<size_t>& timesteps() {
    return timesteps_;
  }
  
  const size_t& size() const {
    return size_;
  }
  
  size_t& size() {
    return size_;
  }
  
  const size_t& sizeOfBatch() const {
    return batchSize_;
  }
  
  size_t& sizeOfBatch() {
    return batchSize_;
  }
  
  const size_t& sizeOfDatum() const {
    return datumSize_;
  }
  
  const size_t& sizeOfDatum() {
    return datumSize_;
  }
  
  bool isBatched() const {
    return batched_;
  }
  
  bool empty() const {
    return (size_ == 0);
  }
  
  size_t rank() const {
    return dimensions_.size();
  }
  
  std::vector<size_t> capacities() const {
    std::vector<size_t> result;
    if (rank() > 2  && batched_) {
      result.push_back(timeStepCapacity());
      result.push_back(batchCapacity());
    }
    return result;
  }
  
  size_t timeStepCapacity() const {
    if (tensorPtr_->dims() == 0 || tensorPtr_->dim_size(0) == 0 || !batched_) {
      return 0; // unallocated or unbatched
    } else {
      return dimensions_.end()[-2];
    }
  }
  
  size_t batchCapacity() const {
    if (tensorPtr_->dims() == 0 || tensorPtr_->dim_size(0) == 0 || !batched_) {
      return 0; // unallocated or unbatched
    } else {
      return static_cast<size_t>(tensorPtr_->dim_size(0) / dimensions_.end()[-2]);
    }
  }
  
  const std::vector<size_t>& datumDimensions() const {
    return datumDimensions_;
  }
  
  tensorflow::DataType dtype() const {
    return type_;
  }
  
  /*
   * Allocation operations
   */

  void configureSizes(const std::vector<size_t>& dimensions) {
    datumSize_ = 1;
    batchSize_ = 1;
    size_ = 1;
    if (batched_) {
      std::vector<size_t> dims(dimensions.begin(), dimensions.end()-2);
      datumDimensions_ = dims;
      for (size_t i = 0; i < dimensions.size() - 2; ++i) {
        datumSize_ *= dimensions[i];
      }
      batchSize_ = datumSize_ * dimensions[rank() - 2];
      size_ = batchSize_ * dimensions.back();
    } else {
      datumDimensions_ = dimensions;
      for (const auto& dim: dimensions) {
        size_ *= dim;
      }
      batchSize_ = size_;
      datumSize_ = size_;
    }
  }
  
  void configureTimeSteps(bool preserve_existing, bool batched) {
    if (rank() > 1 && batched) {
      timesteps_.resize(dimensions_.back(), 0);
      if (preserve_existing) {
        for (size_t i = 0; i < dimensions_.back(); i++) {
          timesteps_[i] = std::min(timesteps_[i], timeStepCapacity());
        }
      } else {
        std::fill(timesteps_.begin(), timesteps_.end(), timeStepCapacity());
      }
    } else {
      timesteps_.clear();
    }
  }
  
  void fillTimeSteps() {
    if (batched_) {
      timesteps_.resize(dimensions_.back());
      std::fill(timesteps_.begin(), timesteps_.end(), timeStepCapacity());
    }
  }
  
  void fillBatches() {
    if (batched_) {
      dimensions_.end()[-1] = batchCapacity();
      configureSizes(dimensions_);
    }
  }
  
  
  bool reshape(const std::vector<size_t>& new_shape, bool batched=false) {
    bool keep_batch_info = false;
    size_t old_batch_capacity = batchCapacity();
    tensorflow::TensorShape shape = getShapeFromDimensions(new_shape, batched);
    bool result = tensorPtr_->CopyFrom(*tensorPtr_, shape);
    NFATAL_IF(!result, "Tensor allocation failed due to invalid tensorflow::Tensor::CopyFrom() allocation!");
    if (batched_ && new_shape.end()[-2] == timeStepCapacity() &&
        new_shape.back() == old_batch_capacity &&
        new_shape.size() == dimensions_.size()) {
      keep_batch_info = true;
      for (size_t i=0;i<rank()-2;i++) {
        dimensions_[i] = new_shape[i];
      }
    } else {
      dimensions_ = new_shape;
    }
    batched_ = batched;
    configureSizes(dimensions_);
    return keep_batch_info;
  }
  
  void allocateEmpty() {
    // empty tensor (dims = {0}). size_ = 0
    dimensions_ = {0};
    datumDimensions_ = {0};
    timesteps_.clear();
    size_ = 0;
    batchSize_ = 0;
    datumSize_ = 0;
    batched_ = false;
    *tensorPtr_ = tensorflow::Tensor();
  }
  
  void allocateScalar() {
    // scalar case (dims = {}). size_ = 1;
    dimensions_.clear();
    datumDimensions_.clear();
    timesteps_.clear();
    size_ = 1;
    batchSize_ = 1;
    datumSize_ = 1;
    batched_ = false;
    *tensorPtr_ = tensorflow::Tensor(type_, tensorflow::TensorShape({}));
    fillWithZeros();
  }
  
  void allocateSimple(const std::vector<size_t>& new_dimensions) {
    // Normal cases
    auto new_dims = new_dimensions;
    auto new_size = (size_t) std::accumulate(std::begin(new_dims), std::end(new_dims), 1, std::multiplies<ScalarType>());
    bool preserve = false;
    if (datumSize_ != 0) {
      int dimCheck = compareDimensions(new_dims, dimensions_);
      if (dimCheck == static_cast<int>(rank()) && !batched_) {
        return; //no need to change
      }
      preserve = (new_size == size_);
    }
    batched_ = false;
    if (preserve) {
      if (size_ == static_cast<size_t>(tensorPtr_->NumElements())) {
        reshape(new_dims);
      } else {
        tensorflow::TensorShape shape = getShapeFromDimensions(new_dims, false);
        auto temp = new ScalarType[new_size];// temporary storage
        memcpy(temp, data(), sizeof(ScalarType) * new_size); // deepcopy data to temporary storage
        dimensions_ = new_dims;
        *tensorPtr_ = tensorflow::Tensor(type_, shape);
        configureSizes(dimensions_);
        fillWithZeros();
        memcpy(data(), temp, sizeof(ScalarType) * new_size);
        delete[] temp;
      }
    } else {
      auto dims = new_dims;
      auto shape = getShapeFromDimensions(dims, false);
      dimensions_ = new_dims;
      *tensorPtr_ = tensorflow::Tensor(type_, shape); // allocate new storage
      configureSizes(dimensions_);
      fillWithZeros();
    }
  }
  
  bool allocateBatched(const std::vector<size_t>& new_dimensions, size_t new_batch_capacity = 0, bool conservative = true) {
    auto new_dims = new_dimensions;
    bool preserve = false;
    if (datumSize_ != 0 && conservative) {
      int dimCheck = compareDimensions(new_dims, dimensions_);
      if ( (dimCheck == static_cast<int>(rank())) && (batchCapacity() == new_batch_capacity || new_batch_capacity == 0) && (batched_) ) {
        return true; //no need to change
      }
      preserve = true;
      if ((rank() > 2) && (dimCheck < ((int) rank() - 2))) { // datum shape mismatch
        preserve = false;
      }
      if (dimCheck < 0) { // rank mismatch
        preserve = false;
      }
      if (rank() == 1 && new_dims.size() == 1) { // rank1 -> rank1
        preserve = true;
      }
    }
    batched_ = true;
    if (new_dims.size() == 2) {
      new_dims.insert(new_dims.begin(), 1);
    }
    if (preserve) {
      if (new_batch_capacity == 0) {
        new_batch_capacity = std::max(this->batchCapacity(), new_dims.back());
      }
      conservativeResize(new_dims, new_batch_capacity);
    } else {
      auto dims = new_dims;
      dims.back() = std::max(new_batch_capacity, dims.back());
      auto shape = getShapeFromDimensions(dims, true);
      dimensions_ = new_dims;
      *tensorPtr_ = tensorflow::Tensor(type_, shape); // allocate new storage
      configureSizes(dimensions_);
      fillWithZeros();
    }
    return preserve;
  }
  
  void enableBatching() {
    if (!batched_) {
      allocateBatched(dimensions_, 0, true);
    }
  }
  
  void disableBatching() {
    if (batched_) {
      allocateSimple(dimensions_);
      timesteps_.clear();
    }
  }

  /*
   * Assignment operations
   */
  
  void copy(const TensorAllocator& rhs) {
    dimensions_ = rhs.dimensions_;
    datumDimensions_ = rhs.datumDimensions_;
    timesteps_ = rhs.timesteps_;
    type_ = rhs.type_;
    size_ = rhs.size_;
    batchSize_ = rhs.batchSize_;
    datumSize_ = rhs.datumSize_;
    batched_ = rhs.batched_;
  }
  
  void move(const TensorAllocator& rhs) {
    dimensions_ = std::move(rhs.dimensions_);
    datumDimensions_ = std::move(rhs.datumDimensions_);
    timesteps_ = std::move(rhs.timesteps_);
    type_ = rhs.type_;
    size_ = rhs.size_;
    batchSize_ = rhs.batchSize_;
    datumSize_ = rhs.datumSize_;
    batched_ = rhs.batched_;
  }
  
  /*
   * Helper functions
   */
  
  /*!
   * @brief Used in to retrieve the dimension of the allocation as an Eigen::Dsizes object.
   * @return Dimensions of the tensor in column-major order.
   */
  template<int NumDim_>
  auto getEigenDSizes() const {
    Eigen::DSizes<Eigen::DenseIndex, NumDim_> dSizes;
    for (int d = 0; d < NumDim_; d++) {
      dSizes[d] = dimensions_[d];
    }
    return dSizes;
  }
  
  /*!
   * @brief Helper function which generates a string containing a description of the full state of the internal allocation.
   * @return Allocation description string.
   */
  std::string info() const {
    std::string info;
    info += "\nType: " + std::to_string(static_cast<int>(dtype()));
    info += "\nDimensions: " + utils::vector_to_string(dimensions_, "[]");
    info += "\nCapacities {Time, Batch}: " + utils::vector_to_string(capacities());
    info += "\nTimeSteps: " + utils::vector_to_string(timesteps_);
    info += "\nSizes {Datum, Batch, Total}: " + utils::vector_to_string(std::vector<size_t>{datumSize_, batchSize_, size_});
    return info;
  }

private:

  inline const ScalarType* data() const {
    return tensorPtr_->flat<ScalarType>().data();
  }

  inline ScalarType* data() {
    return tensorPtr_->flat<ScalarType>().data();
  }

  inline int compareDimensions(const std::vector<size_t>& dims0, const std::vector<size_t>& dims1, size_t num = 0, bool check_rank = true) {
    if (check_rank && dims0.size() != dims1.size()) {
      return -1;
    }
    if (num == 0) {
      num = dims0.size();
    }
    for (size_t i = 0; i < num; i++) {
      if (dims0[i] != dims1[i]) {
        return (int) i;
      }
    }
    return (int) num;
  }

  inline tensorflow::TensorShape getShapeFromDimensions(const std::vector<size_t>& dimensions, bool batched) {
    tensorflow::TensorShape shape;
    shape.Clear();
    auto dims = dimensions;
    if (batched && !dims.empty()) {
      auto batch_num = dims.end()[-1];
      auto time_max = dims.end()[-2];
      shape.AddDim(time_max * batch_num);
      dims.pop_back();
      dims.pop_back();
    }
    for (size_t k = dims.size(); k > 0; k--) { // iterate backwards
      shape.AddDim(dims[k - 1]);
    }
    return shape;
  }

  /*!
   * @brief Internal helper which conservatively resizes along the last 2 axis.
   * @note  for rank 1,2 tensor, works the same as Eigen's method
   */
  // TODO: needs to be refactored to using something like a switch statement for clarity
  inline void conservativeResize(const std::vector<size_t>& newDimensions, size_t new_batch_capacity) {
    tensorflow::TensorShape shape;
    size_t new_batch_num = std::min(new_batch_capacity, newDimensions.back());
    size_t prevBatchSize = batchSize_;
    size_t commonBatches = std::min(dimensions_.back(), new_batch_num);
    bool keep_time_capacity = false;
    bool keep_batch_capacity = false;
    size_t new_time_max = 0; // TODO: what is a good initial value for this

    if (newDimensions.size() > 1) {
      new_time_max = newDimensions[newDimensions.size() - 2];
      if (new_time_max == dimensions_.end()[-2]) {
        keep_time_capacity = true;
      }
      if (new_batch_capacity == batchCapacity()) {
        keep_batch_capacity = true;
      }
    } else { // rank 1 Tensor case
      keep_time_capacity = true;
    }

    if (keep_time_capacity && keep_batch_capacity){
      dimensions_.back() = new_batch_num;
      configureSizes(dimensions_);
      return;
    } else {
      auto new_dims = newDimensions;
      new_dims.back() = new_batch_capacity;
      shape = getShapeFromDimensions(new_dims, true);
    }

    if (keep_time_capacity){
      size_t validDataSize = batchSize_ * commonBatches;
      dimensions_.back() = new_batch_num;
      configureSizes(dimensions_);
      auto temp = new ScalarType[validDataSize];// temporary storage
      memcpy(temp, data(), sizeof(ScalarType) * validDataSize); // deepcopy data to temporary storage
      *tensorPtr_ = tensorflow::Tensor(type_, shape);
      fillWithZeros();
      memcpy(data(), temp, sizeof(ScalarType) * validDataSize);
      delete[] temp;
    } else {
      size_t commonTimeSteps = std::min(dimensions_.end()[-2], new_time_max);
      dimensions_.end()[-2] = new_time_max;
      dimensions_.back() = new_batch_num;
      configureSizes(dimensions_);
      using EigenStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
      EigenStride stride = EigenStride(prevBatchSize, 1);
      using EigenMatrix = Eigen::Matrix<ScalarType, -1, -1>;
      EigenMatrix mat(batchSize_, commonBatches); // new batch size X commonBatches
      mat.setZero();
      using EigenMap = Eigen::Map<EigenMatrix, 0, EigenStride>;
      EigenMap map(data(), datumSize_ * commonTimeSteps, commonBatches, stride);
      mat.block(0, 0, datumSize_ * commonTimeSteps, commonBatches) = map;
      *tensorPtr_ = tensorflow::Tensor(type_, shape);
      fillWithZeros();
      memcpy(data(), mat.data(), sizeof(ScalarType) * mat.size());
    }
  }

  inline void fillWithZeros() {
    Eigen::Map<Eigen::Matrix<ScalarType, -1, -1>> map(data(), 1, tensorPtr_->NumElements());
    map.setZero();
  }

  static tensorflow::DataType DataType() {
    if (typeid(ScalarType) == typeid(double)) {
      return tensorflow::DataType::DT_DOUBLE;
    } else if (typeid(ScalarType) == typeid(float)) {
      return tensorflow::DataType::DT_FLOAT;
    } else if (typeid(ScalarType) == typeid(int)) {
      return tensorflow::DataType::DT_INT32;
    }
    return tensorflow::DataType::DT_INVALID;
  }

private:
  //! @brief Shape of tensor in column-major order.
  std::vector<size_t> dimensions_ = {0};
  //! @brief Shape of primitive datum;
  std::vector<size_t> datumDimensions_;
  //! @brief Keeps track of the number of valid data in time axis (second to last dimension).
  std::vector<size_t> timesteps_;
  //! @brief A pointer to the target tensorflow::Tensor instance for which the allocation operations are performed.
  tensorflow::Tensor *tensorPtr_ = nullptr;
  //! @brief Enum describing the fundamental scalar type used by the tensor.
  tensorflow::DataType type_ = DataType();
  //! @brief Total size of valid tensor(dimensions_).
  size_t size_ = 0;
  //! @brief Total number of scalars contained in a single batch: datum-size * max-time-steps.
  size_t batchSize_ = 0;
  //! @brief Total number of scalars contained in a datum primitive.
  size_t datumSize_ = 0;
  //! @brief If batched, batchCapacity >= dimensions_.back()
  bool batched_ = false;
};

} // namespace noesis

#endif // NOESIS_FRAMEWORK_CORE_TENSOR_ALLOCATOR_HPP_

/* EOF */
