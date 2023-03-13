/*!
 * @author    Joonho Lee
 * @email     junja94@gmail.com
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    David Hoeller
 * @email     dhoeller@ethz.ch
 * @author    Jemin Hwangbo
 * @email     jhwangno@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_CORE_TENSOR_HPP_
#define NOESIS_FRAMEWORK_CORE_TENSOR_HPP_

// Noesis
#include "noesis/framework/core/TensorAllocator.hpp"
#include "noesis/framework/core/TensorMap.hpp"

namespace noesis {

/*!
 * @class Tensor
 * @brief It provides an interface to tensorflow::Tensor via Eigen::Tensor and Eigen::Matrix/Vector
 *
 * @note: - if isBatched() = true :
 *          0. batched Tensor ([datum shape, time, batch])
 *          1. dimensions.back() equals the number of valid batches
 *          2. batchCapacity >= dimensions.back()
 *          3. timeStepCapacity = dimensions[dimensions.size() - 2]
 *        - else:
 *          0. normal Tensor. ([datum shape])
 *          1. batchCapacity =  dimensions.back() (Meaningless)
 *          2. timesteps() is empty
 *
 * @see https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
 */
template<typename ScalarType_>
class Tensor
{
public:
  // Aliases
  using ScalarType = typename std::remove_const<ScalarType_>::type;
  using TensorMapType = TensorMap<ScalarType, size_t>;
  using ConstTensorMapType = const TensorMap<const ScalarType, const size_t>;
  using MatrixMap = Eigen::Map<Eigen::Matrix<ScalarType, -1, -1>>;
  using ConstMatrixMap = const Eigen::Map<const Eigen::Matrix<ScalarType, -1, -1>>;
  using VectorMap = Eigen::Map<Eigen::Matrix<ScalarType, -1, 1>>;
  using ConstVectorMap = const Eigen::Map<const Eigen::Matrix<ScalarType, -1, 1>>;
  using RandomNumberGeneratorType = math::RandomNumberGenerator<ScalarType>;
  
  /*
   * Constructors
   */

  /*!
   * @brief Default constructor.
   * @note New objects are initialized as empty tensors.
   */
  Tensor():
    allocator_(&namedTensor_.second)
  {
    allocator_.allocateEmpty();
  }

  /*!
   * @brief Construct as empty tensor. Resize has to be called before use.
   * @param name The name to assign to the tensor instance.
   */
  explicit Tensor(const std::string& name):
    Tensor()
  {
    setName(name);
  }

  /*!
   * @brief Construct empty tensor(filled with 0) with specified dimensions.
   * @note: If isBatched() = true, timesteps are initialized to 0. (this Tensor is empty)
   * @param dimensions Vector containing the dimensions of the tensor to be generated.
   * @param name The name assigned to the tensor instance.
   * @param batched Enables batching functionality in the tensor, i.e. additional batch dimension.
   */
  explicit Tensor(const std::vector<size_t>& dimensions, bool batched):
    allocator_(&namedTensor_.second)
  {
    // Performs the initial memory allocation to the shape specified by 'dimensions'
    resize(dimensions, batched);
    // Sets the tensor is filled - NOTE: allocator fill operations check internally if the allocations is batched.
    allocator_.fillBatches();
    allocator_.fillTimeSteps();
  }

  /*!
   * @brief Construct empty tensor(filled with 0) with specified dimensions.
   * @note: If isBatched() = true, timesteps are initialized to 0. (this Tensor is empty)
   * @param dimensions Vector containing the dimensions of the tensor to be generated.
   * @param name The name assigned to the tensor instance.
   * @param batched Enables batching functionality in the tensor, i.e. additional batch dimension.
   */
  explicit Tensor(const std::string& name, const std::vector<size_t>& dimensions, bool batched):
    Tensor(dimensions, batched)
  {
    setName(name);
  }

  /*!
   * @brief Construct with specified dimensions filled with constant value.
   * @note: If isBatched() = true, timesteps are filled up. (timesteps()[i] = timeStepCapacity())
   * @param dimensions Vector containing the dimensions of the tensor to be generated.
   * @param constant A constant value to assign to each element in the Tensor.
   * @param name The name assigned to the tensor instance.
   * @param batched Enables batching functionality in the tensor, i.e. additional batch dimension.
   */
  explicit Tensor(const std::string& name, const std::vector<size_t>& dimensions, ScalarType constant, bool batched):
    Tensor(name, dimensions, batched)
  {
    setConstant(constant);
  }

  /*!
   * @brief Move constructor.
   * @note Retains the same underlying storage as the source tensor.
   * @param other The source tensor instance.
   */
  Tensor(Tensor&& other) noexcept:
    namedTensor_(std::move(other.namedTensor_)),
    allocator_(std::move(other.allocator_))
  {
    allocator_.setTensor(namedTensor_.second);
  }

  /*!
   * @brief Copy constructor.
   * @note Retains the same underlying storage as the source tensor.
   * @param other The source tensor instance.
   */
  Tensor(const Tensor& other)
  {
    *this = other;
    setName(other.name());
  }

  /*!
   * @brief Copy constructor from TensorMap
   * @note This method performs a deep copy of the data pointed to by the TensorMapType instance.
   * @param map The source TensorMapType instance from which to copy the contents.
   * @param batched Enables batching functionality in the tensor, i.e. additional batch dimension.
   */
  explicit Tensor(TensorMapType map, bool batched) :
    Tensor(map.dimensions(), batched)
  {
    std::memcpy(data(), map.data(), sizeof(ScalarType) * map.size());
    if (batched) {
      for (size_t i = 0; i < map.timesteps().size(); i++) {
        allocator_.timesteps()[i] = *map.timesteps()[i];
      }
    }
  }

  /*
   * Configurations
   */

  void setName(const std::string& name) {
    namedTensor_.first = name;
  }

  void setTimeSteps(size_t batch_index, size_t value) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors")
    NFATAL_IF(batch_index >= batches(), "[" << name() << "]: batch_index exceeds total batch")
    allocator_.timesteps()[batch_index] = value;
  }

  /*
   * Properties
   */

  const std::string& name() const {
    return namedTensor_.first;
  }
  
  tensorflow::DataType dtype() const {
    return allocator_.dtype();
  }
  
  const std::vector<size_t>& dimensions() const {
    return allocator_.dimensions();
  }

  std::vector<size_t> capacities() const {
    return allocator_.capacities();
  }

  const std::vector<size_t>& timesteps() const {
    return allocator_.timesteps();
  }

  size_t batches() const {
    size_t result = 0;
    if (isBatched() && !dimensions().empty()) { result = dimensions().back(); }
    return result;
  }

  size_t size() const {
    return allocator_.size();
  }
  
  size_t batch_size() const {
    return batches();
  }
  
  size_t time_size() const {
    return timeStepCapacity();
  }
  
  size_t sizeOfBatch() const {
    return allocator_.sizeOfBatch();
  }

  size_t sizeOfDatum() const {
    return allocator_.sizeOfDatum();
  }

  size_t rank() const {
    return allocator_.dimensions().size();
  }

  bool empty() const {
    return allocator_.empty();
  }

  bool isBatched() const {
    return allocator_.isBatched();
  }

  const std::vector<size_t>& datumDimensions() const {
    return allocator_.datumDimensions();
  }

  size_t batchCapacity() const {
    return allocator_.batchCapacity();
  }

  size_t timeStepCapacity() const {
    return allocator_.timeStepCapacity();
  }

  size_t maxTimeSteps() const {
    size_t max = 0;
    if (isBatched() && !timesteps().empty()) { max = *std::max_element(timesteps().begin(), timesteps().end()); }
    return max;
  }

  size_t minTimeSteps() const {
    size_t min = 0;
    if (isBatched() && !timesteps().empty()) { min = *std::min_element(timesteps().begin(), timesteps().end()); }
    return min;
  }

  size_t totalTimeSteps() const {
    size_t sum = 0;
    if (isBatched() && !timesteps().empty()) { sum = std::accumulate(timesteps().begin(), timesteps().end(), sum); }
    return sum;
  }

  bool isFull() const {
    bool result = true;
    if (isBatched()) { result = minTimeSteps() == timeStepCapacity(); }
    return result;
  }
  
  /*
   * Memory operations
   */

  /*!
   * @brief Sets the occupancy of the tensor to maximum capacity across all dimensions.
   * @note batches are filled all the way until maximum capacity.
   */
  void fill() {
    if (isBatched()) {
      allocator_.fillBatches();
      allocator_.fillTimeSteps();
    } else {
      NWARNING("[" << name() << "]: Filling non-batched tensor. Operation has no effect.");
    }
  }

  /*!
   * @brief Resets all properties and configurations of the tensor instance.
   * @note This effectively renders the tensor as *empty*.
   * @warning This function releases all resources.
   */
  void reset() {
    allocator_.allocateEmpty();
  }

  /*!
   * @brief Clears all batches and resets to respective counter to zero.
   * @note Keeps datum shape, re-sizes to [datum shape, timeStepCapacity, 0] and clears timesteps().
   */
  void clear() {
    if (isBatched()) {
      resizeBatches(0);
    } else {
      NWARNING("[" << name() << "]: Clearing non-batched tensor. Operation has no effect.");
    }
  }

  /*!
   * @brief Reserves memory space by keeping redundant space in the batch axis.
   * @note This method preserves data if datum shape matches
   * @note This method may remove data by reducing the number of valid batches if new_capacities.back() < batches().
   * @warning  This method only works on batched tensor. If a tensor is empty(size() = 0) sets isBatched() = true automatically.
   * @param new_dimensions The dimensions vector specifying the new shape of the tensor memory allocation.
   */
  void reserve(const std::vector<size_t>& new_dimensions) {
    std::vector<size_t> dimensions = new_dimensions;
    size_t new_batch_capacity = dimensions.back();
    dimensions.back() = std::min(batches(), new_batch_capacity);
    bool preserve = allocator_.allocateBatched(dimensions, new_batch_capacity, true);
    allocator_.configureTimeSteps(preserve, true);
  }

  /*!
   * @breif Performs an allocation of memory for a new number time-steps and batches.
   * @note The elements are initialized as empty.
   * @param new_time_max The new capacity in the time dimension.
   * @param new_batch_num The new capacity in the batch dimension.
   */
  void reserve(size_t new_time_cap, size_t new_batch_cap) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!");
    if (batchCapacity() == new_batch_cap && timeStepCapacity() == new_time_cap) { return; }
    std::vector<size_t> new_dimension = dimensions();
    new_dimension[new_dimension.size() - 2] = new_time_cap;
    bool preserve = allocator_.allocateBatched(new_dimension, new_batch_cap, true);
    allocator_.configureTimeSteps(preserve, true);
  }

  /*!
   * @breif Performs an allocation of memory for a new number of batches.
   * @note The batches are initialized empty.
   * @param new_batch_num The new capacity in the batch dimensions.
   */
  void reserve(size_t new_batch_cap) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!")
    if (batchCapacity() == new_batch_cap) { return; }
    bool preserve = allocator_.allocateBatched(dimensions(), new_batch_cap, true);
    allocator_.configureTimeSteps(preserve, true);
  }

  /*!
   *  - If batched = true:
   *        1. If datum shape is the same, conservatively re-sizes along the last two dimensions.
   *        2. If dimensions.back() >= batchCapacity, doubles the batch capacity of the Tensor.
   *        3. data and timesteps of new batches are set to 0.
   *        4. always conservative for a rank 1 tensor
   *  - If batched = false;
   *        1. Keeps the underlying buffer (conservative) if new_dimensions has the same size.
   *
   */
  void resize(const std::vector<size_t>& new_dimensions, bool batched, bool shrink=true) {
    NFATAL_IF(new_dimensions.size() < 2 && batched, "[" << name() << "]: Tensors with batching enabled must be of rank >= 2!");
    if (new_dimensions == dimensions() && batched == isBatched()) { return; }
    bool preserve = false;
    if (new_dimensions.empty()) {
      allocator_.allocateScalar();
    } else if (new_dimensions[0] == 0 && new_dimensions.size() == 1) {
      allocator_.allocateEmpty();
    } else if (batched) {
      size_t new_capacity = (shrink) ? new_dimensions.back() : 0;
      preserve = allocator_.allocateBatched(new_dimensions, new_capacity, true);
    } else {
      allocator_.allocateSimple(new_dimensions);
    }
    allocator_.configureTimeSteps(preserve, batched);
  }

  /*!
   * @brief Reshapes the tensor while retaining the underlying storage.
   * @param new_shape The new dimensions/shape for the tensor object.
   * @param batched Set to true if the shape is to be interpreted as having batching (time, batch) dimensions.
   */
  void reshape(const std::vector<size_t>& new_shape, bool batched) {
    auto new_size = static_cast<int64_t>(std::accumulate(std::begin(new_shape), std::end(new_shape), 1, std::multiplies<size_t>()));
    NFATAL_IF(namedTensor_.second.NumElements() != new_size, "[" << name() << "]: The size of new shape is not compatible!");
    bool preserve = allocator_.reshape(new_shape, batched);
    allocator_.configureTimeSteps(preserve, batched);
  }

  /*!
   *  Conservatively changes batch-size of the tensor
   *  (... , batches()) -> (... , new_batchSize)
   *  @warning The tensor should be batched.
   */
  void resizeBatches(size_t new_batch_size) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!");
    if (batches() == new_batch_size) { return; }
    std::vector<size_t> new_dimension = dimensions();
    new_dimension.back() = new_batch_size;
    bool preserve =  allocator_.allocateBatched(new_dimension, 0, true);
    allocator_.configureTimeSteps(preserve, true);
  }

  /*!
   *  Conservatively changes last two dimensions
   *  @warning The tensor should be batched.
   */
  void resizeBatches(size_t new_time_cap, size_t new_batch_size) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!");
    if (batches() == new_batch_size && timeStepCapacity() == new_time_cap) { return; }
    std::vector<size_t> new_dimension = dimensions();
    new_dimension.back() = new_batch_size;
    new_dimension[new_dimension.size() - 2] = new_time_cap;
    bool preserve = allocator_.allocateBatched(new_dimension, 0, true);
    allocator_.configureTimeSteps(preserve, true);
  }

  void reshapeBatches(size_t new_time_cap, size_t new_batch_size, const std::vector<size_t>& new_timesteps) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!");
    NFATAL_IF((long)(new_time_cap*new_batch_size) != namedTensor_.second.dim_size(0),
      "[" << name() << "]: New shape does not match the total number of batches!");
    NFATAL_IF(new_timesteps.size() != new_batch_size, "[" << name() << "]: `timesteps` must be of size 'new_batch_size'!");
    if ( (batches() == new_batch_size) && (timeStepCapacity() == new_time_cap) && (new_timesteps == timesteps()) ) { return; }
    allocator_.dimensions().end()[-2] = new_time_cap;
    allocator_.dimensions().end()[-1] = new_batch_size;
    allocator_.timesteps() = new_timesteps;
    allocator_.configureSizes(allocator_.dimensions());
  }

  void reshapeBatches(size_t new_time_cap, size_t new_batch_size) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!");
    NFATAL_IF((long)(new_time_cap*new_batch_size) != namedTensor_.second.dim_size(0),
      "[" << name() << "]: New shape does not match the total number of batches!");
    if (batches() == new_batch_size && timeStepCapacity() == new_time_cap) { return; }
    allocator_.dimensions().end()[-2] = new_time_cap;
    allocator_.dimensions().end()[-1] = new_batch_size;
    allocator_.fillTimeSteps();
    allocator_.configureSizes(allocator_.dimensions());
  }

  void clearTimeSteps() {
    if (isBatched()) {
      for (auto& steps: allocator_.timesteps()) { steps = 0; }
    } else {
      NWARNING("[" << name() << "]: clearTimeSteps: Cannot clear time-steps of non-batched tensor. Operations has no effect.");
    }
  }

  void clearTimeSteps(size_t batch) {
    if (isBatched()) {
      NFATAL_IF(batch >= batches(), "[" << name() << "]: Batch index argument exceeds total number of batches");
      allocator_.timesteps()[batch] = 0;
    } else {
      NWARNING("[" << name() << "]: clearTimeSteps: Cannot clear time-steps of non-batched tensor. Operations has no effect.");
    }
  }

  /*
   * Access operations
   */

  const ScalarType* data() const {
    return namedTensor_.second.flat<ScalarType>().data(); // returns const type
  }

  ScalarType* data() {
    return namedTensor_.second.flat<ScalarType>().data();
  }

  const ScalarType& operator[](size_t i) const {
    return data()[i];
  }


  ScalarType& operator[](size_t i) {
    return data()[i];
  }

  /*!
   * @brief Flattens the data and returns 1D Eigen::Map.same
   * @note TensorMaps only valid batches, but possibly invalid batch time-steps.
   */
  auto asFlat() const {
    return ConstVectorMap(data(), size(), 1);
  }

  /*!
   * @brief Flattens the data and returns 1D Eigen::Map.same
   * @note TensorMaps only valid batches, but possibly invalid batch time-steps.
   */
  auto asFlat() {
    return VectorMap(data(), size(), 1);
  }

  /*!
   * @brief Enables user to manipulate data using Eigen::Matrix methods
   * @note TensorMaps only valid batches, but possibly invalid batch time-steps.
   * @return const Eigen::Map of the data in column-major order (opposite to tensorflow::tensor)
   */
  auto asEigenMatrix() const {
    size_t dim = (dimensions().empty()) ? 1 : dimensions()[0];
    return ConstMatrixMap(data(), dim, size() / dim);
  }

  /*!
   * @brief Enables user to manipulate data using Eigen::Matrix methods
   * @note TensorMaps only valid batches, but possibly invalid batch time-steps.
   * @return Eigen::Map of the data in column-major order (opposite to tensorflow::tensor)
   */
  auto asEigenMatrix() {
    size_t dim = (dimensions().empty()) ? 1 : dimensions()[0];
    return MatrixMap(data(), dim, size() / dim);
  }

  /*!
   * @brief Enables user to manipulate data using Eigen::Tensor methods
   * @note TensorMaps only valid batches, but possibly invalid batch time-steps.
   * @return Eigen::TensorMap of the data in column-major order (opposite to tensorflow::tensor)
   */
  template<int NumDim_>
  auto asEigenTensor() const {
    auto dSizes = allocator_.template getEigenDSizes<NumDim_>();
    return Eigen::TensorMap<const Eigen::Tensor<ScalarType, NumDim_>, Eigen::Aligned>(data(), dSizes);
  }

  /*!
   * @brief Enables user to manipulate data using Eigen::Tensor methods
   * @note TensorMaps only valid batches, but possibly invalid batch time-steps.
   * @return Eigen::TensorMap of the data in column-major order (opposite to tensorflow::tensor)
   */
  template<int NumDim_>
  auto asEigenTensor() {
    auto dSizes = allocator_.template getEigenDSizes<NumDim_>();
    Eigen::TensorMap<Eigen::Tensor<ScalarType, NumDim_>, Eigen::Aligned> map(data(), dSizes);
    return map;
  }

  /*
   * Slicing operations
   */

  auto operator()(size_t time_index, size_t batch_index) const {
    DNFATAL_IF(size() == 0, "[" << name() << "]: This tensor is empty!")
    DNFATAL_IF(!isBatched(), "[" << name() << "]: This method only works for batched tensors!")
    DNFATAL_IF(time_index >= timeStepCapacity(), "[" << name() << "]: time index exceeds time capacity!")
    DNFATAL_IF(batch_index >= dimensions().back(), "[" << name() << "]: batch index exceeds the number of batches!")
    std::vector<size_t> dims(dimensions().begin(), dimensions().end()-2);
    return ConstTensorMapType(data() + batch_index * sizeOfBatch() + time_index * sizeOfDatum(), dims, &namedTensor_.first);
  }

  auto operator()(size_t time_index, size_t batch_index) {
    DNFATAL_IF(size() == 0, "[" << name() << "]: This tensor is empty!")
    DNFATAL_IF(!isBatched(), "[" << name() << "]: This method only works for batched tensors!")
    DNFATAL_IF(time_index >= timeStepCapacity(), "[" << name() << "]: time index exceeds time capacity!")
    DNFATAL_IF(batch_index >= dimensions().back(), "[" << name() << "]: batch index exceeds the number of batches!")
    std::vector<size_t> dims(dimensions().begin(), dimensions().end()-2);
    return TensorMapType(data() + batch_index * sizeOfBatch() + time_index * sizeOfDatum(), dims, &namedTensor_.first);
  }

  auto operator()(size_t batch_index) const {
    DNFATAL_IF(size() == 0, "[" << name() << "]: This tensor is empty!")
    DNFATAL_IF(!isBatched(), "[" << name() << "]: This method only works for batched tensors!")
    DNFATAL_IF(batch_index >= dimensions().back(), "[" << name() << "]: batch index exceeds the number of batches!")
    std::vector<size_t> dims(dimensions().begin(), dimensions().end()-1);
    return ConstTensorMapType(data() + batch_index * sizeOfBatch(), dims, &namedTensor_.first, allocator_.timesteps().data() + batch_index, 1);
  }

  auto operator()(size_t batch_index) {
    DNFATAL_IF(size() == 0, "[" << name() << "]: This tensor is empty!")
    DNFATAL_IF(!isBatched(), "[" << name() << "]: This method only works for batched tensors!")
    DNFATAL_IF(batch_index >= dimensions().back(), "[" << name() << "]: batch index exceeds the number of batches!")
    std::vector<size_t> dims(dimensions().begin(), dimensions().end()-1);
    return TensorMapType(data() + batch_index * sizeOfBatch(), dims, &namedTensor_.first, allocator_.timesteps().data() + batch_index, 1);
  }

  auto operator()(size_t batch_index, const std::vector<size_t> shape) {
    DNFATAL_IF(size() == 0, "[" << name() << "]: This tensor is empty!")
    DNFATAL_IF(!isBatched(), "[" << name() << "]: This method only works for batched tensors!")
    DNFATAL_IF(batch_index >= dimensions().back(), "[" << name() << "]: batch index exceeds the number of batches!")
    size_t sizeIn = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_t>());
    DNFATAL_IF(sizeIn != sizeOfBatch(), "[" << name() << "]: 'shape' has incompatible size: " << sizeIn << "vs " << sizeOfBatch());
    return ConstTensorMapType(data() + batch_index * sizeOfBatch(), shape, &namedTensor_.first);
  }

  auto operator()(size_t batch_index, const std::vector<size_t> shape) const {
    DNFATAL_IF(size() == 0, "[" << name() << "]: This tensor is empty!")
    DNFATAL_IF(!isBatched(), "[" << name() << "]: This method only works for batched tensors!")
    DNFATAL_IF(batch_index >= dimensions().back(), "[" << name() << "]: batch index exceeds the number of batches!")
    size_t sizeIn = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_t>());
    DNFATAL_IF(sizeIn != sizeOfBatch(), "[" << name() << "]: 'shape' has incompatible size: " << sizeIn << "vs " << sizeOfBatch());
    return TensorMapType(data() + batch_index * sizeOfBatch(), shape, &namedTensor_.first);
  }
  
  auto batch_block(size_t start_index, size_t number_of_batches) const {
    DNFATAL_IF(size() == 0, "[" << name() << "]: This tensor is empty!")
    DNFATAL_IF(!isBatched(),  "[" << name() << "]: This method is for batched tensors!")
    DNFATAL_IF((start_index + number_of_batches > dimensions().back()),
      "[" << name() << "]: The last batch index exceeds the number of batches!");
    std::vector<size_t> dims(dimensions().begin(), dimensions().end()-1);
    dims.push_back(number_of_batches);
    return ConstTensorMapType(data() + start_index * sizeOfBatch(), dims, &namedTensor_.first,
      allocator_.timesteps().data() + start_index, number_of_batches);
  }

  auto batch_block(size_t start_index, size_t number_of_batches) {
    DNFATAL_IF(size() == 0, "[" << name() << "]: This tensor is empty!")
    DNFATAL_IF(!isBatched(),  "[" << name() << "]: This method is for batched tensors!")
    DNFATAL_IF((start_index + number_of_batches > dimensions().back()),
      "[" << name() << "]: The last batch index exceeds the number of batches!");
    std::vector<size_t> dims(dimensions().begin(), dimensions().end()-1);
    dims.push_back(number_of_batches);
    return TensorMapType(data() + start_index * sizeOfBatch(), dims, &namedTensor_.first,
      allocator_.timesteps().data()+ start_index, number_of_batches);
  }
  
  auto block(size_t start_index, size_t number_of_elements) const {
    DNFATAL_IF(size() == 0, "[" << name() << "]: This tensor is empty!")
    DNFATAL_IF(!isBatched(), "[" << name() << "]: This method is for batched tensors!")
    DNFATAL_IF((start_index + number_of_elements > size()/sizeOfDatum()), "[" << name() << "]: The last index exceeds the number of data!");
    auto dims = datumDimensions();
    dims.push_back(number_of_elements);
    return ConstTensorMapType(data() + start_index * sizeOfDatum(), dims, &namedTensor_.first);
  }
  
  auto block(size_t start_index, size_t number_of_elements) {
    DNFATAL_IF(size() == 0, "[" << name() << "]: This tensor is empty!")
    DNFATAL_IF(!isBatched(), "[" << name() << "]: This method is for batched tensors!")
    DNFATAL_IF((start_index + number_of_elements > size()/sizeOfDatum()), "[" << name() << "]: The last index exceeds the number of data!");
    auto dims = datumDimensions();
    dims.push_back(number_of_elements);
    return TensorMapType(data() + start_index * sizeOfDatum(), dims, &namedTensor_.first);
  }
  
  auto datum(size_t index) const {
    auto dims = datumDimensions();
    return ConstTensorMapType(data() + index * sizeOfDatum(), dims, &namedTensor_.first);
  }
  
  auto datum(size_t index) {
    auto dims = datumDimensions();
    return TensorMapType(data() + index * sizeOfDatum(), dims, &namedTensor_.first);
  }
  
  /*
   * Batch manipulation operations
   */

  //
  // TODO: 1. break this up into multiple functions
  // TODO: 2. refactor and simplify this push-back function
  //

  /*!
   * @brief Adds the data of the input tensor to the current.
   *
   *  - If `in` is bigger than a batch (i.e. in.size() > sizeOfBatch()), in.size/sizeOfBatch() number of batches are added.
   *  - If `in` is batched:
   *        1. appends timesteps() of `in` in the back.
   *  - If this tensor is empty:
   *        1. If `in` is batched, deep copies data and shape of `in`.
   *        2. else, the tensor's shape becomes [in_shape, 1, 1] (i.e. regards `in` as a datum)
   *
   * @param tensor The source tensor to be added to the current tensor instance.
   */
  void pushBack(const Tensor& tensor) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!");
    int dimCheck = compareDimensions(dimensions(), tensor.dimensions());
    size_t number_of_new_batches = tensor.size() / sizeOfBatch();
    size_t previous_batch_num = dimensions().back();
    bool expand_time = false;
    size_t new_time_capacity = timeStepCapacity();
    if (tensor.isBatched()) {
      NFATAL_IF(tensor.rank() != rank(),"[" << name() << "]: Rank mismatch!")
      NFATAL_IF(dimCheck < (int)rank() - 2, "[" << name() << "]: Datum shape mismatch!")
      number_of_new_batches = tensor.batches();
      for (size_t i = 0; i < number_of_new_batches; i++) {
        if (tensor.timesteps()[i]> new_time_capacity) {
          expand_time = true;
          new_time_capacity = tensor.timesteps()[i];
        }
      }
      if (expand_time) {
        resizeBatches(new_time_capacity, previous_batch_num + number_of_new_batches);
      } else {
        resizeBatches(previous_batch_num + number_of_new_batches);
      }
      size_t position = previous_batch_num * sizeOfBatch();
      size_t position2 = 0;
      for (size_t i = 0; i < number_of_new_batches; i++) {
        allocator_.timesteps()[previous_batch_num + i] = tensor.timesteps()[i];
        std::memcpy(data() + position, tensor.data() + position2, sizeof(ScalarType) * sizeOfDatum() * tensor.timesteps()[i]);
        position += sizeOfBatch();
        position2 += tensor.size()/tensor.batches();
      }
    } else {
      NFATAL_IF(tensor.rank() != rank() - 1,"[" << name() << "]: Rank mismatch!")
      NFATAL_IF(dimCheck < (int) rank() - 1, "[" << name() << "]: Batch shape mismatch!")
      NFATAL_IF(number_of_new_batches != 1, "[" << name() << "]: Undefined behavior!")
      resizeBatches(previous_batch_num + 1);
      allocator_.timesteps()[previous_batch_num] = timeStepCapacity();
      std::memcpy(data() + previous_batch_num * sizeOfBatch(), tensor.data(), sizeof(ScalarType) * sizeOfBatch() );
    }
  }

  /*!
   *  If `in` is batched (i.e. in.timesteps() is not empty), copies time steps.
   */
  template<typename DerivedScalarType_, typename DerivedIndexType_>
  void pushBack(const TensorMap<DerivedScalarType_, DerivedIndexType_>& in) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!");
    int dimCheck = compareDimensions(dimensions(), in.dimensions());
    size_t number_of_new_batches = in.size() / sizeOfBatch();
    size_t previous_batch_num = dimensions().back();
    bool expand_time = false;
    size_t new_time_capacity = timeStepCapacity();
    if (in.isBatched()) {
      NFATAL_IF(dimCheck < (int)rank() - 2, "[" << name() << "]: Datum shape mismatch")
      number_of_new_batches = in.batches();
      resizeBatches(dimensions().back() + number_of_new_batches);
      for (size_t i = 0; i < number_of_new_batches; i++) {
        if (in.timesteps()[i] > new_time_capacity) {
          expand_time = true;
          new_time_capacity = in.timesteps()[i];
        }
      }
      if (expand_time) {
        resizeBatches(new_time_capacity, previous_batch_num + number_of_new_batches);
      } else {
        resizeBatches(previous_batch_num + number_of_new_batches);
      }
      size_t position = previous_batch_num * sizeOfBatch();
      size_t position2 = 0;
      for (size_t i = 0; i < number_of_new_batches; i++) {
        allocator_.timesteps()[previous_batch_num + i] = in.timesteps()[i];
        std::memcpy(data() + position, in.data() + position2, sizeof(ScalarType) * sizeOfDatum() * in.timesteps()[i]);
        position += sizeOfBatch();
        position2 += in.size()/in.batches();
      }
    } else {
      NFATAL_IF(dimCheck < (int) rank() - 1, "[" << name() << "]: Batch shape mismatch")
      NFATAL_IF(number_of_new_batches != 1, "[" << name() << "]: Undefined behavior. ")
      resizeBatches(previous_batch_num + 1);
      allocator_.timesteps()[previous_batch_num] = timeStepCapacity();
      std::memcpy(data() + previous_batch_num * sizeOfBatch(), in.data(), sizeof(ScalarType) * sizeOfBatch() );
    }
  }

  /*!
   * Appends data along the time axis of specified batch.
   * Automatically reserves 2X space if time axis or batch axis is full.
   * If batch_index + 1 > number of batches, pushes back to a new batch.
   * @param batch_index
   * @param in
   */
  void pushBack(size_t batch_index, const Tensor& in) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!");
    NFATAL_IF(in.dimensions().size() > rank(), "[" << name() << "]: Argument's rank is not compatible!");
    NFATAL_IF(in.datumDimensions() != datumDimensions(), "[" << name() << "]: Argument's datum shape is not compatible!");
    size_t new_time_steps = in.size() / sizeOfDatum();
    if (new_time_steps > 1) { // multi time steps
      NFATAL_IF(in.batches() > 1,  "[" << name() << "]: Undefined behavior");
      new_time_steps = in.timesteps()[0];
    }
    if (batch_index + 1 > batches()) {
      allocator_.timesteps().resize(batch_index + 1, 0);
    }
    size_t oldTimeStep = timesteps()[batch_index];
    allocator_.timesteps()[batch_index] += new_time_steps;
    if (timesteps()[batch_index] >= timeStepCapacity() && batch_index + 1 > batches()) {
      resizeBatches(timesteps()[batch_index], batch_index + 1);
    } else if (timesteps()[batch_index] >= timeStepCapacity()) {
      resizeBatches(timesteps()[batch_index], batches());
    } else if (batch_index + 1 > batches()) {
      resizeBatches(batch_index + 1);
    }

    size_t position = batch_index * sizeOfBatch() + oldTimeStep * sizeOfDatum();
    if (new_time_steps > 1) {
      std::memcpy(data() + position, in.data() , sizeof(ScalarType) * in.timesteps()[0] * sizeOfDatum());
    } else {
      std::memcpy(data() + position, in.data(), sizeof(ScalarType) * in.size());
    }
  }

  template<typename DerivedScalarType_, typename DerivedIndexType_>
  void pushBack(size_t batch_index, const TensorMap<DerivedScalarType_, DerivedIndexType_>& in) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!");
    NFATAL_IF(in.dimensions() != datumDimensions(), "[" << name() << "]: Argument's datum shape is not compatible!");
    
    size_t new_time_steps = in.size() / sizeOfDatum();
    if (new_time_steps > 1) { // multi time steps
      NFATAL_IF(in.batches() > 1,  "[" << name() << "]: Cannot push-back multiple batches of steps (undefined behavior)!")
      new_time_steps = in.timesteps()[0];
    }
    if (batch_index + 1 > batches()) {
      allocator_.timesteps().resize(batch_index + 1, 0);
    }
    
    size_t oldTimeStep = timesteps()[batch_index];
    allocator_.timesteps()[batch_index] += new_time_steps;
    if (timesteps()[batch_index] >= timeStepCapacity() && batch_index + 1 > batches()) {
      resizeBatches(timesteps()[batch_index], batch_index + 1);
    } else if (timesteps()[batch_index] >= timeStepCapacity()) {
      resizeBatches(timesteps()[batch_index], batches());
    } else if (batch_index + 1 > batches()) {
      resizeBatches(batch_index + 1);
    }
    
    // TODO: why does coservativeResize() copy to buffer and back?
    // TODO: why does re-size call coservativeResize() and memcopy if the memory does not need to be re-allocated or modified?
    // TODO: why not replace the if-else with std::max() over both timesteps and batches?
    // TODO: reserve allocator_.timesteps() according to capacity
    
    size_t position = batch_index * sizeOfBatch() + oldTimeStep * sizeOfDatum();
    size_t size = sizeof(ScalarType) * sizeOfDatum() * new_time_steps;
    if (new_time_steps > 1) {
      size = sizeof(ScalarType) * in.timesteps()[0] * sizeOfDatum();
    }
    std::memcpy(data() + position, in.data(), size);
  }

  void popBackBatch(size_t batch_index) {
    eraseBatch(batch_index);
  }

  void popBackBatch(size_t batch_index, Tensor& popped) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!")
    NFATAL_IF(sizeOfDatum() == 0, "[" << name() << "]: The tensor's shape is not defined!")
    NFATAL_IF((batch_index >= dimensions().back()),
      "[" << name() << "]: batch_index should be smaller that the last idx: " << batch_index << ">" << dimensions().back() - 1);
    popped = operator()(batch_index);
    eraseBatch(batch_index);
  }

  void popBackTimeSteps(size_t number_of_timesteps) {
    for (size_t i = batches(); i > 0; --i) {
      popBackTimeSteps(i - 1, number_of_timesteps);
    }
  }

  void popBackTimeSteps(size_t batch_index, size_t number_of_timesteps) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!")
    NFATAL_IF(sizeOfDatum() == 0, "[" << name() << "]: The tensor's shape is not defined!")
    NFATAL_IF((batch_index >= dimensions().back()),
      "[" << name() << "]: batch_index should be smaller that the last idx: " << batch_index << " vs " << dimensions().back() - 1);
    NFATAL_IF(number_of_timesteps > timesteps()[batch_index],
      "[" << name() << "]: Argument must not exceed the current timesteps: " << number_of_timesteps << " vs " << timesteps()[batch_index]);
    allocator_.timesteps()[batch_index] -= number_of_timesteps;
    size_t position = batch_index * sizeOfBatch() + timesteps()[batch_index] * sizeOfDatum();
    std::fill(data() + position, data() + position + number_of_timesteps * sizeOfDatum(), 0.0);
  }

  /*!
   * Shrink underlying storage to fit dimensions() (i.e. batchCapacities = dimensions().back();
   */
  void shrinkBatches() {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!")
    NFATAL_IF(sizeOfDatum() == 0, "[" << name() << "]: The tensor's shape is not defined!")
    allocator_.allocateBatched(dimensions(), dimensions().back(), true);
    allocator_.timesteps().resize(dimensions().back());
  }

  /*!
   * Removes a batch, keeps total capacity.
   */
  void eraseBatch(size_t batch_index) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors!")
    NFATAL_IF(sizeOfDatum() == 0, "[" << name() << "]: The tensor's shape is not defined!")
    NFATAL_IF((batch_index >= dimensions().back()),
      "[" << name() << "]: batch_index should be smaller that the last idx: " << batch_index << ">" << dimensions().back() - 1);
    if (batch_index +1 < dimensions().back()){ // erase intermediate batch
      size_t size_behind = dimensions().back() - 1 - batch_index;
      size_behind *= sizeOfBatch();
      auto temp = new ScalarType[size_behind];// temporary storage
      memcpy(temp, data() + (batch_index + 1) * sizeOfBatch(), sizeof(ScalarType) * size_behind);
      memcpy(data() + batch_index * sizeOfBatch(), temp, sizeof(ScalarType) * size_behind);
      delete[] temp;
      allocator_.dimensions().back() -= 1;
      std::fill(data() + sizeOfBatch() * dimensions().back(), data() + sizeOfBatch()* (dimensions().back() + 1), 0.0);
      allocator_.timesteps().erase(timesteps().begin() + batch_index);
      allocator_.size() -= sizeOfBatch();
    } else {
      allocator_.dimensions().back() -= 1;
      std::fill(data() + sizeOfBatch() * dimensions().back(), data() + sizeOfBatch() * (dimensions().back() + 1), 0.0);
      allocator_.timesteps().resize(dimensions().back());
      allocator_.size() -= sizeOfBatch();
    }
  }

  /*!
   * Collects batches of indices between start_index ~ end_index and resize conservatively
   */
  void keepOnlyBatches(size_t start_index, size_t end_index, bool keep_capacity=true) {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors")
    NFATAL_IF((start_index > end_index), "[" << name() << "]: start_index cannot be smaller than end_index");
    NFATAL_IF((end_index > dimensions().back() - 1), "[" << name() << "]: endBatch_ID exceeds # of batches");
    size_t batchNum = end_index - start_index + 1;
    // temporary storage
    auto temp = new ScalarType[size()];// temporary storage
    memcpy(temp, data(), sizeof(ScalarType) * size());
    // non-conservative resize
    std::vector<size_t> new_dimension = dimensions();
    size_t newCapacity = 0;
    new_dimension.back() = batchNum;
    if (!keep_capacity) newCapacity = batchNum;
    allocator_.allocateBatched(new_dimension, newCapacity, false);
    memcpy(data(), temp + start_index * sizeOfBatch(), sizeof(ScalarType) * size());
    delete[] temp;
    std::vector<size_t>::const_iterator first = timesteps().begin() + start_index;
    std::vector<size_t>::const_iterator last = timesteps().begin() + end_index + 1;
    std::vector<size_t> timeSteps(first, last);
    allocator_.timesteps() = timeSteps;
  }

  /*!
    *   flattens last dimension. keep the shape of the data
    *   [data shape, timesteps, batchnum] -> [data shape, totalLength]
    */
  Tensor getFlattenedBatches(const size_t validTimeSteps = 1) const {
    NFATAL_IF(!isBatched(), "[" << name() << "]: This method works only for the batched tensors")
    NFATAL_IF(rank() < 2, "[" << name() << "]: this method supports tensors with dimensions at least 2");
    std::vector<size_t> flattenedDimensions;
    size_t totalLength = 0;
    for (size_t i = 0; i < dimensions().back(); i++) {
      if (timesteps()[i] >= validTimeSteps) {
        totalLength += timesteps()[i];
      }
    }
    for (size_t i = 0; i < rank() - 2; i++) {
      flattenedDimensions.push_back(dimensions()[i]);
    }
    flattenedDimensions.push_back(totalLength);
    flattenedDimensions.push_back(1);
    Tensor output(flattenedDimensions, true);
    output.setTimeSteps(0, totalLength);
    size_t position = 0;
    for (size_t i = 0; i < dimensions().back(); i++) {
      if (timesteps()[i] >= validTimeSteps) {
        size_t size = sizeOfDatum() * timesteps()[i];
        memcpy(output.data() + position, data() + sizeOfBatch() * i, size * sizeof(ScalarType));
        position += size;
      }
    }
    return output;
  }

  /*
   * Copy operations
   */
  
  /*!
   * @brief Creates an exact duplicate of a target tensor, in value and in name.
   * @note New storage is allocated and set to the same capacities, dimensions and values as the source.
   * @param other The source tensor to be duplicated.
   */
  void clone(const Tensor& other) {
    copy(other);
    setName(other.name());
  }

  /*!
   * Copies data from noesis::Tensor (data, shape)
   */

  /*!
   * @brief Creates a data-level copy of a target tensor, but only in values. Current name of the destination tensor is preserved.
   * @note New storage is allocated and set to the same capacities, dimensions and values as the source.
   * @param other
   */
  void copy(const Tensor& other) {
    allocator_.copy(other.allocator_);
    if (!namedTensor_.second.IsSameSize(other.namedTensor_.second) || namedTensor_.second.SharesBufferWith(other.namedTensor_.second)) {
      namedTensor_.second = tensorflow::Tensor(allocator_.dtype(), other.namedTensor_.second.shape());
    } else {
      std::vector<size_t> dims = allocator_.dimensions();
      if (allocator_.isBatched()) {
        dims.end()[-2] = allocator_.capacities()[0];
        dims.end()[-1] = allocator_.capacities()[1];
      }
      allocator_.reshape(dims, allocator_.isBatched());
    }
    allocator_.setTensor(namedTensor_.second);
    memcpy(data(), other.data(), sizeof(ScalarType) * size());
  }

  /*
   * Assignment operations
   */

  /*!
   * @brief Copy-assignment sets the current lhs instance to share the same underlying storage as the rhs.
   * @param rhs The source tensor whose dimensions and storage is to be shared by the lhs.
   * @return Reference to the current lhs instance.
   */
  Tensor& operator=(const Tensor& rhs) {
    allocator_.copy(rhs.allocator_);
    namedTensor_.second = rhs.namedTensor_.second;
    allocator_.setTensor(namedTensor_.second);
    return *this;
  }

  /*!
   * @brief Move-assignment sets the current lhs instance to share the same underlying storage as the rhs.
   * @param rhs The source tensor whose dimensions and storage is to be shared by the lhs.
   * @return Reference to the current lhs instance.
   */
  Tensor& operator=(Tensor&& rhs) noexcept {
    allocator_.move(rhs.allocator_);
    namedTensor_.second = std::move(rhs.namedTensor_.second);
    return *this;
  }

  /*!
   * @brief Copy-assignment from map type copies all mapped values into the lhs tensor instance.
   * @param rhs The source tensor map whose data is to be copied to the lhs tensor.
   * @return Reference to the current lhs instance.
   */
  template<typename DerivedScalarType_, typename DerivedIndexType_>
  Tensor& operator=(const TensorMap<DerivedScalarType_, DerivedIndexType_>& rhs) {
    NFATAL_IF(typeid(ScalarType_) != typeid(DerivedScalarType_), "[" << name() << "]: Scalar type of 'rhs' is not compatible!")
    NFATAL_IF(typeid(size_t) != typeid(DerivedIndexType_), "[" << name() << "]: Index type of 'rhs' is not compatible!")
    auto new_dims = rhs.dimensions();
    auto batches = rhs.batches();
    if (batches == 1) {
      new_dims.push_back(batches);
    }
    resize(new_dims, rhs.isBatched());
    std::memcpy(data(), rhs.data(), sizeof(ScalarType) * size());
    for (size_t i = 0; i < batches; i++) {
      allocator_.timesteps()[i] = rhs.timesteps()[i];
    }
    return *this;
  }

  /*!
   * @brief Copy-assignment from a low-level tensorflow::Tensor object.
   * @note lhs' underlying storage is set from rhs argument.
   * @warning This operation keeps batching capability of the lhs tensor.
   * @param rhs The source tensorflow::Tensor instance whose underlying storage is to be assigned to current.
   * @return Reference to the current lhs instance.
   */
  Tensor& operator=(const tensorflow::Tensor& rhs) {
    // Initialize new tensor allocation definitions
    bool is_batched = false;
    std::vector<size_t> new_dims;
    std::vector<size_t> new_timesteps;
    // Retrieve the dimensions of the source tensorflow tensor
    for (int i = rhs.dims()-1; i > -1; i--) {
      new_dims.push_back(static_cast<size_t>(rhs.dim_size(i)));
    }
    // Check for the special case of empty tensors
    if (rhs.NumElements() == 0) {
      new_dims.clear();
      new_dims.push_back(0);
    }
    // Force batching if the tensor has been configured and the dimensions disagree
    if ( (size() != 0) && (dimensions() != new_dims) ) {
      new_timesteps.resize(1, new_dims.back());
      new_dims.push_back(1);
      is_batched = true;
    } else {
      new_timesteps.clear();
    }
    // Set the internals
    allocator_.setAllocation(new_dims, new_timesteps, is_batched);
    // Capture the buffer to the actual data
    namedTensor_.second = rhs;
    return *this;
  }

  /*
   * Arithmetic operations
   */

  Tensor& operator+=(const Tensor& rhs) {
    NFATAL_IF(dimensions() != rhs.dimensions(), "[" << name() << "]: 'rhs' dimensions are not compatible with current!");
    asEigenMatrix() += rhs.asEigenMatrix();
    return *this;
  }

  Tensor& operator-=(const Tensor& rhs) {
    NFATAL_IF(dimensions() != rhs.dimensions(), "[" << name() << "]: 'rhs' dimensions are not compatible with current!");
    asEigenMatrix() -= rhs.asEigenMatrix();
    return *this;
  }

  Tensor& operator*=(const Tensor& rhs) {
    NFATAL_IF(dimensions() != rhs.dimensions(), "[" << name() << "]: 'rhs' dimensions are not compatible with current!");
    asEigenMatrix() = asEigenMatrix().cwiseProduct(rhs.asEigenMatrix());
    return *this;
  }

  Tensor& operator/=(const Tensor& rhs) {
    NFATAL_IF(dimensions() != rhs.dimensions(), "[" << name() << "]: 'rhs' dimensions are not compatible with current!");
    asEigenMatrix() = asEigenMatrix().cwiseQuotient(rhs.asEigenMatrix());
    return *this;
  }
  
  template<typename DerivedScalarType_, typename DerivedIndexType_>
  Tensor& operator+=(const TensorMap<DerivedScalarType_,DerivedIndexType_>& rhs) {
    NFATAL_IF(dimensions() != rhs.dimensions(), "[" << name() << "]: 'rhs' dimensions are not compatible with current!");
    asEigenMatrix() += rhs.asEigenMatrix();
    return *this;
  }
  
  template<typename DerivedScalarType_, typename DerivedIndexType_>
  Tensor& operator-=(const TensorMap<DerivedScalarType_,DerivedIndexType_>& rhs) {
    NFATAL_IF(dimensions() != rhs.dimensions(), "[" << name() << "]: 'rhs' dimensions are not compatible with current!");
    asEigenMatrix() -= rhs.asEigenMatrix();
    return *this;
  }
  
  template<typename DerivedScalarType_, typename DerivedIndexType_>
  Tensor& operator*=(const TensorMap<DerivedScalarType_,DerivedIndexType_>& rhs) {
    NFATAL_IF(dimensions() != rhs.dimensions(), "[" << name() << "]: 'rhs' dimensions are not compatible with current!");
    asEigenMatrix() = asEigenMatrix().cwiseProduct(rhs.asEigenMatrix());
    return *this;
  }
  
  template<typename DerivedScalarType_, typename DerivedIndexType_>
  Tensor& operator/=(const TensorMap<DerivedScalarType_,DerivedIndexType_>& rhs) {
    NFATAL_IF(dimensions() != rhs.dimensions(), "[" << name() << "]: 'rhs' dimensions are not compatible with current!");
    asEigenMatrix() = asEigenMatrix().cwiseQuotient(rhs.asEigenMatrix());
    return *this;
  }
  
  Tensor& operator+=(ScalarType rhs) {
    asEigenMatrix().array() += rhs;
    return *this;
  }
  
  Tensor& operator-=(ScalarType rhs) {
    asEigenMatrix().array() -= rhs;
    return *this;
  }
  
  Tensor& operator*=(ScalarType rhs) {
    asEigenMatrix().array() *= rhs;
    return *this;
  }
  
  Tensor& operator/=(ScalarType rhs) {
    auto eps = std::numeric_limits<ScalarType>::epsilon();
    asEigenMatrix().array() /= (rhs + eps);
    return *this;
  }
  
  friend Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    Tensor result;
    result.copy(lhs);
    result += rhs;
    return result;
  }
  
  friend Tensor operator+(const Tensor& lhs, Tensor&& rhs) {
    rhs += lhs;
    return rhs;
  }
  
  friend Tensor operator+(Tensor&& lhs, const Tensor& rhs) {
    lhs += rhs;
    return lhs;
  }
  
  friend Tensor operator+(Tensor&& lhs, Tensor&& rhs) {
    lhs += rhs;
    return lhs;
  }
  
  friend Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    Tensor result;
    result.copy(lhs);
    result -= rhs;
    return result;
  }
  
  friend Tensor operator-(Tensor&& lhs, const Tensor& rhs) {
    lhs -= rhs;
    return lhs;
  }
  
  friend Tensor operator-(const Tensor& lhs, Tensor&& rhs) {
    rhs -= lhs;
    return (-1)*rhs;
  }
  
  friend Tensor operator-(Tensor&& lhs, Tensor&& rhs) {
    lhs -= rhs;
    return lhs;
  }
  
  friend Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    Tensor result;
    result.copy(lhs);
    result *= rhs;
    return result;
  }
  
  friend Tensor operator*(Tensor&& lhs, const Tensor& rhs) {
    lhs *= rhs;
    return lhs;
  }
  
  friend Tensor operator*(const Tensor& lhs, Tensor&& rhs) {
    rhs *= lhs;
    return rhs;
  }
  
  friend Tensor operator*(Tensor&& lhs, Tensor&& rhs) {
    lhs *= rhs;
    return lhs;
  }
  
  friend Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    Tensor result;
    result.copy(lhs);
    result /= rhs;
    return result;
  }
  
  friend Tensor operator/(Tensor&& lhs, const Tensor& rhs) {
    lhs /= rhs;
    return lhs;
  }
  
  friend Tensor operator/(const Tensor& lhs, Tensor&& rhs) {
    Tensor result;
    result.copy(lhs);
    result /= rhs;
    return result;
  }
  
  friend Tensor operator/(Tensor&& lhs, Tensor&& rhs) {
    lhs /= rhs;
    return lhs;
  }
  
  friend Tensor operator+(const Tensor& lhs, ScalarType rhs) {
    Tensor result;
    result.copy(lhs);
    result += rhs;
    return result;
  }
  
  friend Tensor operator+(Tensor&& lhs, ScalarType rhs) {
    lhs += rhs;
    return lhs;
  }
  
  template <typename T>
  friend Tensor operator+(ScalarType lhs, T&& rhs) {
    return operator+(std::forward<T>(rhs), lhs);
  }
  
  friend Tensor operator-(const Tensor& lhs, ScalarType rhs) {
    Tensor result;
    result.copy(lhs);
    result -= rhs;
    return result;
  }
  
  friend Tensor operator-(Tensor&& lhs, ScalarType rhs) {
    lhs -= rhs;
    return lhs;
  }
  
  template <typename T>
  friend Tensor operator-(ScalarType lhs, T&& rhs) {
    return (-1)*operator-(std::forward<T>(rhs), lhs);
  }
  
  friend Tensor operator*(const Tensor& lhs, ScalarType rhs) {
    Tensor result;
    result.copy(lhs);
    result *= rhs;
    return result;
  }
  
  friend Tensor operator*(Tensor&& lhs, ScalarType rhs) {
    lhs *= rhs;
    return lhs;
  }
  
  template <typename T>
  friend Tensor operator*(ScalarType lhs, T&& rhs) {
    return operator*(std::forward<T>(rhs), lhs);
  }
  
  friend Tensor operator/(Tensor& lhs, ScalarType rhs) {
    Tensor result;
    result.copy(lhs);
    result /= rhs;
    return result;
  }
  
  friend Tensor operator/(Tensor&& lhs, ScalarType rhs) {
    lhs /= rhs;
    return lhs;
  }
  
  template <typename T>
  friend Tensor operator/(ScalarType lhs, T&& rhs) {
    return operator/(std::forward<T>(rhs), lhs);
  }
  
  /*
   * Value setting operations
   */

  void setConstant(const ScalarType constant) {
    NFATAL_IF(size() == 0, "[" << name() << "]: Cannot set values for empty Tensor.");
    asEigenMatrix().setConstant(constant);
    if (isBatched() && rank() > 1) {
      allocator_.fillTimeSteps();
    }
  }

  void setZero() {
    NFATAL_IF(size() == 0, "[" << name() << "]: Cannot set values for empty Tensor.");
    asEigenMatrix().setConstant(0);
    if (isBatched() && rank() > 1) {
      allocator_.fillTimeSteps();
    }
  }
  
  void setRandom() {
    asEigenMatrix().setRandom();
  }
  
  void setRandomUnitUniform(RandomNumberGeneratorType& generator) {
    NFATAL_IF(size() == 0, "[" << name() << "]: Cannot set values for empty Tensor.");
    for (size_t dataIndex = 0; dataIndex < size(); dataIndex++) {
      data()[dataIndex] = generator.sampleUnitUniform();
    }
    if (isBatched() && rank() > 1) {
      allocator_.fillTimeSteps();
    }
  }

  void setRandomUniform(RandomNumberGeneratorType& generator, const Tensor& min, const Tensor& max) {
    NFATAL_IF(size() == 0, "[" << name() << "]: Cannot set values for empty Tensor.");
    NFATAL_IF(min.dimensions() != datumDimensions(), "[" << name() << "]: bounds should match the datum dimensions of this tensor.");
    if (isBatched() && rank() > 1) {
      allocator_.fillTimeSteps();
      for (size_t batch=0; batch<batches(); batch++) {
        for (size_t time=0; time < maxTimeSteps(); time++) {
          auto datum = (*this)(time, batch);
          for (size_t k = 0; k < datum.size(); k++) {
            datum[k] = generator.sampleUniform(min[k], max[k]);
          }
        }
      }
    } else {
      for (size_t k = 0; k < size(); k++) {
        data()[k] = generator.sampleUniform(min[k], max[k]);
      }
    }
  }

  void setRandomStandardNormal(RandomNumberGeneratorType& generator) {
    NFATAL_IF(size() == 0, "[" << name() << "]: Cannot set values for empty Tensor.");
    for (size_t k = 0; k < size(); k++) {
      data()[k] = generator.sampleStandardNormal();
    }
    if (isBatched() && rank() > 1) {
      allocator_.fillTimeSteps();
    }
  }

  void setRandomNormal(RandomNumberGeneratorType& generator, const Tensor& min, const Tensor& max) {
    NFATAL_IF(size() == 0, "[" << name() << "]: Cannot set values for empty Tensor.");
    NFATAL_IF(min.dimensions() != datumDimensions(), "[" << name() << "]: bounds should match the datum dimensions of this tensor.");
    if (isBatched() && rank() > 1) {
      allocator_.fillTimeSteps();
      for (size_t batch=0; batch<batches(); batch++) {
        for (size_t time=0; time<maxTimeSteps(); time++) {
          auto datum = (*this)(time, batch);
          for (size_t k = 0; k < datum.size(); k++) {
            auto delta = max[k] - min[k];
            auto center = 0.5 * delta + min[k];
            datum[k] = (0.166666667 * delta) * generator.sampleStandardNormal()  + center; // NOTE: 3*sigma ~= delta/2
          }
        }
      }
    } else {
      for (size_t k = 0; k < size(); k++) {
        auto delta = max[k] - min[k];
        auto center = 0.5 * delta + min[k];
        data()[k] = (0.166666667 * delta) * generator.sampleStandardNormal() + center; // NOTE: 3*sigma ~= delta/2
      }
    }
  }
  
  /*
   * Shuffling operations
   */
  
  void shuffle(const Eigen::VectorXi& indices) {
    const auto dim = sizeOfDatum();
    auto map = MatrixMap(data(), dim, size() / dim);
    map = map * indices.asPermutation();
  }
  
  /*
   * Casting operations
   */

  explicit operator std::pair<std::string, tensorflow::Tensor>() const {
    return namedTensor_;
  };

  explicit operator std::pair<std::string, tensorflow::Tensor>() {
    return namedTensor_;
  };

  /*
   * Data validity operations
   */

  bool hasNaN() const {
    return asEigenMatrix().hasNaN();
  }

  bool hasInf() const {
    return !asEigenMatrix().allFinite();
  }
  
  bool allFinite() const {
    return asEigenMatrix().allFinite();
  }

  /*
   * Helper functions
   */
  
  bool hasSameStorageWith(const Tensor& tensor) {
    return namedTensor_.second.SharesBufferWith(tensor.namedTensor_.second);
  }
  
  std::string info() const {
    std::string info = "\n[noesis::Tensor]";
    info += "\nName: '" + name() + "'";
    info += allocator_.info();
    info += "\nTensorFlow " + namedTensor_.second.DebugString();
    return info;
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.info() << "\nData:";
    if (tensor.size() != 0) {
      if (tensor.isBatched()) {
        for (size_t b = 0; b < tensor.batches(); ++b) {
          os << "\n(" << b << ")----------\n" << tensor(b).asEigenMatrix();
        }
      } else {
        os << "\n" << tensor.asEigenMatrix();
      }
    } else {
      os << " Empty\n";
    }
    return os;
  }

private:

  /*!
   * @brief Helper function which compares two dimensions vectors for how many dimensions elements match.
   * @param dims0 The first dimensions set.
   * @param dims1 The second dimensions set.
   * @return The number of matching dimensions. If all dimensions match, then the result is the rank of both dimensions.
   */
  inline int compareDimensions(const std::vector<size_t> &dims0, const std::vector<size_t> &dims1) {
    // First compare rank - if ranks do not match then return negative number
    if (dims0.size() != dims1.size()) {
      return -1;
    }
    // Compare each dimension and return how many are the same
    size_t number_of_dims = dims0.size();
    for (size_t i = 0; i < number_of_dims; i++) {
      if (dims0[i] != dims1[i]) {
        return static_cast<int>(i);
      }
    }
    // Fall through means that all dimensions match
    return static_cast<int>(number_of_dims);
  }

private:
  //! @brief  Container to the underlying TensorFlow tensor container associated with a lexical (string) identifier.
  std::pair<std::string, tensorflow::Tensor> namedTensor_;
  //! @brief Internal module which manages the memory allocation and configuration of the Tensor instance.
  TensorAllocator<ScalarType> allocator_;
}; // class Tensor

} // namespace noesis

#endif // NOESIS_FRAMEWORK_CORE_TENSOR_HPP_

/* EOF */
