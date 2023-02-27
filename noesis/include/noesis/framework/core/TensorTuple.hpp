/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_CORE_TENSOR_TUPLE_HPP_
#define NOESIS_FRAMEWORK_CORE_TENSOR_TUPLE_HPP_

// Boost
#include <boost/filesystem/path.hpp>

// Noesis
#include "noesis/framework/core/Tensor.hpp"
#include "noesis/framework/core/TensorsSpec.hpp"
#include "noesis/framework/utils/string.hpp"

namespace noesis {

template<typename ScalarType_, typename IndexType_=size_t>
class TensorTuple
{
public:
  // Aliases
  using IndexType = IndexType_;
  using ScalarType = ScalarType_;
  using TensorType = Tensor<ScalarType>;
  using TensorVectorType = std::vector<TensorType>;
  using DimensionsType = std::vector<size_t>;
  using DimensionsVectorType = std::vector<DimensionsType>;
  
  /*!
   * @brief Constructor for complete specification of tensor tuple instances from TensorsSpec object.
   * @param scope The scope within which the tuple instance is defined.
   * @param spec A tensor specification object to configure the tensor elements from.
   * @param time_size The number of time-steps for each tensor element.
   * @param batch_size The number of batches for each tensor element.
   */
  explicit TensorTuple(const std::string& scope,
                       const TensorsSpec& spec,
                       size_t time_size,
                       size_t batch_size):
    tensors_(spec.size()),
    datumDimensions_(),
    scope_(scope),
    timesteps_(time_size),
    batches_(batch_size)
  {
    NFATAL_IF(time_size*batch_size==0 && time_size+batch_size!=0,
      "[" << this->scope() << "]: 'time_size' and 'batch_size' must be either both zero or non-zero!");
    setFromSpec(spec);
  }
  
  /*!
   * @brief Constructor for complete specification of tensor tuple instances.
   * @param scope The scope within which the tuple instance is defined.
   * @param names The names to be assigned to the tensor elements.
   * @param dimensions The vector specifying the dimensions for each constructed tensor element.
   * @param time_size The number of time-steps for each tensor element.
   * @param batch_size The number of batches for each tensor element.
   */
  explicit TensorTuple(const std::string& scope,
                       const std::vector<std::string>& names,
                       const DimensionsVectorType& dimensions,
                       size_t time_size,
                       size_t batch_size):
    tensors_(dimensions.size()),
    datumDimensions_(dimensions),
    scope_(scope),
    timesteps_(time_size),
    batches_(batch_size)
  {
    NFATAL_IF(time_size*batch_size==0 && time_size+batch_size!=0,
      "[" << this->scope() << "]: 'time_size' and 'batch_size' must be either both zero or non-zero!");
    setNames(names);
    configureTensors();
  }

  /*!
   * @brief Constructs a tensor tuple from a vector of dimensions but without scoping.
   * @note if time_size and batch_size are non-zero, then all tensors constructed will be batched.
   * @param dimensions The vector specifying the dimensions for each constructed tensor element.
   * @param time_size The number of time-steps for each tensor element.
   * @param batch_size The number of batches for each tensor element.
   */
  explicit TensorTuple(const DimensionsVectorType& dimensions, size_t time_size, size_t batch_size):
    tensors_(dimensions.size()),
    datumDimensions_(dimensions),
    scope_(),
    timesteps_(time_size),
    batches_(batch_size)
  {
    NFATAL_IF(time_size*batch_size==0 && time_size+batch_size!=0,
      "[" << scope() << "]: 'time_size' and 'batch_size' must be either both zero or non-zero!");
    configureTensors();
  }

  /*!
   * @brief Constructs empty tuple with configured scope and extended dimensions.
   * @param scope The scope within which the tuple instance is defined.
   * @param time_size The number of time-steps for each tensor element.
   * @param batch_size The number of batches for each tensor element.
   */
  explicit TensorTuple(std::string scope, size_t time_size, size_t batch_size):
    tensors_(),
    datumDimensions_(),
    scope_(std::move(scope)),
    timesteps_(time_size),
    batches_(batch_size)
  {
    NFATAL_IF(time_size*batch_size==0 && time_size+batch_size!=0,
      "[" << this->scope() << "]: 'time_size' and 'batch_size' must be either both zero or non-zero!");
  }

  /*!
   * @brief Constructs empty tuple but with a configured scope.
   * @param scope The scope within which the tuple instance is defined.
   */
  explicit TensorTuple(std::string scope):
    tensors_(),
    datumDimensions_(),
    scope_(std::move(scope))
  {
  }
  
  /*!
   * @brief Default constructor - the tuple is empty and is created with a scope and batching disabled.
   */
  TensorTuple() = default;
  
  /*!
   * @brief Default destructor.
   */
  virtual ~TensorTuple() = default;
  
  /*
   * Configurations
   */
  
  /*!
   * @brief Configures a tensor-tuple using element names and dimensions from an TensorsSpec object.
   * @note Any existing name-scope as well as additional time and batching dimensions, are preserved in the new configuration.
   * @param specs The TensorsSpec object containing name-dimensions pairs.
   * TODO: Rename to specify()
   */
  void setFromSpec(const TensorsSpec& specs) {
    datumDimensions_.clear();
    datumDimensions_.reserve(specs.size());
    tensors_.clear();
    tensors_.reserve(specs.size());
    for (auto& spec: specs) {
      addTensor(spec.first, spec.second);
    }
  }
  
  /*!
   * @brief Adds an new element to the tensor-tuple, configured explicitly via name and dimensions.
   * @param name The name of the new tensor element to be added.
   * @note The name must not be name-scoped.
   * @param dimensions A vector of `size_t` elements defining the dimensions of a primitive datum.
   * @warning Time and batching dimensions are set using any existing in the tuple.
   */
  void addTensor(const std::string& name, const DimensionsType& dimensions) {
    datumDimensions_.push_back(dimensions);
    auto dims = dimensions;
    if (isBatched()) {
      // Add as container extend with both time and batch dimensions
      dims.insert(dims.end(), {timesteps_, batches_});
      tensors_.emplace_back(utils::make_namescope({scope_, name}), dims, true);
    } else {
      /// Add as primitive container
      tensors_.emplace_back(utils::make_namescope({scope_, name}), dims, false);
    }
  }
  
  void setNames(const std::vector<std::string>& names) {
    NFATAL_IF(names.size() != tensors_.size(),
      "[" << scope() << "]: 'names' argument must contain the same number of elements as the tensor tuple!");
    for (size_t k=0; k<tensors_.size(); k++) {
      setName(k, names[k]);
    }
  }
  
  void setName(IndexType index, const std::string& name) {
    NFATAL_IF(index >= tensors_.size(), "[" << scope() << "]: 'index' argument must not exceed the number of tensors!");
    tensors_[index].setName(utils::make_namescope({scope_, name}));
  }
  
  void setScope(const std::string& scope) {
    scope_ = scope;
    if (!tensors_.empty()) {
      for (auto& tensor: tensors_) {
        tensor.setName(utils::make_namescope({scope_, utils::remove_namescope(tensor.name())}));
      }
    }
  }
  
  /*
   * Properties
   */
  
  /*!
   * @brief Retrieves the configuration of the tensor-tuple as a structured TensorsSpec object.
   * @note The tensor-tuple configuration includes the number of tensors, and the names and dimensions of each.
   * @warning The retrieved names do not retain the owning scope, but only the instance names.
   * @warning The retrieved dimensions do not retain time and batching dimensions, but only the dimensions of the data primitives.
   * @return
   */
  TensorsSpec spec() const {
    TensorsSpec spec;
    for (size_t k = 0; k < tensors_.size(); ++k) {
      spec.emplace_back(std::make_pair(utils::remove_namescope(tensors_[k].name()), datumDimensions_[k]));
    }
    return spec;
  }
  
  /*!
   * @brief Retrieves the size of the tensor tuple, i.e. the number of tensor elements contained.
   * @return The number of tensors contained by the tuple instance.
   */
  size_t size() const {
    return tensors_.size();
  }
  
  bool empty() const {
    return tensors_.empty();
  }
  
  std::vector<std::string> names() const {
    std::vector<std::string> names;
    for (const auto& tensor: tensors_) {
      names.emplace_back(tensor.name());
    }
    return names;
  }
  
  const std::string& name(IndexType index) const {
    NFATAL_IF(index >= tensors_.size(), "[" << scope() << "]: 'index' argument must not exceed the number of tensors!");
    return tensors_[index].name();
  }
  
  const std::string& scope() const {
    return scope_;
  }
  
  DimensionsVectorType dimensions() const {
    DimensionsVectorType dims;
    for (auto& tensor: tensors_) {
      dims.push_back(tensor.dimensions());
    }
    return dims;
  }
  
  DimensionsType dimensions(IndexType index) const {
    NFATAL_IF(index >= tensors_.size(), "[" << scope() << "]: 'index' argument must not exceed the number of tensors!");
    DimensionsType dims = tensors_[index].dimensions();
    return dims;
  }
  
  const DimensionsVectorType& datumDimensions() const {
    return datumDimensions_;
  }
  
  const DimensionsType& datumDimensions(IndexType index) const {
    NFATAL_IF(index >= tensors_.size(), "[" << scope() << "]: 'index' argument must not exceed the number of tensors!");
    return datumDimensions_[index];
  }
  
  size_t datum_size() const {
    size_t size = 0;
    for (auto& dims: datumDimensions_) { size += std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>()); }
    return size;
  }
  
  size_t timesteps() const {
    return timesteps_;
  }
  
  size_t batches() const {
    return batches_;
  }
  
  bool isBatched() const {
    return (timesteps_ > 0 && batches_ > 0);
  }
  
  auto capacities() const {
    return DimensionsType({timesteps_, batches_});
  }
  
  /*
   * Access operators
   */

  void set(const TensorVectorType& tensors) {
    tensors_ = tensors;
  }
  
  const TensorVectorType& get() const {
    return tensors_;
  }
  
  TensorVectorType& get() {
    return tensors_;
  }
  
  const TensorType& operator[](IndexType index) const {
    return tensors_[index];
  }
  
  TensorType& operator[](IndexType index) {
    return tensors_[index];
  }
  
  /*
   * Allocation & memory operations
   */

  /*!
   * @brief Resets all tensors as empty.
   * @warning This operations releases all resources.
   */
  void reset() {
    for (auto& tensor: tensors_) {
      tensor.reset();
    }
  }
  
  /*!
   * @brief Clears all internal counters and sets the tensors as data-less.
   * @note This retains the memory allocation, so data can be added via pushBack() operations.
   */
  void clear() {
    for (auto& tensor: tensors_) {
      tensor.clear();
    }
  }
  
  /*!
   * @brief Sets all sizes to equal the respective capacities.
   * @note This operation only affects the internal occupancy counters and does not affect the data.
   */
  void fill() {
    for (auto& tensor: tensors_) {
      tensor.fill();
    }
  }
  
  /*!
   * @brief Re-sizes the tensor-tuple from scratch, using explicit datum, time and batching dimensions.
   * @param dimensions Vectors of `size_t` elements defining the dimensions of each tensor's primitive datum.
   * @param time_size The size of the time dimension to apply to all tensor elements.
   * @param batch_size The size of the batching dimension to apply to all tensor elements.
   */
  void resize(const DimensionsVectorType& dimensions, size_t time_size, size_t batch_size) {
    if (datumDimensions_ != dimensions) {
      datumDimensions_ = dimensions;
      tensors_.resize(datumDimensions_.size());
    }
    timesteps_ = time_size;
    batches_ = batch_size;
    configureTensors();
  }


  /*!
   * @brief Re-sizes the tensor-tuple's elements, but only regarding the data dimensions.
   * @note Any existing time and batching dimensions are preserved.
   * @param dimensions Vectors of `size_t` elements defining the dimensions of each tensor's primitive datum.
   */
  void resize(const DimensionsVectorType& dimensions) {
    if (datumDimensions_ != dimensions) {
      datumDimensions_ = dimensions;
      tensors_.resize(datumDimensions_.size());
    }
    configureTensors();
  }

  /*!
   * @brief Re-sizes the tensor-tuple in terms of the time and batch dimensions.
   * @param time_size The size of the time dimension to apply to all tensor elements.
   * @param batch_size The size of the batching dimension to apply to all tensor elements.
   */
  void resize(size_t time_size, size_t batch_size) {
    timesteps_ = time_size;
    batches_ = batch_size;
    configureTensors();
  }

  /*!
   * @brief Re-sizes the tensor-tuple in terms of batch dimension only.
   * @param batch_size The size of the batching dimension to apply to all tensor elements.
   */
  void resize(size_t batch_size) {
    batches_ = batch_size;
    configureTensors();
  }

  /*!
   * @brief Allocates memory for the specified size in terms of the time and batch dimensions.
   * @note Datum dimensions and existing tensor names are preserved.
   * @param time_size The size of the time dimension to apply to all tensor elements.
   * @param batch_size The size of the batching dimension to apply to all tensor elements.
   */
  void reserve(size_t time_size, size_t batch_size) {
    NFATAL_IF(tensors_.empty(), "[" << scope() << "]: Cannot reserve empty tuple. Elements must be defined (names, dimensions).");
    for (auto& tensor: tensors_) { tensor.reserve(time_size, batch_size); }
    timesteps_ = time_size;
    batches_ = batch_size;
  }

  /*!
   * @brief Allocates memory for the specified batch size.
   * @note Datum dimensions and existing tensor names are preserved.
   * @param time_size The size of the time dimension to apply to all tensor elements.
   * @param batch_size The size of the batching dimension to apply to all tensor elements.
   */
  void reserve(size_t batch_size) {
    NFATAL_IF(tensors_.empty(), "[" << scope() << "]: Cannot reserve for empty tuple. Elements must be defined (names, dimensions).");
    for (auto& tensor: tensors_) { tensor.reserve(batch_size); }
    batches_ = batch_size;
  }
  
  /*!
  * @brief Reduces the memory allocation of all tensor elements to match the current batch size.
  */
  void shrink() {
    for (auto& tensor: tensors_) { tensor.shrinkBatches(); }
    if (!tensors_.empty()) { batches_ = tensors_.front().batches(); }
  }
  
  /*!
   * @brief Performs a re-shaping of the dimensions without modifying the underling storage, and sets explcit time-steps for each batch.
   * @param time_size The new maximum time-step capacity.
   * @param batch_size The new batch occupancy.
   * @param timesteps A vector of time-step occupancies corresponding to the batches.
   */
  void reshape(size_t time_size, size_t batch_size, const std::vector<size_t>& timesteps) {
    for (auto& tensor: tensors_) { tensor.reshapeBatches(time_size, batch_size, timesteps); }
    timesteps_ = time_size;
    batches_ = batch_size;
  }

  /*!
   * @brief Performs a re-shaping of the dimensions without modifying the underling storage, and sets max time-steps for each batch.
   * @param time_size The new maximum time-step capacity.
   * @param batch_size The new batch occupancy.
   */
  void reshape(size_t time_size, size_t batch_size) {
    for (auto& tensor: tensors_) { tensor.reshapeBatches(time_size, batch_size); }
    timesteps_ = time_size;
    batches_ = batch_size;
  }
  
  /*
   *  Clone & copy operations
   */
  
  TensorTuple& clone(const TensorTuple& rhs) {
    NFATAL_IF(rhs.empty(), "[" << scope() << "]: Cannot clone from 'rhs' argument which is empty!");
    datumDimensions_ = rhs.datumDimensions_;
    batches_ = rhs.batches_;
    timesteps_ = rhs.timesteps_;
    tensors_.resize(rhs.size());
    for (size_t k=0; k<tensors_.size(); k++) { tensors_[k].clone(rhs[k]); }
    return *this;
  }
  
  TensorTuple& copy(const TensorTuple& rhs) {
    NFATAL_IF(rhs.empty(), "[" << scope() << "]: Cannot copy from 'rhs' argument which is empty!");
    datumDimensions_ = rhs.datumDimensions_;
    timesteps_ = rhs.timesteps_;
    batches_ = rhs.batches_;
    tensors_.resize(rhs.size());
    for (size_t k=0; k<tensors_.size(); k++) { tensors_[k].copy(rhs[k]); }
    return *this;
  }
  
  TensorTuple& copy(const TensorVectorType& rhs) {
    NFATAL_IF(rhs.empty(), "[" << scope() << "]: Cannot copy from 'rhs' argument which is empty!");
    timesteps_ = rhs[0].maxTimeSteps();
    batches_ = rhs[0].batches();
    tensors_.resize(rhs.size());
    for (size_t k=0; k<tensors_.size(); k++) {
      tensors_[k].copy(rhs[k]);
      if (rhs[k].isBatched()) {
        datumDimensions_[k] = std::move(DimensionsType(rhs[k].dimensions().begin(), rhs[k].dimensions().end()-2));
      } else {
        datumDimensions_[k] = rhs[k].dimensions();
      }
    }
    return *this;
  }
  
  TensorTuple& mimic(const TensorTuple& rhs) {
    resize(rhs.datumDimensions_, rhs.timesteps_, rhs.batches_);
    scope_ = rhs.scope_;
    for (size_t k = 0; k < size(); ++k) { tensors_[k].setName(rhs.names()[k]); }
    return *this;
  }
  
  /*
   * Assignment operators
   */

  TensorTuple& operator=(const TensorTuple& rhs) {
    NFATAL_IF((rhs.size() != tensors_.size()), "[" << scope() << "]: 'rhs' does not contain the same number of tensors!");
    timesteps_ = rhs.timesteps_;
    batches_ = rhs.batches_;
    for (size_t k=0; k<tensors_.size(); k++) { tensors_[k] = rhs[k]; }
    return *this;
  }

  TensorTuple& operator=(const TensorVectorType& rhs) {
    NFATAL_IF((rhs.size() != tensors_.size()), "[" << scope() << "]: 'rhs' does not contain the same number of tensors!");
    for (size_t k=0; k<tensors_.size(); k++) { tensors_[k] = rhs[k]; }
    timesteps_ = rhs[0].maxTimeSteps();
    batches_ = rhs[0].batches();
    return *this;
  }

  /*
   * Value setting operators
   */

  void setConstant(ScalarType value) {
    for (auto& tensor: tensors_) { tensor.setConstant(value); }
  }
  
  void setConstant(IndexType index, ScalarType value) {
    tensors_[index].setConstant(value);
  }
  
  void setZero() {
    for (auto& tensor: tensors_) { tensor.setZero(); }
  }

  void setZero(IndexType index) {
    tensors_[index].setZero();
  }
  
  void setRandom() {
    for (auto& tensor: tensors_) { tensor.setRandom(); }
  }
  
  void setRandomUnitUniform(math::RandomNumberGenerator<ScalarType>& generator) {
    for (auto& tensor: tensors_) { tensor.setRandomUnitUniform(generator); }
  }
  
  void setRandomStandardNormal(math::RandomNumberGenerator<ScalarType>& generator) {
    for (auto& tensor: tensors_) { tensor.setRandomStandardNormal(generator); }
  }
  
  void shuffle(const Eigen::VectorXi& indices) {
    for (auto& tensor: tensors_) { tensor.shuffle(indices); }
  }
  
  /*
   * Validity checks
   */
  
  bool hasNaN() const {
    bool result = false;
    for (const auto& tensor: tensors_) { result |= tensor.hasNaN(); }
    return result;
  }

  bool hasInf() const {
    bool result = false;
    for (const auto& tensor: tensors_) { result |= tensor.hasInf(); }
    return result;
  }
  
  bool allFinite() const {
    bool result = false;
    for (const auto& tensor: tensors_) { result |= tensor.allFinite(); }
    return result;
  }
  
  /*
   * Helper functions
   */

  bool hasSameStorageWith(const TensorTuple& tuple) {
    bool result = true;
    for (size_t k = 0; k < tensors_.size(); ++k) { result &= tensors_[k].hasSameStorageWith(tuple[k]); }
    return result;
  }
  
  friend inline std::ostream& operator<<(std::ostream& os, const TensorTuple& tuple) {
    os << "\n[noesis::TensorTuple]";
    for (size_t k = 0; k < tuple.size(); ++k) { os << "\n[" << std::to_string(k) << "]" << tuple[k]; }
    return os;
  }

private:

  /*!
   * @brief Internal helper function which performs tensor element allocations based on the current dimensional configurations.
   */
  void configureTensors() {
    NFATAL_IF(tensors_.empty(), "[" << scope() << "]: Cannot resize empty tuple. Elements must be defined (names, dimensions).");
    for (size_t k = 0; k < datumDimensions_.size(); k++) {
      auto dims = datumDimensions_[k];
      if (isBatched()) {
        // Extend with both time and batch dimensions
        dims.insert(dims.end(), {timesteps_, batches_});
        tensors_[k].resize(dims, true);
      } else {
        // Initialize as primitive container
        tensors_[k].resize(dims, false);
      }
    }
  }

private:
  //! @brief The underlying STD vector of core::Tensors objects.
  TensorVectorType tensors_;
  //! @brief The dimensions of the basic datum each tensor holds (i.e. without trajectory and batching).
  DimensionsVectorType datumDimensions_;
  //! @brief The (parent) name-scope to which the named tensors belong.
  std::string scope_;
  //! @brief The number of trajectory time-steps the tuples extend to.
  size_t timesteps_ = 0;
  //! @brief The number of batch instances the tuples extend to.
  size_t batches_ = 0;
};

} // namespace noesis

#endif // NOESIS_FRAMEWORK_CORE_TENSOR_TUPLE_HPP_

/* EOF */
