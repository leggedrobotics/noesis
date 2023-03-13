/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    David Hoeller
 * @email     dhoeller@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_MEMORY_SEQUENCE_VECTOR_MEMORY_HPP_
#define NOESIS_RL_MEMORY_SEQUENCE_VECTOR_MEMORY_HPP_

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/framework/core/TensorTuple.hpp"
#include "noesis/framework/hyperparam/hyper_parameters.hpp"

namespace noesis {
namespace memory {

struct SequenceVectorMemoryConfig {
  std::vector<size_t> vector_dims;
  std::string name{"memory"};
  std::string scope{""};
  size_t number_of_instances{1u};
  bool verbose{false};
};

template<typename ScalarType_>
class SequenceVectorMemory final: public noesis::core::Object
{
public:
  // Aliases
  using ScalarType = ScalarType_;
  using MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
  using TensorTupleType = noesis::TensorTuple<ScalarType>;
  
  explicit SequenceVectorMemory(const SequenceVectorMemoryConfig& config):
    noesis::core::Object(config.name, config.scope, config.verbose),
    sequences_(),
    buffers_(config.number_of_instances),
    vectorDimensions_(config.vector_dims),
    sequenceLength_(1, noesis::utils::make_namescope({config.scope, config.name, "length"}), {1, 100}),
    sequenceStride_(1, noesis::utils::make_namescope({config.scope, config.name, "stride"}), {1, 100}),
    numberOfInstances_(config.number_of_instances)
  {
    // Add hyper-parameters to the global manager
    noesis::hyperparam::manager->addParameter(sequenceLength_);
    noesis::hyperparam::manager->addParameter(sequenceStride_);
  }
  
  ~SequenceVectorMemory() final {
    // Remove hyper-parameters to the global manager
    noesis::hyperparam::manager->removeParameter(sequenceLength_);
    noesis::hyperparam::manager->removeParameter(sequenceStride_);
  }
  
  /*
   * Configurations
   */
  
  void setVectorDimensions(std::vector<size_t> dimensions) {
    vectorDimensions_ = dimensions;
  }
  
  void setSequenceLength(size_t length) {
    sequenceLength_ = length;
  }
  
  void setSequenceStride(size_t stride) {
    sequenceStride_ = stride;
  }
  
  /*
   * Properties
   */
  
  const std::vector<size_t>& getVectorDimensions() const {
    return vectorDimensions_;
  }
  
  size_t getSequenceLength() const {
    return static_cast<size_t>(sequenceLength_);
  }
  
  size_t getSequenceStride() const {
    return static_cast<size_t>(sequenceStride_);
  }
  
  size_t getSequenceSize() const {
    return sequenceSize_;
  }
  
  size_t getNumberOfInstances() const {
    return numberOfInstances_;
  }
  
  const TensorTupleType& getSequences() const {
    return sequences_;
  }
  
  TensorTupleType& getSequences() {
    return sequences_;
  }
  
  /*
   * Operations
   */
  
  void configure() {
    auto name = namescope();
    NFATAL_IF(vectorDimensions_.empty(), "[" << name << "]: Vector dimensions are undefined!");
    NFATAL_IF(sequenceLength_ == 0, "[" << name << "]: Sequence length must be great than zero!");
    NFATAL_IF(sequenceSize_ == 0, "[" << name << "]: Sequence stride must be great than zero!");
    // Compute the total size of the history
    sequenceSize_ = static_cast<size_t>((sequenceLength_ - 1) * sequenceStride_ + 1);
    // Configure the history container
    sequences_ = TensorTupleType("sequence", 1, 1);
    for (size_t k = 0; k < vectorDimensions_.size(); ++k) {
      sequences_.addTensor(std::to_string(k), {vectorDimensions_[k]});
    }
    // Configure buffers
    for (auto& buf: buffers_) {
      buf.resize(vectorDimensions_.size());
    }
    // Set the time-steps and instances of the history
    sequences_.resize(sequenceSize_, numberOfInstances_);
    // Reset all values to zero by default
    reset();
    // Diagnostic output
    NINFO("[" << name << "]: Vectors dimensions: " << utils::vector_to_string(vectorDimensions_));
    NINFO("[" << name << "]: Sequence length: " << static_cast<int>(sequenceLength_));
    NINFO("[" << name << "]: Sequence stride: " << static_cast<int>(sequenceStride_));
    NINFO("[" << name << "]: Sequence size: " << static_cast<int>(sequenceSize_));
    NINFO_IF(isVerbose(), "[" << name << "]: Sequences: " << sequences_);
  }
  
  void reset() {
    for (auto& seq: sequences_.get()) {
      seq.setZero();
    }
  }
  
  void reset(size_t instance) {
    for (auto& seq: sequences_.get()) {
      seq(instance).setZero();
    }
  }
  
  void insert(size_t instance, const std::vector<MatrixType>& inputs) {
    NFATAL_IF(instance >= numberOfInstances_,
      "[" << namescope() << "]: 'instance' (" << instance
      << ") must be less than the number of instances (" << numberOfInstances_ << ")!");
    NFATAL_IF(inputs.size() != sequences_.size(),
      "[" << namescope() << "]: 'inputs' does not have the same size as the histories: "
      << inputs.size() << " vs. " << sequences_.size());
    // Shift all current elements by one time-step and add the new values at the front
    for (size_t k = 0; k < inputs.size(); ++k) {
      buffers_[instance][k] = sequences_[k](instance).asEigenMatrix().leftCols(sequenceSize_ - 1);
      sequences_[k](instance).asEigenMatrix().rightCols(sequenceSize_ - 1) = buffers_[instance][k];
      sequences_[k](instance).asEigenMatrix().col(0) = inputs[k];
    }
  }
  
  void extract(size_t instance, std::vector<MatrixType>& outputs) {
    NFATAL_IF(instance >= numberOfInstances_,
      "[" << namescope() << "]: 'instance' (" << instance
      << ") must be less than the number of instances (" << numberOfInstances_ << ")!");
    NFATAL_IF(outputs.size() != sequences_.size(),
      "[" << namescope() << "]: 'outputs' does not have the same size as the histories: "
      << outputs.size() << " vs. " << sequences_.size());
    // Extract decimated elements according to the stride
    auto stride = static_cast<size_t>(sequenceStride_);
    auto length = static_cast<size_t>(sequenceLength_);
    for (size_t k = 0; k < sequences_.size(); ++k) {
      if (stride > 1) {
        for (size_t t = 0; t < length; ++t) {
          outputs[k].col(t) = sequences_[k](instance).asEigenMatrix().col(t * stride);
        }
      } else {
        outputs[k] = sequences_[k](instance).asEigenMatrix();
      }
    }
  }
  
  /*
   * Helper functions
   */
  
  friend inline std::ostream& operator<<(std::ostream& os, const SequenceVectorMemory& rhs) {
    os << "[" << rhs.namescope() << "]:";
    os << "\n[Configurations]:";
    os << "\n  Dimensions of vectors: " << utils::vector_to_string(rhs.vectorDimensions_);
    os << "\n  Sequence length: " << static_cast<int>(rhs.sequenceLength_);
    os << "\n  Sequence stride: " << static_cast<int>(rhs.sequenceStride_);
    os << "\n\n[Sequences]:" << rhs.sequences_;
    return os;
  }

private:
  TensorTupleType sequences_;
  std::vector<std::vector<MatrixType>> buffers_;
  std::vector<size_t> vectorDimensions_;
  noesis::hyperparam::HyperParameter<int> sequenceLength_;
  noesis::hyperparam::HyperParameter<int> sequenceStride_;
  size_t sequenceSize_{1u};
  const size_t numberOfInstances_;
};

} // namespace memory
} // namespace noesis

#endif // NOESIS_RL_MEMORY_SEQUENCE_VECTOR_MEMORY_HPP_

/* EOF */
