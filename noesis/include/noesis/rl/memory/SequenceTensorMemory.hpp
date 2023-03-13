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
#ifndef NOESIS_RL_MEMORY_SEQUENCE_TENSOR_MEMORY_HPP_
#define NOESIS_RL_MEMORY_SEQUENCE_TENSOR_MEMORY_HPP_

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/framework/core/TensorTuple.hpp"
#include "noesis/framework/hyperparam/hyper_parameters.hpp"

namespace noesis {
namespace memory {

struct SequenceTensorMemoryConfig {
  noesis::TensorsSpec tensors_spec;
  std::string name{"memory"};
  std::string scope{""};
  size_t number_of_instances{1u};
  bool verbose{false};
};

template<typename ScalarType_>
class SequenceTensorMemory final: public noesis::core::Object
{
public:
  // Aliases
  using ScalarType = ScalarType_;
  using MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
  using TensorTupleType = noesis::TensorTuple<ScalarType>;
  
  explicit SequenceTensorMemory(const SequenceTensorMemoryConfig& config):
    noesis::core::Object(config.name, config.scope, config.verbose),
    sequences_(),
    buffers_(config.number_of_instances),
    tensorsSpec_(config.tensors_spec),
    sequenceLength_(1, noesis::utils::make_namescope({config.scope, config.name, "length"}), {1, 100}),
    sequenceStride_(1, noesis::utils::make_namescope({config.scope, config.name, "stride"}), {1, 100}),
    numberOfInstances_(config.number_of_instances)
  {
    // Add hyper-parameters to the global manager
    noesis::hyperparam::manager->addParameter(sequenceLength_);
    noesis::hyperparam::manager->addParameter(sequenceStride_);
  }
  
  ~SequenceTensorMemory() final {
    // Remove hyper-parameters to the global manager
    noesis::hyperparam::manager->removeParameter(sequenceLength_);
    noesis::hyperparam::manager->removeParameter(sequenceStride_);
  }
  
  /*
   * Configurations
   */
  
  void setTensorsSpecifications(const noesis::TensorsSpec& spec) {
    tensorsSpec_ = spec;
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
  
  const noesis::TensorsSpec& getTensorsSpecifications() const {
    return tensorsSpec_;
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
    NFATAL_IF(tensorsSpec_.empty(), "[" << name << "]: Tensors specifications are undefined!");
    NFATAL_IF(sequenceLength_ == 0, "[" << name << "]: Sequence length must be great than zero!");
    NFATAL_IF(sequenceSize_ == 0, "[" << name << "]: Sequence stride must be great than zero!");
    // Compute the total size of the history
    sequenceSize_ = static_cast<size_t>((sequenceLength_ - 1) * sequenceStride_ + 1);
    // Configure the history container
    sequences_ = TensorTupleType("sequence", 1, 1);
    for (auto spec: tensorsSpec_) {
      sequences_.addTensor(spec.first, spec.second);
    }
    // Configure buffers
    for (auto& buf: buffers_) {
      buf.resize(tensorsSpec_.size());
    }
    // Set the time-steps and instances of the history
    sequences_.resize(sequenceSize_, numberOfInstances_);
    // Reset all values to zero by default
    reset();
    // Diagnostic output
    NINFO("[" << name << "]: Tensors specifications: " << tensorsSpec_);
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
  
  void insert(size_t instance, const TensorTupleType& inputs) {
    NFATAL_IF(instance >= numberOfInstances_,
      "[" << namescope() << "]: 'instance' (" << instance
      << ") must be less than the number of instances (" << numberOfInstances_ << ")!");
    NFATAL_IF(inputs.size() != sequences_.size(),
      "[" << namescope() << "]: 'inputs' does not have the same size as the histories: "
      << inputs.size() << " vs. " << sequences_.size());
    // Shift all current elements by one time-step and add the new values at the front
    for (size_t k = 0; k < inputs.size(); ++k) {
      buffers_[instance][k] = sequences_[k].block(instance*sequenceSize_, sequenceSize_-1).asEigenMatrix();
      sequences_[k].block(instance*sequenceSize_+1, sequenceSize_-1).asEigenMatrix() = buffers_[instance][k];
      sequences_[k].datum(instance*sequenceSize_) = inputs[k](0, instance);
    }
  }
  
  void extract(size_t instance, TensorTupleType& outputs) {
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
          outputs[k](t, instance) = sequences_[k](t * stride, instance);
        }
      } else {
        outputs[k](instance) = sequences_[k](instance);
      }
    }
  }
  
  /*
   * Helper functions
   */
  
  friend inline std::ostream& operator<<(std::ostream& os, const SequenceTensorMemory& rhs) {
    os << "[" << rhs.namescope() << "]:";
    os << "\n[Configurations]:";
    os << "\n  Tensors specification: " << rhs.tensorsSpec_;
    os << "\n  Sequence length: " << static_cast<int>(rhs.sequenceLength_);
    os << "\n  Sequence stride: " << static_cast<int>(rhs.sequenceStride_);
    os << "\n\n[Sequences]:" << rhs.sequences_;
    return os;
  }

private:
  TensorTupleType sequences_;
  std::vector<std::vector<MatrixType>> buffers_;
  noesis::TensorsSpec tensorsSpec_;
  noesis::hyperparam::HyperParameter<int> sequenceLength_;
  noesis::hyperparam::HyperParameter<int> sequenceStride_;
  size_t sequenceSize_{1u};
  const size_t numberOfInstances_;
};

} // namespace memory
} // namespace noesis

#endif // NOESIS_RL_MEMORY_SEQUENCE_TENSOR_MEMORY_HPP_

/* EOF */
