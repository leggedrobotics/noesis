/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_MDP_TYPES_HPP_
#define NOESIS_MDP_TYPES_HPP_

// Noesis
#include <noesis/framework/core/TensorTuple.hpp>
#include <noesis/framework/log/metric.hpp>

namespace noesis {
namespace mdp {

/*
 *  Define Agent-Environment Interface Types
 */

//! @brief Observations are generally defined as tuples of Tensors and
//! can thus represent heterogeneous collections of data.
template<typename ScalarType_>
using Observations = ::noesis::TensorTuple<ScalarType_>;

//! @brief Actions are generally defined as tuples of Tensors and
//! can thus represent heterogeneous collections of data.
template<typename ScalarType_>
using Actions = ::noesis::TensorTuple<ScalarType_>;

//! @brief Rewards are generally defined as tensors and can thus be
//! represented as vectors or matrices for multi-task problems.
template<typename ScalarType_>
using Rewards = ::noesis::Tensor<ScalarType_>;

//! @brief Metrics are stored as collections of name-value pairs with
//! the underlying numeric values stored as Eigen metrics enabling
//! efficient arithmetic operations to be performed on such collections.
template<typename ScalarType_>
using Metrics = ::noesis::log::Metrics<ScalarType_>;

/*!
 * @brief An MDP termination event.
 *
 * @note This struct realizes state absorption in MDPs [1], whereby, termination events represent
 *       transitions to states in which actions have no effect. In such cases, the tail of the
 *       return is therefore a constant (since state transitions cannot occur), and is thus
 *       represented by the `Termination::value` member variable.
 *
 *      Given reward samples r_t for each point in time t, the total undiscounted cumulative reward,
 *      i.e. return R_t at time t is:
 *
 *      R_t = sum[from k=t, to k=inf](r_k)
 *
 *      If a termination event occurs at time T, then state absorption corresponds to:
 *
 *      R_t = sum[from k=t, to k=T](r_k) + v_T
 *
 *      where v_T corresponds to the tail of the return from t=T+1 to t=inf, and is thus
 *      represented by the `Termination::value` member of the termination event.
 *
 * [1] Sutton, Richard S., Joseph Modayil, Michael Delp Thomas Degris, Patrick M. Pilarski, and Adam White.
 *     "Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction."
 *     Proceedings of the 10th International Conference on Autonomous Agents and Multiagent Systems, 2011.
 *
 * @tparam ScalarType_ The scalar type used for the terminal value of the termination event.
 */
template<typename ScalarType_>
struct Termination
{
  //! @brief Define an enum of the supported termination types.
  enum class Type: int {
    Unterminated = 0,
    TerminalState = 1,
    TimeOut = 2,
    InvalidState = -1,
    InvalidAction = -2,
    InvalidReward = -3,
    InvalidObservation = -4
  };
  //! @brief The terminal value assigned to the corresponding termination event.
  ScalarType_ value{0};
  //! @brief The termination type of the corresponding termination event.
  Type type{Type::Unterminated};
  //! @brief An (optional) environment-specific termination identifier for the corresponding termination event.
  int id{0};
  /*!
   * @brief Converts the termination event to a string.
   * @note The output format is <TYPE>{<VALUE>}
   * @return The string containing the converted termination type.
   */
  inline std::string as_string() const {
    std::string out;
    switch (type) {
      case Type::Unterminated:
        out = "Unterminated";
        break;
      case Type::TerminalState:
        out = "TerminalState";
        break;
      case Type::TimeOut:
        out = "TimeOut";
        break;
      case Type::InvalidState:
        out = "InvalidState";
        break;
      case Type::InvalidAction:
        out = "InvalidAction";
        break;
      case Type::InvalidReward:
        out = "InvalidReward";
        break;
      case Type::InvalidObservation:
        out = "InvalidObservation";
        break;
    }
    out += "{value=" + std::to_string(value) + ",id="+ std::to_string(id) + "}";
    return out;
  }
  /*!
   * @brief Helper operator for streaming the termination event to output streams such as std::cout.
   * @param os The target output stream to which the termination event shall be written.
   * @param termination The termination event to be streamed.
   * @return The augmented output stream.
   */
  friend inline std::ostream& operator<<(std::ostream &os, const Termination& termination) {
    return os << termination.as_string();
  }
};

//! Alias for vectorized termination status
template<typename ScalarType_>
using Terminations = std::vector<Termination<ScalarType_>>;

} // namespace mdp
} // namespace noesis

#endif // NOESIS_MDP_TYPES_HPP_

/* EOF */
