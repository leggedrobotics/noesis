/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_COMMON_SIGNALS_HPP_
#define NOESIS_GYM_ENVS_RAISIM_COMMON_SIGNALS_HPP_

// C/C++
#include <cmath>
#include <random>
#include <iostream>

namespace noesis {
namespace gym {
namespace signals {


/*!
 * @brief Scalar step signal.
 * @tparam Scalar Scalar floating-point type used for the operation.
 * @param t The current physical time.
 * @param t0 The starting time of the signal.
 * @return The value of the signal output.
 */
template <typename Scalar>
static inline Scalar step(Scalar t, Scalar t0=0) {
  return (t > t0) ? 1 : 0;
}

/*!
 * @brief Scalar sinusoidal signal
 * @tparam Scalar Scalar floating-point type used for the operation.
 * @param t The current physical time.
 * @param f The signal frequency.
 * @return The value of the signal output.
 */
template <typename Scalar>
static inline Scalar sine(Scalar t, Scalar f) {
  return std::sin(2.0 * M_PI * f * t);
}

/*!
 * @brief Scalar exponential chirp signal.
 * @tparam Scalar Scalar floating-point type used for the operation.
 * @param t The current physical time.
 * @param T The ending physical time of the signal.
 * @param f0 The initial frequency of the chirp.
 * @param f1 The final frequency of the chirp.
 * @return The value of the signal output.
 */
template <typename Scalar>
static inline Scalar chirp(Scalar t, Scalar T=1, Scalar f0=1, Scalar f1=10) {
  constexpr auto eps = std::numeric_limits<Scalar>::epsilon();
  auto k = std::pow(f1/f0, t/T);
  return std::sin(2.0 * M_PI * f0 * (k - 1.0)/(std::log(k) + eps));
}

} // namespace signals
} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_COMMON_SIGNALS_HPP_

/* EOF */
