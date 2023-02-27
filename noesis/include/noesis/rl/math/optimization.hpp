/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_RL_MATH_OPTIMIZATION_HPP_
#define NOESIS_RL_MATH_OPTIMIZATION_HPP_

// Eigen
#include <Eigen/Dense>

// Noesis
#include "noesis/framework/log/message.hpp"

namespace noesis {
namespace math {

/*!
 * @brief Implements Backtracking Line Search (BLS) as described on page. 37 of Nocedal and Wright [1].
 *
 * @note This variant uses a constant contraction factor.
 * @note This variant uses a fixed maximum vector and does not recompute the gradient of the cost at each iteration.
 *
 * [1] Jorge Nocedal and Stephen Wright,
 *     "Numerical optimization",
 *     Springer Science & Business Media, 2006.
 *
 * @tparam ScalarType_
 * @param f A standard C++ functional or lambda providing evaluations of the objective function f(x).
 * @param x A dynamic Eigen vector containing the initial value of the decision variable x.
 * @param dx A dynamic Eigen vector containing the initial (largest) step direction dx (synonymous with Δx := alpha * p_0).
 * @param rho A constant contraction factor used to decay the step-length.
 * @param k_max The maximum number of iterations permitted for the line-search operation.
 * @param verbose Set to true to enable verbose output.
 * @return Returns A pair containing: 1) the norm of the optimizing step vector, and 2) the optimizing iteration index.
 */
template<typename ScalarType_>
static inline std::pair<ScalarType_, int> backtracking_line_search(
    const std::function<ScalarType_(const Eigen::Matrix<ScalarType_, Eigen::Dynamic, 1>&)>& f,
    const Eigen::Matrix<ScalarType_, Eigen::Dynamic, 1>& x,
    Eigen::Matrix<ScalarType_, Eigen::Dynamic, 1>& dx,
    ScalarType_ rho = 1.0,
    int k_max = 1000,
    bool verbose = false) {
  Eigen::Matrix<ScalarType_, Eigen::Dynamic, 1> x_k = x + dx;
  Eigen::Matrix<ScalarType_, Eigen::Dynamic, 1> dx_k = dx;
  auto L_k = f(x_k);
  NINFO_IF(verbose, "Line Search: L_0: " << L_k);
  auto L_star = L_k;
  int k_star = 0;
  for (int k = 1; k < k_max; ++k) {
    dx_k *= rho;
    x_k = x + dx_k;
    L_k = f(x_k);
    NINFO_IF(verbose, "Line Search: L_" << k << ": " << L_k);
    if (L_k < L_star) {
      dx = dx_k;
      L_star = L_k;
      k_star = k;
    }
  }
  auto dx_star_norm = dx.norm();
  NINFO_IF(verbose, "Line search: Best index: " << k_star);
  NINFO_IF(verbose, "Line search: Best loss: " << L_star);
  NINFO_IF(verbose, "Line search: Best step norm: " << dx_star_norm);
  return {dx_star_norm, k_star};
}

} // namespace math
} // namespace noesis

#endif // NOESIS_RL_MATH_OPTIMIZATION_HPP_

/* EOF*/
