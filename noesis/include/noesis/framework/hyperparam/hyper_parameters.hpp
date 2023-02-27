/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    David Hoeller
 * @email     dhoeller@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETERS_HPP_
#define NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETERS_HPP_

// Noesis
#include "noesis/framework/hyperparam/HyperParameter.hpp"
#include "noesis/framework/hyperparam/HyperParameterManager.hpp"

namespace noesis {
namespace hyperparam {

/*!
 * @brief Declaration of the global hyper-parameter manager instance.
 */
extern std::shared_ptr<noesis::hyperparam::HyperParameterManager> manager;

} // namespace hyperparam
} // namespace noesis

#endif // NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETERS_HPP_

/* EOF */
