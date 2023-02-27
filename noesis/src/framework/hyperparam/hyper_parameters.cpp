/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Noesis
#include <noesis/framework/hyperparam/hyper_parameters.hpp>

namespace noesis {
namespace hyperparam {

/*!
 * @brief The instantiation of the global hyper-parameter manager.
 */
std::shared_ptr<noesis::hyperparam::HyperParameterManager> manager = std::make_shared<noesis::hyperparam::HyperParameterManager>();

} // namespace hyperparam
} // namespace noesis

/* EOF */
