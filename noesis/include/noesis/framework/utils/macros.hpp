/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_UTILS_MACROS_HPP_
#define NOESIS_FRAMEWORK_UTILS_MACROS_HPP_

//! @brief Helper macro which suppresses unused variable warnings in stub functions.
#define UNUSED_VARIABLE(var) do { (void)(var); } while (0)

//! @brief Helper macros for converting macros to strings.
#define MAKE_STRING(x) _MAKE_STRING(x)
#define _MAKE_STRING(x) #x

#endif // NOESIS_FRAMEWORK_UTILS_MACROS_HPP_

/* EOF */
