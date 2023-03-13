/*!
 * @author    Mayank Mittal
 * @email     mittalma@student.ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_UTILS_IMAGE_ENCODING_HPP_
#define NOESIS_FRAMEWORK_UTILS_IMAGE_ENCODING_HPP_

// C/C++
#include <string>

// Eigen
#include <Eigen/Dense>

namespace noesis {
namespace utils {

//! @brief Convert eigen matrices for RGBA channels into png string
extern int png_image_to_string(const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& red,
                               const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& green,
                               const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& blue,
                               const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& alpha,
                               std::string& output);

//! @brief Convert eigen matrices for RGB channels into png string
extern int png_image_to_string(const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& red,
                               const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& green,
                               const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& blue,
                               std::string& output);

//! @brief Convert eigen matrices for grey-scale image into png string
extern int png_image_to_string(const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& grey,
                               std::string& output);

} // namespace utils
} // namespace noesis

#endif // NOESIS_FRAMEWORK_UTILS_IMAGE_ENCODING_HPP_

/* EOF */
