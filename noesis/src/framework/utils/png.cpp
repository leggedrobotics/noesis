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

// Noesis
#include "noesis/framework/utils/png.hpp"

// C/C++
#include <vector>

// stb_image
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace noesis {
namespace utils {

//! @brief Convert eigen matrices for RGBA channels into png string
int png_image_to_string(const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& red,
                        const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& green,
                        const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& blue,
                        const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& alpha,
                        std::string& output) {
  // check all channels are of same height and width
  assert((red.rows() == green.rows()) && (green.rows() == blue.rows()) && (blue.rows() == alpha.rows()));
  assert((red.cols() == green.cols()) && (green.cols() == blue.cols()) && (blue.cols() == alpha.cols()));
  const int comp = 4;                                     // 4 channels: Red, Green, Blue, Alpha
  const int stride_in_bytes = red.rows() * comp;          // length of one row in bytes
  std::vector<unsigned char> data(red.size() * comp, 0);  // the image itself
  for (unsigned i = 0; i < red.rows(); ++i) {
    for (unsigned j = 0; j < red.cols(); ++j) {
      data[(j * red.rows() * comp) + (i * comp) + 0] = red(i, red.cols() - 1 - j);
      data[(j * red.rows() * comp) + (i * comp) + 1] = green(i, red.cols() - 1 - j);
      data[(j * red.rows() * comp) + (i * comp) + 2] = blue(i, red.cols() - 1 - j);
      data[(j * red.rows() * comp) + (i * comp) + 4] = alpha(i, red.cols() - 1 - j);
    }
  }
  // convert unsigned char vector into buffer
  int len;
  unsigned char* png_buffer_ptr = stbi_write_png_to_mem(data.data(), stride_in_bytes, red.rows(), green.cols(), comp, &len);
  // write blocks of data into encoded string
  if (png_buffer_ptr != nullptr) {
    std::ostringstream os;
    for (int i = 0; i < len; i++) {
      os << *(png_buffer_ptr + i);
    }
    output = os.str();
    // release buffer pointer
    free(png_buffer_ptr);
    return 0;
  } else {
    // release buffer pointer
    free(png_buffer_ptr);
    return 1;
  }
}

//! @brief Convert eigen matrices for RGB channels into png string
int png_image_to_string(const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& red,
                        const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& green,
                        const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& blue,
                        std::string& output) {
  // check all channels are of same height and width
  assert((red.rows() == green.rows()) && (green.rows() == blue.rows()));
  assert((red.cols() == green.cols()) && (green.cols() == blue.cols()));
  const int comp = 3;                                       // 3 channels: Red, Green, Blue
  const int stride_in_bytes = red.rows() * comp;            // length of one row in bytes
  std::vector<unsigned char> data(red.size() * comp, 0);    // the image itself
  for (unsigned i = 0; i < red.rows(); ++i) {
    for (unsigned j = 0; j < red.cols(); ++j) {
      data[(j * red.rows() * comp) + (i * comp) + 0] = red(i, red.cols() - 1 - j);
      data[(j * red.rows() * comp) + (i * comp) + 1] = green(i, red.cols() - 1 - j);
      data[(j * red.rows() * comp) + (i * comp) + 2] = blue(i, red.cols() - 1 - j);
    }
  }
  // convert unsigned char vector into buffer
  int len;
  unsigned char* png_buffer_ptr = stbi_write_png_to_mem(data.data(), stride_in_bytes, red.rows(), red.cols(), comp, &len);
  // write blocks of data into encoded string
  if (png_buffer_ptr != nullptr) {
    std::ostringstream os;
    for (int i = 0; i < len; i++) {
      os << *(png_buffer_ptr + i);
    }
    output = os.str();
    // release buffer pointer
    free(png_buffer_ptr);
    return 0;
  } else {
    // release buffer pointer
    free(png_buffer_ptr);
    return 1;
  }
}

//! @brief Convert eigen matrices for grey-scale image into png string
int png_image_to_string(const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& grey,
                        std::string& output) {
  const int comp = 1;                                       // 1 Channel: Grey
  const int stride_in_bytes = grey.rows() * comp;           // length of one row in bytes
  std::vector<unsigned char> data(grey.size() * comp, 0);   // the image itself
  for (unsigned i = 0; i < grey.rows(); ++i) {
    for (unsigned j = 0; j < grey.cols(); ++j) {
      data[(j * grey.rows() * comp) + (i * comp) + 0] = grey(i, grey.cols() - 1 - j);
    }
  }
  // convert unsigned char vector into buffer
  int len;
  unsigned char* png_buffer_ptr = stbi_write_png_to_mem(data.data(), stride_in_bytes, grey.rows(), grey.cols(), comp, &len);
  // write blocks of data into encoded string
  if (png_buffer_ptr != nullptr) {
    std::ostringstream os;
    for (int i = 0; i < len; i++) {
      os << *(png_buffer_ptr + i);
    }
    output = os.str();
    // release buffer pointer
    free(png_buffer_ptr);
    return 0;
  } else {
    // release buffer pointer
    free(png_buffer_ptr);
    return 1;
  }
}

} // namespace utils
} // namespace noesis

/* EOF */
