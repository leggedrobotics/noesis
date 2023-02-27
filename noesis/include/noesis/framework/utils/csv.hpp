/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_UTILS_CSV_HPP_
#define NOESIS_FRAMEWORK_UTILS_CSV_HPP_

// C/C++
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Boost
#include <boost/algorithm/string.hpp>

namespace noesis {
namespace utils {

template<typename ScalarType_>
class CsvReader
{
public:

  using ScalarType = ScalarType_;
  using ChannelType = std::pair<std::string, std::vector<ScalarType>>;
  using DataSetType = std::vector<ChannelType>;

  explicit CsvReader(std::string filename, std::string delimiter=",") :
  fileName_(std::move(filename)),
  delimiter_(std::move(delimiter))
  {
  }

  ~CsvReader() = default;

  bool parse(DataSetType& data) {
    // Ensure that data is empty before proceeding
    if (!data.empty()) {
      std::cout << "Error: argument must be an empty data set!";
    }
    // Parse and load the data into process memory
    std::ifstream file(fileName_);
    std::string line;
    while (getline(file, line)) {
      // Allocate new buffers
      std::vector<std::string> charBuffer;
      std::vector<ScalarType> valueBuffer;
      // Split using the delimiter
      boost::algorithm::split(charBuffer, line, boost::is_any_of(delimiter_));
      // Store into contiguous memory
      valueBuffer.resize(charBuffer.size()-1);
      for (size_t k = 1; k < charBuffer.size(); ++k) {
        if (!charBuffer[k].empty()) {
          valueBuffer[k-1] = static_cast<ScalarType>(std::stod(charBuffer[k]));
        }
      }
      // Create a new data channel element
      data.push_back(std::make_pair(charBuffer[0], valueBuffer));
    }
    // Close the File
    file.close();
    // Success
    return true;
  }

  int find(const DataSetType& data, const std::string& channel_name) {
    int index = -1;
    for (int k = 0; k < data.size(); ++k) {
      if (data[k].first == channel_name) {
        index = k;
      }
    }
    return index;
  }

private:
  std::string fileName_;
  std::string delimiter_;
};

} // utils
} //noesis

#endif // NOESIS_FRAMEWORK_UTILS_CSV_HPP_

/* EOF */
