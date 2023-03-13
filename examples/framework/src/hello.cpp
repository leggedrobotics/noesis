/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// C/C++
#include <iostream>

// Noesis
#include <noesis/noesis.hpp>

int main(int argc, char **argv)
{
  /*
   * Before calling any other noesis function, you should call noesis::init() once. This takes arguments to the name of the process and
   * the path of the logging directory. If the path is empty, a directory is created in ~/.noesis/proc to save all the logs.
   */
  noesis::init("noesis_hello_example");
  
  /*
   * Print a basic message to ensure that the noesis package has been installed properly.
   */
  NINFO("Hello!");
  
  /*
   * The noesis::tf::available_devices() function, checks which all CPU/GPU devices are present in the system.
   * @note If TensorFlow is built in the CPU mode (i.e. CPU_ONLY flag is enabled in CMake), then no GPU devices would be detected.
   */
  NINFO("TensorFlow: Found the following compute devices: " << noesis::utils::vector_to_string(noesis::tf::available_devices()));
  
  /*
   * Print system/platform information in the form of dependency versions.
   */
  NINFO("System:"
    << "\nCompiler: " << noesis::compiler_version()
    << "\nC++: " << noesis::cxx_version()
    << "\nEigen: " << noesis::eigen_version()
    << "\nTensorFlow: " << noesis::tensorflow_version()
  );
  
  // Success
  return 0;
}

/* EOF */
