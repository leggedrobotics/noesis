/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Noesis
#include <noesis/framework/math/random.hpp>

int main(int argc, char **argv)
{
  /*
   * This is a simple example on the noesis::math::RandomNumberGenerator class. It provides several commonly known probability distribution
   * for scalar values. The first argument in the constructor sets the seed of the pseudo-random number generator.
   */
  noesis::math::RandomNumberGenerator<float> fprng0(42);
  noesis::math::RandomNumberGenerator<float> fprng1(37);
  noesis::math::RandomNumberGenerator<double> fprng2(42);

  const int numOfSamples = 10;
  std::vector<float> samples0(numOfSamples, 0.0);
  std::vector<float> samples1(numOfSamples, 0.0);
  std::vector<double> samples2(numOfSamples, 0.0);

  for (int k=0; k<numOfSamples; k++) {
    samples0[k] = fprng0.sampleStandardUniform();
    samples1[k] = fprng1.sampleUnitUniform();
    samples2[k] = fprng2.sampleUniform(-42.0, 42.0);
  }

  std::cout << "Samples:\n";
  for (int k=0; k<numOfSamples; k++) {
    std::cout << "  " << samples0[k] << ", " << samples1[k] << ", " << samples2[k] << "\n";
  }

  // Success
  return 0;
}

/* EOF */
