/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    David Hoeller
 * @email     dhoeller@ethz.ch
 * @author    Jemin Hwangbo
 * @email     jhwangno@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

#ifndef NOESIS_FRAMEWORK_MATH_RANDOM_HPP_
#define NOESIS_FRAMEWORK_MATH_RANDOM_HPP_

// C/C++
#include <cstdlib>
#include <mutex>
#include <unordered_set>

// Boost
#include <boost/random.hpp>
#include <boost/math/distributions.hpp>

// Eigen
#include <Eigen/Core>

// Noesis
#include "noesis/framework/log/message.hpp"

namespace noesis {
namespace math {

/*!
 * @brief A simple (pseudo) random-number generator for scalar values.
 * @tparam ScalarType_ The base arithmetic type defines the sample (event) space.
 */
template<typename ScalarType_>
class RandomNumberGenerator
{
public:
  
  static_assert(std::is_arithmetic<ScalarType_>::value, "ScalarType_ must be an arithmetic type: {int, float, double, etc..}");

  //! Alias for the scalar type
  using Scalar = ScalarType_;
  using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  /*!
   * @brief Default constructor.
   * @note A default seed is applied to initialize the underlying engine.
   */
  RandomNumberGenerator():
    generator_(static_cast<uint32_t>(0)) {
  }

  /*!
   * @brief Explicit constructor which initializes the seed of the random generator.
   * @param seed The random number generator seed.
   */
  explicit RandomNumberGenerator(uint32_t seed):
    generator_(static_cast<uint32_t>(seed)) {
  }

  /*!
   * @brief Default destructor.
   */
  ~RandomNumberGenerator() = default;
  
  /*
   * Configurations
   */
  
  /*!
   * @brief Sets the seed for the underlying random-number generator.
   * @note You can use this method to make the random samples reproducible.
   * @param seed The integer seed value.
   */
  void seed(uint32_t seed) {
    generator_.seed(seed);
  }
  
  /*
   * Properties
   */
  
  /*!
   * @brief Retrieves the underlying PRNG engine.
   * @return Returns reference to the PRNG engine instance.
   */
  auto& generator() {
    return generator_;
  }
  
  /*
   * Scalar Operations
   */
  
  /*!
   * @brief Samples scalar values from a continuous uniform distribution over the range [0, 1].
   * @return Random scalar sample.
   */
  Scalar sampleStandardUniform() const {
    return sampleUniform(0, 1);
  }

  /*!
   * @brief Samples scalar values from a continuous uniform distribution over the range [-1, 1].
   * @return Random scalar sample.
   */
  Scalar sampleUnitUniform() const {
    return sampleUniform(-1, 1);
  }

  /*!
   * @brief Samples scalar values from a continuous uniform distribution over the range [min, max].
   * @return Random scalar sample.
   */
  Scalar sampleUniform(Scalar min, Scalar max) const {
    DNFATAL_IF(min >= max, "Argument 'min' must be smaller than argument 'max'.");
    auto distribution = boost::uniform_real<Scalar>(min, max);
    return distribution(generator_);
  }

  /*!
   * @brief Samples scalar values from a continuous standard normal distribution N(x; 0, 1.0).
   * @return Random scalar sample.
   */
  Scalar sampleStandardNormal() const {
    return sampleNormal(0.0, 1.0);
  }

  /*!
   * @brief Samples scalar values from a continuous standard normal distribution N(x; mu, sigma^2).
   * @param mu The mean value.
   * @param sigma The standard deviation.
   * @return Random scalar sample.
   */
  Scalar sampleNormal(Scalar mu, Scalar sigma) const {
    auto distribution = boost::random::normal_distribution<Scalar>(mu, sigma);
    return distribution(generator_);
  }

  /*!
   * @brief Samples from a standardized bernoulli distribution Bernoulli(x; 0.5), with equal probability for either boolean result values.
   * @return Boolean value indicating if the result of the sampling.
   */
  bool sampleStandardBernoulli() const {
    auto distribution = boost::bernoulli_distribution<Scalar>();
    return distribution(generator_);
  }

  /*!
   * @brief Samples from a standardized bernoulli distribution Bernoulli(x; phi).
   * @param phi The probability of the random variable x being equal to 1 --> Pr(X=1) = phi, Pr(X=0) = 1 - phi
   * @return Boolean value indicating if the result of the sampling.
   */
  bool sampleBernoulli(Scalar phi) const {
    auto distribution = boost::bernoulli_distribution<Scalar>(phi);
    return  distribution(generator_);
  }

  /*!
   * @brief Samples scalar values from a discrete uniform distribution over the range [INT_MIN, INT_MAX].
   * @return Random scalar sample.
   */
  int sampleIntegerUniform() const {
    return sampleIntegerUniform(std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max());
  }

  /*!
   * @brief Samples scalar values from a discrete uniform distribution over the range [min, max].
   * @return Random scalar sample.
   */
  int sampleIntegerUniform(int min, int max) const {
    DNFATAL_IF(min >= max, "Argument 'min' must be smaller than argument 'max'.");
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator_);
  }
  
  /*
   * Vector Operations
   */
  
  /*!
   * @brief Samples 2D vector with coefficients from a continuous standard normal distribution N(x; mu, sigma^2).
   * @param mu The uniform mean value.
   * @param sigma The uniform standard deviation.
   * @return Random 2D vector sample.
   */
  Vector2 sampleNormalVec2(Scalar mu, Scalar sigma) const {
    return Vector2(sampleNormal(mu, sigma), sampleNormal(mu, sigma));
  }
  
  /*!
   * @brief Samples 2D vector with coefficients from a continuous standard normal distribution N(x; mu, sigma^2).
   * @param mu The vector of mean values.
   * @param sigma The vector of standard deviations.
   * @return Random 2D vector sample.
   */
  Vector2 sampleNormalVec2(const Vector2& mu, const Vector2& sigma) const {
    const Vector2 sample(sampleStandardNormal(), sampleStandardNormal());
    return mu + sigma.cwiseProduct(sample);
  }
  
  /*!
   * @brief Samples 3D vector with coefficients from a continuous standard normal distribution N(x; mu, sigma^2).
   * @param mu The uniform mean value.
   * @param sigma The uniform standard deviation.
   * @return Random 3D vector sample.
   */
  Vector3 sampleNormalVec3(Scalar mu, Scalar sigma) const {
    return Vector3(sampleNormal(mu, sigma), sampleNormal(mu, sigma), sampleNormal(mu, sigma));
  }
  
  /*!
   * @brief Samples 3D vector with coefficients from a continuous standard normal distribution N(x; mu, sigma^2).
   * @param mu The vector of mean values.
   * @param sigma The vector of standard deviations.
   * @return Random 3D vector sample.
   */
  Vector3 sampleNormalVec3(const Vector3& mu, const Vector3& sigma) const {
    const Vector3 sample(sampleStandardNormal(), sampleStandardNormal(), sampleStandardNormal());
    return mu + sigma.cwiseProduct(sample);
  }
  
  /*!
 * @brief Samples 2D vector with coefficients from a continuous standard normal distribution N(x; mu, sigma^2).
 * @param mu The uniform mean value.
 * @param sigma The uniform standard deviation.
 * @return Random 2D vector sample.
 */
  Vector2 sampleUniformVec2(Scalar min, Scalar max) const {
    return Vector2(sampleUniform(min, max), sampleUniform(min, max));
  }
  
  /*!
   * @brief Samples 2D vector with coefficients from a continuous standard normal distribution N(x; mu, sigma^2).
   * @param mu The vector of mean values.
   * @param sigma The vector of standard deviations.
   * @return Random 2D vector sample.
   */
  Vector2 sampleUniformVec2(const Vector2& min, const Vector2& max) const {
    return Vector2(sampleUniform(min(0), max(0)), sampleUniform(min(1), max(1)));
  }
  
  /*!
   * @brief Samples 3D vector with coefficients from a continuous standard normal distribution N(x; mu, sigma^2).
   * @param mu The uniform mean value.
   * @param sigma The uniform standard deviation.
   * @return Random 3D vector sample.
   */
  Vector3 sampleUniformVec3(Scalar min, Scalar max) const {
    return Vector3(sampleUniform(min, max), sampleUniform(min, max), sampleUniform(min, max));
  }
  
  /*!
   * @brief Samples 3D vector with coefficients from a continuous standard normal distribution N(x; mu, sigma^2).
   * @param mu The vector of mean values.
   * @param sigma The vector of standard deviations.
   * @return Random 3D vector sample.
   */
  Vector3 sampleUniformVec3(const Vector3& min, const Vector3& max) const {
    const Vector3 sample(sampleStandardNormal(), sampleStandardNormal(), sampleStandardNormal());
    return Vector3(sampleUniform(min(0), max(0)), sampleUniform(min(1), max(1)), sampleUniform(min(2), max(2)));
  }
  
protected:
  //! The underlying pseudo-random number generation (PRNG) engine.
  mutable boost::random::mt19937_64 generator_;
};

} // namespace math
} // namespace noesis

#endif // NOESIS_FRAMEWORK_MATH_RANDOM_HPP_

/* EOF */
