/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_CORE_OBJECT_HPP_
#define NOESIS_FRAMEWORK_CORE_OBJECT_HPP_

// C/C++
#include <string>

// Noesis
#include "noesis/framework/utils/string.hpp"

namespace noesis {
namespace core {

/*!
 * @brief Helper struct for object construction.
 */
struct ObjectConfig {
  std::string name;
  std::string scope;
  bool verbose{false};
};

/*!
 * @brief The base class for Noesis objects which are name-scoped.
 */
class Object
{
public:
  
  Object(Object&& other) = default;
  Object& operator=(Object&& other) = default;
  
  Object(const Object& other) = default;
  Object& operator=(const Object& other) = default;
  
  /*!
   * @brief Constructor enforcing initialization of the object's name and scope at construction time.
   * @param name The unique object name with the local scope.
   * @param scope The scope within which the object instance exists.
   */
  explicit Object(std::string name, std::string scope="/", bool verbose=false):
    name_(std::move(name)),
    scope_(std::move(scope)),
    verbose_(verbose)
  {
  }
  
  /*!
   * @brief Delegating constructor for creating objects using the respective helper struct.
   * @param config The helper configuration struct instance.
   */
  explicit Object(const ObjectConfig& config):
    Object(config.name, config.scope, config.verbose)
  {
  }

  /*!
   * @brief Default destructor.
   */
  virtual ~Object() = default;
  
  /*
   * Properties
   */
  
  /*!
   * @brief Retrieves the name of the object instance within its local scope.
   * @return An STL string containing the object's local name.
   */
  const std::string& name() const {
    return name_;
  }
  
  /*!
   * @brief Retrieves the scope within which the object instance exists.
   * @return An STL string containing the object's scope.
   */
  const std::string& scope() const {
    return scope_;
  }
  
  /*!
   * @brief Retrieves the fully qualified (scoped) name of the object instance.
   * @note This name should be unique within the global (root) '/' scope.
   * @return An STL string containing the object's scoped named.
   */
  std::string namescope() const {
    return utils::make_namescope({scope_, name_});
  }
  
  /*!
   * @brief Retrieves the current verbosity of the object instance.
   * @return True if verbose output is enabled.
   */
  bool verbosity() const {
    return verbose_;
  }
  
  /*!
   * @brief Checks if the current object instance is set for verbose output.
   * @return True if verbose output is enabled.
   */
  bool isVerbose() const {
    return verbose_;
  }

protected:
  
  /*
   * Configurations
   */
  
  void setName(std::string name) {
    name_ = name;
  }
  
  void setScope(std::string scope) {
    scope_ = scope;
  }
  
  void setVerbosity(bool verbose) {
    verbose_ = verbose;
  }
  
private:
  //! @brief The name of the agent instance.
  std::string name_;
  //! @brief The name-scope to which the agent instance belongs.
  std::string scope_;
  //! @brief Enables verbose output.
  bool verbose_;
};

} // namespace core
} // namespace noesis

#endif // NOESIS_FRAMEWORK_CORE_OBJECT_HPP_

/* EOF */
