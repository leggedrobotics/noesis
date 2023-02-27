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
#ifndef NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETER_TUPLE_HPP_
#define NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETER_TUPLE_HPP_

// C/C++
#include <algorithm>
#include <unordered_map>

// Noesis
#include "noesis/framework/hyperparam/HyperParameter.hpp"
#include "noesis/framework/utils/string.hpp"

namespace noesis {
namespace hyperparam {

/*!
 * @brief A container to instantiate a tuple/set of hyper-parameters and manage them as a group.
 */
class HyperParameterTuple
{
public:

  // Aliases
  using TupleType = std::unordered_map<std::string, std::shared_ptr<internal::HyperParameterInterface>>;

  /*!
   * @brief Construct a named HyperParameter tuple of type (identifier) `type`.
   * @param name The name (scoped) of the tuple instance.
   * @param category The category of the tuple (optimizer, parameter_decay, architecture)
   * @param type The auxiliary tuple type identifier.
   */
  explicit HyperParameterTuple(std::string scope, std::string name, std::string category, std::vector<std::string> range);

  /*!
   * @brief Construct a named HyperParameter tuple of type (identifier) `type`.
   * @param name The name (scoped) of the tuple instance.
   * @param category The category of the tuple (optimizer, parameter_decay, architecture)
   * @param type The auxiliary tuple type identifier.
   */
  explicit HyperParameterTuple(std::string scope, std::string name, std::string category);

  /*!
   * @brief Copy constructor.
   * @param other The source hyper-parameter tuple to copy.
   */
  HyperParameterTuple(const HyperParameterTuple& other) = default;
  
  /*!
   * @brief Default destructor.
   */
  virtual ~HyperParameterTuple() = default;
  
  /*!
   * @brief Retrieve the scope within which the parameter tuple is defined.
   * @return The name of the parameter tuple instance.
   */
  const std::string& getScope() const;
  
  /*!
   * @brief Retrieve the name of the parameter tuple instance.
   * @return The name of the parameter tuple instance.
   */
  const std::string& getName() const;
  
  /*!
   * @brief Retrieve the category of the parameter tuple instance.
   * @return The category of the parameter tuple instance.
   */
  const std::string& getCategory() const;
  
  /*!
   * @brief Retrieve the full name of the parameter which includes its scope and category.
   * @return The scoped name (<scope>/<category>/<name>) of the parameter tuple instance.
   */
  std::string namescope() const;
  
  /*!
   * @brief Retrieve the type of the parameter tuple instance.
   * @return The type of the parameter tuple instance.
   */
  std::string getType() const;
  
  /*!
   * @brief Sets the type string to describe the parameter tuple.
   * @param type An std::string containing the parameter type description.
   */
  virtual void setType(const std::string& type);
  
  /*!
   * @brief Checks if the tuple holds a parameter by a specific name.
   * @param name The name of the parameter to be searched.
   * @return True if the tuple holds a parameter by that name.
   */
  bool exists(const std::string& name) const;
  
  /*!
   * @brief Retrieves the current
   * @return The number of parameters in the tuple.
   */
  size_t size() const;
  
  /*!
   * @brief Clears all parameters held by the tuple instance.
   * @warning Parameters are removed from the tuple, but will still be allocated if registered with the manager.
   */
  void clear();
  
  /*!
   * @brief Retrieves the underlying hyper-parameter tuple container.
   * @return Const-reference to the internal tuple container.
   */
  const TupleType& get() const;
  
  /*!
   * @brief Add a new parameter to the tuple  with range limits.
   * @tparam ValueType_ The fundamental data type.
   * @param value The default value to set for the new parameter.
   * @param name The name of the new parameter - must be unique in the tuple.
   * @param range
   */
  template<typename ValueType_>
  void addParameter(ValueType_ value, const std::string& name, const std::vector<ValueType_>& range) {
    NFATAL_IF(exists(name), "[" << this->namescope() << "]: '" << name << "' already exist in this tuple.");
    auto param_scope = utils::make_namescope({scope_, category_, name_, name});
    parameters_->emplace(name, std::make_shared<hyperparam::HyperParameter<ValueType_>>(value, param_scope, range));
  }

  /*!
   * @brief Add a new parameter to the tuple, but with range limits.
   * @tparam ValueType_ The fundamental data type.
   * @param value The default value to set for the new parameter.
   * @param name The name of the new parameter - must be unique in the tuple.
   */
  template<typename ValueType_>
  void addParameter(ValueType_ value, const std::string& name) {
    NFATAL_IF(exists(name), "[" << this->namescope() << "]: '" << name << "' already exist in this tuple.");
    auto param_scope = utils::make_namescope({scope_, category_, name_, name});
    parameters_->emplace(name, std::make_shared<hyperparam::HyperParameter<ValueType_>>(value, param_scope));
  }

  /*!
   * @brief Retrieves the value of a specific parameter in the tuple by name.
   * @tparam ValueType_ The expected value type of the target parameter element.
   * @param name The name of the parameter.
   * @return The expected value fof the target parameter.
   */
  template<typename ValueType_>
  ValueType_ getParameterValue(const std::string& name) const {
    NFATAL_IF(!exists(name), "[" << this->namescope() << "]: '" << name << "' does not exist in this tuple.");
    return parameters_->at(name)->getValue<ValueType_>();
  }

  /*!
   * @brief Sets the value of a specific parameter in the tuple by name.
   * @tparam ValueType_ The expected value type of the target parameter element.
   * @param name The name of the parameter.
   */
  template<typename ValueType_>
  void setParameterValue(const std::string& name, ValueType_ value) const {
    NFATAL_IF(!exists(name), "[" << this->namescope() << "]: '" << name << "' does not exist in tuple '" << name_ << "'.");
    parameters_->at(name)->setValue<ValueType_>(value);
  }
  
  /*!
   * @brief Loads the hyper-parameters from an XML element description.
   */
  void fromXml(const TiXmlElement& element);
  
  /*!
  * @brief Stores hyper-parameters into an XML element description.
  * @return Returns the XML element with the parameter tuple description.
  */
  TiXmlElement* toXml(bool simplified=false) const;
  
  /*!
   * @brief Outputs all currently held parameters to XML format as a std::string instead of an TiXmlElement.
   * @param simplified Set to true to output the simplified XML format (removes ranges and type strings).
   * @return The std::string containing the XML description of the tuple.
   */
  std::string toXmlStr(bool simplified=false);
  
  /*!
   * @brief Constructs the tuple parameters directly form parsing the XML element description.
   * @warning The XML element must contain the parameter descriptions according to the correct format.
   * @param node The TiXmlElement containing the listed parameters to be created.
   */
  void createParametersFromXml(const TiXmlElement& element);
  
protected:

  bool checkType(const std::string& type) const;

private:
  //! The range of supported types
  std::vector<std::string> range_;
  //! The scope name in which the parameter tuple exits.
  std::string scope_;
  //! The name of the parameter with the owning scope.
  std::string name_;
  //! The category of the tuple (e.g. optimizer, parameter_decay, architecture)
  std::string category_;
  //! The type of tuple within the category.
  std::shared_ptr<std::string> type_;
  //! The internal tuple container.
  std::shared_ptr<TupleType> parameters_;
};

} // namespace hyperparam
} // namespace noesis

/*!
 * @brief HyperParameterTuple: stream operator
 * @param os Target output stream to output the tuple description.
 * @param tuple The source hyper-parameter tuple.
 * @return The augmented output stream.
 */
std::ostream& operator<<(std::ostream& os, const noesis::hyperparam::HyperParameterTuple& tuple);

#endif // NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETER_TUPLE_HPP_

/* EOF */
