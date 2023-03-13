/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 * @author    David Hoeller
 * @email     dhoeller@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETER_HPP_
#define NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETER_HPP_

// C/C++
#include <iostream>
#include <memory>
#include <type_traits>

// Boost
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/lock_types.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/spirit/home/support/string_traits.hpp>

// TinyXML
#include <tinyxml.h>

// Noesis
#include "noesis/framework/system/process.hpp"
#include "noesis/framework/utils/macros.hpp"
#include "noesis/framework/log/message.hpp"
#include "noesis/framework/utils/xml.hpp"

namespace noesis {
namespace hyperparam {

//! Macro list of typeid values of supported data types for hyper-parameters
#define PARAM_SUPPORTED_TYPE_IDS \
typeid(bool).name(), \
typeid(int).name(), \
typeid(float).name(), \
typeid(double).name(), \
typeid(std::string).name(), \
typeid(std::vector<bool>).name(), \
typeid(std::vector<int>).name(), \
typeid(std::vector<float>).name(), \
typeid(std::vector<double>).name(), \
typeid(std::vector<std::string>).name()

//! Macro list of names of supported data types for hyper-parameters
#define PARAM_SUPPORTED_TYPE_NAMES \
"bool", \
"int", \
"float", \
"double", \
"string", \
"bools", \
"ints", \
"floats", \
"doubles", \
"strings"

namespace internal {

/*!
 * @brief Type-trait for parameter type detection: resolves to true only for supported parameter types.
 */
template <typename T>
struct is_supported {
  static constexpr bool value =
    std::is_same<T, bool>::value ||
    std::is_same<T, int>::value ||
    std::is_same<T, float>::value ||
    std::is_same<T, double>::value ||
    std::is_same<T, std::string>::value ||
    std::is_same<T, std::vector<bool>>::value ||
    std::is_same<T, std::vector<int>>::value ||
    std::is_same<T, std::vector<float>>::value ||
    std::is_same<T, std::vector<double>>::value ||
    std::is_same<T, std::vector<std::string>>::value;
};

/*!
 * @brief Type-trait for parameter type detection: resolves to true only for supported scalar parameter types.
 */
template <typename T>
struct is_supported_scalar {
  static constexpr bool value =
    std::is_same<T, bool>::value ||
    std::is_same<T, int>::value ||
    std::is_same<T, float>::value ||
    std::is_same<T, double>::value ||
    std::is_same<T, std::string>::value;
};

/*!
 * @brief Type-trait for parameter type detection: resolves to true only for supported scalar parameter types excluding booleans.
 */
template <typename T>
struct is_supported_nonbool_scalar {
  static constexpr bool value =
    std::is_same<T, int>::value ||
    std::is_same<T, float>::value ||
    std::is_same<T, double>::value ||
    std::is_same<T, std::string>::value;
};

/*!
 * @brief Type-trait for parameter type detection: resolves to true only for supported numeric scalar parameter types.
 */
template <typename T>
struct is_supported_numeric_scalar {
  static constexpr bool value =
    std::is_same<T, int>::value ||
    std::is_same<T, float>::value ||
    std::is_same<T, double>::value;
};

/*!
 * @brief Type-trait for parameter vector type detection: resolves to true only for supported vector parameter types.
 */
template<typename T>
struct is_supported_vector {
  static constexpr bool value =
    std::is_same<T, std::vector<bool>>::value ||
    std::is_same<T, std::vector<int>>::value ||
    std::is_same<T, std::vector<float>>::value ||
    std::is_same<T, std::vector<double>>::value ||
    std::is_same<T, std::vector<std::string>>::value;
};

/*!
 * @brief Type-trait for parameter vector type detection: resolves to true only for supported vector parameter types excluding booleans.
 */
template<typename T>
struct is_supported_nonbool_vector {
  static constexpr bool value =
    std::is_same<T, std::vector<int>>::value ||
    std::is_same<T, std::vector<float>>::value ||
    std::is_same<T, std::vector<double>>::value ||
    std::is_same<T, std::vector<std::string>>::value;
};

/*!
 * @brief Type-trait for parameter vector type detection: resolves to true only for supported numeric vector parameter types.
 */
template<typename T>
struct is_supported_numeric_vector {
  static constexpr bool value =
    std::is_same<T, std::vector<int>>::value ||
    std::is_same<T, std::vector<float>>::value ||
    std::is_same<T, std::vector<double>>::value;
};

/*!
 * @brief Helper class for detecting hyper-parameter types and mapping to these to typeid and string values.
 */
class ParameterTypeMap
{
public:

  explicit ParameterTypeMap():
    types_({PARAM_SUPPORTED_TYPE_NAMES}),
    typesInfo_({PARAM_SUPPORTED_TYPE_IDS}) {
  }

  ~ParameterTypeMap() = default;

  const std::string& getTypeFromTypeInfo(const std::string& typeinfo) {
    auto it = std::find(typesInfo_.begin(), typesInfo_.end(), typeinfo);
    if (it == typesInfo_.end()) {
      NFATAL("TypeInfo " << typeinfo << " is not supported.");
    }
    long index = distance(typesInfo_.begin (), it);
    return types_[index];
  }

protected:
  const std::vector<std::string> types_;
  const std::vector<std::string> typesInfo_;
};

/*!
 * @brief Internal interface class for parameter values - abstracts data validity checks.
 */
class ParameterValueInterface
{
public:

  explicit ParameterValueInterface():
    used_(true) {
  }

  virtual ~ParameterValueInterface() = default;

  bool isUsed() {
    return used_;
  }

  void setAsUnused() {
    used_ = false;
  }

protected:
  bool used_;
};

/*!
 * @brief Parameter value abstraction class to create generic parameter access: vector<T> and scalar T (arithmetic types).
 * @note Provides thread-safety via mutex-based data synchronization for concurrent access with the process.
 * @tparam ValueType_ The parameter type can be anything.
 */
template<typename ValueType_>
class ParameterValue final: public ParameterValueInterface
{
public:

  explicit ParameterValue(ValueType_ value, std::vector<ValueType_> range, bool range_is_discrete=false):
    ParameterValueInterface(),
    value_(value),
    range_(range),
    mutex_(),
    rangeIsDiscrete_(range_is_discrete) {
  }

  explicit ParameterValue(ValueType_ value):
    ParameterValueInterface(),
    value_(value),
    range_(),
    mutex_(),
    rangeIsDiscrete_(false) {
  }

  ~ParameterValue() override = default;

  const ValueType_& getValue() const {
    boost::shared_lock<boost::shared_mutex> lock(mutex_);
    return value_;
  }
  
  void setValue(const ValueType_& value) {
    boost::unique_lock<boost::shared_mutex> lock(mutex_);
    value_ = value;
  }

  const std::vector<ValueType_>& getRange() const {
    return range_;
  }

  bool rangeIsDiscrete() const {
    return rangeIsDiscrete_;
  }

protected:
  ValueType_ value_;
  const std::vector<ValueType_> range_;
  mutable boost::shared_mutex mutex_;
  const bool rangeIsDiscrete_;
};

/*!
 * @brief This class provides a common interface between parameter instances without requiring knowledge of the underlying type.
 */
class HyperParameterInterface
{
public:

  explicit HyperParameterInterface(const std::string& typeinfo):
    name_(),
    type_(ParameterTypeMap().getTypeFromTypeInfo(typeinfo)),
    scope_(),
    value_()
  {
  }

  virtual ~HyperParameterInterface() {
    if (value_.use_count() <= 2) {
      value_->setAsUnused();
    }
  }

  /*
   * Generic HyperParameter Properties
   */

  const std::string& getName() const {
    return name_;
  }
  
  const std::vector<std::string>& getScope() const {
    return scope_;
  }
  
  const std::string& getType() const {
    return type_;
  }
  
  /*
   * Generic HyperParameter Configurations
   */
  
  void setName(const std::string& name) {
    // We use boost::filesystem to handle name-scoping using "/" separators
    boost::filesystem::path scopedName = name;
    // Create a vector of strings to hold the name-scope internally
    std::vector<std::string> scope;
    for (auto const& subscope : scopedName)
      scope.push_back(subscope.string());
    // Set the name-scope vector into the parameters special name member
    this->setScope(scope);
  }

  void setScope(const std::vector<std::string>& scope) {
    // Set the new scope
    scope_ = scope;
    // Update the full name from the new scope
    name_.clear();
    for (auto& n: scope_) {
      name_ += n + "/";
    }
    name_.erase(name_.size() - 1, 1);
  }

  /*
   * Specialized (templated) HyperParameter Interface
   */

  template<typename ValueType_>
  const ValueType_& getValue() const {
    return this->getValuePtr<ValueType_>()->getValue();
  }

  template<typename ValueType_, typename std::enable_if_t<internal::is_supported_scalar<ValueType_>::value>* = nullptr>
  void setValue(const ValueType_& value) {
    this->checkRange<ValueType_>(value);
    this->getValuePtr<ValueType_>()->setValue(value);
  }
  
  template<typename ValueType_, typename std::enable_if_t<internal::is_supported_vector<ValueType_>::value>* = nullptr>
  void setValue(const ValueType_& value) {
    auto newSize = value.size();
    auto currentSize = this->getValuePtr<ValueType_>()->getValue().size();
    auto name = this->getName();
    this->checkRange<ValueType_>(value);
    this->getValuePtr<ValueType_>()->setValue(value);
  }
  
  template<typename ValueType_>
  const std::vector<ValueType_>& getRange() const {
    return this->getValuePtr<ValueType_>()->getRange();
  }

  template<typename ValueType_>
  bool rangeIsDiscrete() const {
    return this->getValuePtr<ValueType_>()->rangeIsDiscrete();
  }

  template<typename ValueType_, typename std::enable_if_t<internal::is_supported_scalar<ValueType_>::value>* = nullptr>
  TiXmlElement* toXmlElement(bool simplified=false) {
    boost::filesystem::path scopedName = this->getName();
    std::string unscopedName = scopedName.filename().string();
    auto* element = new TiXmlElement(unscopedName);
    utils::xml::setXmlAttribute("value", this->getValue<ValueType_>(), element);
    if (!simplified) {
      utils::xml::setXmlAttribute("type", type_, element);
      auto range = this->getRange<ValueType_>();
      if (!range.empty()) {
        std::ostringstream stream;
        for (size_t k=0; k<range.size(); k++) {
          stream << range[k] << " ";
        }
        utils::xml::setXmlAttribute("range", stream.str(), element);
        if ((range.size() == 2) && (type_ != "string") && (type_ != "strings")) {
          utils::xml::setXmlAttribute("discrete", rangeIsDiscrete<ValueType_>(), element);
        }
      }
    }
    return element;
  }

  template<typename ValueType_, typename std::enable_if_t<internal::is_supported_nonbool_vector<ValueType_>::value>* = nullptr>
  TiXmlElement* toXmlElement(bool simplified=false) {
    boost::filesystem::path scopedName = this->getName();
    std::string unscopedName = scopedName.filename().string();
    auto* element = new TiXmlElement(unscopedName);
    if (!simplified) {
      utils::xml::setXmlAttribute("type", type_, element);
    }
    const auto& ranges = this->getRange<ValueType_>();
    const auto& values = this->getValue<ValueType_>();
    int index = 0;
    for (const auto& elem : values) {
      auto* subElement = new TiXmlElement("element");
      utils::xml::setXmlAttribute("value", elem, subElement);
      if (!simplified) {
        if (ranges.size() == 0) {
          // Do nothing: no range is to be added
        } else if (ranges.size() == 1) {
          auto range = ranges[0];
          if (!range.empty()) {
            std::ostringstream stream;
            for (size_t k = 0; k < range.size(); k++) {
              stream << range[k] << " ";
            }
            utils::xml::setXmlAttribute("range", stream.str(), subElement);
            if ((range.size() == 2) && (type_ != "strings")) {
              utils::xml::setXmlAttribute("discrete", rangeIsDiscrete<ValueType_>(), subElement);
            }
          }
        } else if (ranges.size() == values.size()) {
          auto range = ranges[index];
          if (!range.empty()) {
            std::ostringstream stream;
            for (size_t k = 0; k < range.size(); k++) {
              stream << range[k] << " ";
            }
            utils::xml::setXmlAttribute("range", stream.str(), subElement);
            if ((range.size() == 2) && (type_ != "strings")) {
              utils::xml::setXmlAttribute("discrete", rangeIsDiscrete<ValueType_>(), subElement);
            }
          }
        } else {
          NFATAL("[" << this->getName() << "]: Ranges must be single or contain equal number of elements as values.");
        }
      }
      element->LinkEndChild(subElement);
    }
    return element;
  }
  
  template<typename ValueType_, typename std::enable_if_t<std::is_same<ValueType_, std::vector<bool>>::value>* = nullptr>
  TiXmlElement* toXmlElement(bool simplified=false) {
    boost::filesystem::path scopedName = this->getName();
    std::string unscopedName = scopedName.filename().string();
    auto* element = new TiXmlElement(unscopedName);
    if (!simplified) {
      utils::xml::setXmlAttribute("type", type_, element);
    }
    for (const auto& elem : this->getValue<ValueType_>()) {
      auto* subElement = new TiXmlElement("element");
      utils::xml::setXmlAttribute("value", elem, subElement);
      element->LinkEndChild(subElement);
    }
    return element;
  }

  template<typename ValueType_>
  void fromXmlElement(TiXmlElement* element) {
    // Check type
    const std::string val_type = ParameterTypeMap().getTypeFromTypeInfo(typeid(ValueType_).name());
    const std::string element_type = element->Attribute("type");
    NFATAL_IF((type_ != val_type), "HyperParameter '" << name_ << "'(" << type_ << ") is not of type '" << val_type << "'.");
    NFATAL_IF((type_ != element_type), "HyperParameter '" << name_ << "'(" << type_ << ") is not of type '" << element_type << "'.");
    // Retrieve value from node
    std::stringstream stream(element->Attribute("value"));
    ValueType_ value;
    stream >> value;
    // Check if value is in range
    this->checkRange<ValueType_>(value);
    // Set value from node
    this->setValue<ValueType_>(value);
  }

  template<typename ValueType_>
  std::string toXmlElementStr(bool simplified=false) {
    TiXmlPrinter printer;
    printer.SetIndent("  ");
    TiXmlHandle handle = this->toXmlElement<ValueType_>(simplified);
    handle.ToNode()->Accept(&printer);
    return printer.Str();
  }

  template<typename ValueType_>
  std::shared_ptr<internal::ParameterValue<ValueType_>> getValuePtr() const {
    return std::static_pointer_cast<internal::ParameterValue<ValueType_>>(value_);
  }

  template<typename ValueType_>
  void setValuePtr(const std::shared_ptr<internal::ParameterValue<ValueType_>>& value) {
    value_ = value;
  }
  
  size_t useCount() const {
    return value_.use_count();
  }
  
  template<typename ValueType_, typename std::enable_if_t<std::is_same<ValueType_, bool>::value>* = nullptr>
  inline void checkRange(const ValueType_& value) const {
    UNUSED_VARIABLE(value);
  }
  
  template<typename ValueType_, typename std::enable_if_t<internal::is_supported_numeric_scalar<ValueType_>::value>* = nullptr>
  inline void checkRange(const ValueType_& value) const {
    // Capture local copy of the range member
    auto range = this->getRange<ValueType_>();
    // Abort check if no range has been specified
    if (range.empty()) return;
    // If range has two elements, interpret as min/max
    if (range.size() == 2 && !this->rangeIsDiscrete<ValueType_>()) {
      bool valid = this->checkMinMaxRange<ValueType_>(range, value);
      NFATAL_IF(valid, "[" << this->getName() << "]: Value of '" << value << "' is out of range: [" << range[0] << ", " << range[1]<< "]" );
      return;
    }
    // If the range has more than two elements, the parameter is interpreted as being set-valued
    auto result = this->checkDiscreteRange<ValueType_>(range, value);
    NFATAL_IF(!result.first, "[" << this->getName() << "]: Value of '" << value << "' is out of range: {" << result.second << "}");
  }
  
  template<typename ValueType_, typename std::enable_if_t<std::is_same<ValueType_, std::string>::value>* = nullptr>
  inline void checkRange(const ValueType_& value) const {
    // Capture local copy of the range member
    auto range = this->getRange<ValueType_>();
    // Abort check if no range has been specified
    if (range.empty()) return;
    // If the range has more than two elements, the parameter is interpreted as being set-valued
    auto result = this->checkDiscreteRange<ValueType_>(range, value);
    NFATAL_IF(!result.first, "[" << this->getName() << "]: Value of '" << value << "' is out of range: {" << result.second << "}");
  }
  
  template<typename ValueType_, typename std::enable_if_t<std::is_same<ValueType_, std::vector<bool>>::value>* = nullptr>
  inline void checkRange(const ValueType_& value) const {
    UNUSED_VARIABLE(value);
  }
  
  template<typename ValueType_, typename std::enable_if_t<internal::is_supported_numeric_vector<ValueType_>::value>* = nullptr>
  inline void checkRange(const ValueType_& value) const {
    // Capture local copy of the range member
    auto ranges = this->getRange<ValueType_>();
    // Abort check if no range has been specified
    if (ranges.empty()) return;
    for (size_t k=0; k<value.size(); k++) {
      ValueType_ range;
      if (ranges.size() == 1) {
        range = ranges[0];
      } else {
        range = ranges[k];
      }
      // If range has two elements, interpret as min/max
      if (range.size() == 2 && !this->rangeIsDiscrete<ValueType_>()) {
        bool valid = this->checkMinMaxRange<typename ValueType_::value_type>(range, value[k]);
        NFATAL_IF(valid,
          "["<<this->getName()<<"]: Value '"<<value[k]<<"' at index '"<<k<<"' is out of range: ["<<range[0]<<", "<<range[1]<<"]");
        continue;
      }
      // If the range has more than two elements, the parameter is interpreted as being set-valued
      auto result = this->checkDiscreteRange<typename ValueType_::value_type>(range, value[k]);
      NFATAL_IF(!result.first,
        "["<<this->getName()<<"]: Value '"<<value[k]<<"' at index '"<<k<<"' is out of range: {"<<result.second<<"}");
    }
  }
  
  template<typename ValueType_, typename std::enable_if_t<std::is_same<ValueType_, std::vector<std::string>>::value>* = nullptr>
  inline void checkRange(const ValueType_& value) const {
    // Capture local copy of the range member
    auto ranges = this->getRange<ValueType_>();
    // Abort check if no range has been specified
    if (ranges.empty()) return;
    // string types are always discrete
    for (size_t k=0; k<value.size(); k++) {
      ValueType_ range;
      if (ranges.size() == 1) {
        range = ranges[0];
      } else {
        range = ranges[k];
      }
      auto result = this->checkDiscreteRange<typename ValueType_::value_type>(range, value[k]);
      NFATAL_IF(!result.first,"["<<this->getName()<<"]: Value '"<<value[k]<<"' at index '"<<k<<"' is out of range: {"<<result.second<<"}");
    }
  }
  
  template<typename ValueType_, typename std::enable_if_t<std::is_same<ValueType_, bool>::value>* = nullptr>
  inline void print(std::ostream& os = std::cout) const {
    os << this->getName() << ": "<< std::boolalpha << this->getValue<ValueType_>() << "\n";
  }
  
  template<typename ValueType_, typename std::enable_if_t<internal::is_supported_nonbool_scalar<ValueType_>::value>* = nullptr>
  inline void print(std::ostream& os = std::cout) const {
    os << this->getName() <<": "<< this->getValue<ValueType_>() << "\n";
  }
  
  template<typename ValueType_, typename std::enable_if_t<std::is_same<ValueType_, std::vector<bool>>::value>* = nullptr>
  inline void print(std::ostream& os = std::cout) const {
    const ValueType_ vector = this->getValue<ValueType_>();
    os << this->getName() << ": [";
    for (auto element: vector) {
      os << std::boolalpha << element << " ";
    }
    os << "\b]\n";
  }
  
  template<typename ValueType_, typename std::enable_if_t<internal::is_supported_nonbool_vector<ValueType_>::value>* = nullptr>
  inline void print(std::ostream& os = std::cout) const {
    const ValueType_ vector = this->getValue<ValueType_>();
    os << this->getName() << ": [";
    for (auto element: vector)
      os << element << " ";
    os << "\b]\n";
  }
  
private:

  template<typename ValueType_, typename std::enable_if_t<internal::is_supported_vector<std::vector<ValueType_>>::value>* = nullptr>
  inline bool checkMinMaxRange(const std::vector<ValueType_>& range, ValueType_ value) const {
    double min = range[0];
    double max = range[1];
    return (value < min || value > max);
  }

  template<typename ValueType_, typename std::enable_if_t<internal::is_supported_vector<std::vector<ValueType_>>::value>* = nullptr>
  inline std::pair<bool, std::string> checkDiscreteRange(const std::vector<ValueType_>& range, ValueType_ value) const {
    std::pair<bool, std::string> result = {true, ""};
    if (!(std::find(range.begin(), range.end(), value) != range.end())) {
      std::stringstream ss;
      for (auto element: range) {
        ss << ", " << element;
      }
      std::string rangeString = ss.str();
      rangeString.erase(0,2);
      result.first = false;
      result.second = rangeString;
    }
    return result;
  }

private:
  std::string name_;
  const std::string type_;
  std::vector<std::string> scope_;
  std::shared_ptr<ParameterValueInterface> value_;
};

} // namespace internal

/*!
 * @brief The primary parameter class for user-side instantiation of hyper-parameters
 * @tparam ValueType_ The base type of a specific parameter. e.g. {bool, int, float, ...}
 */
template<typename ValueType_>
class HyperParameter final: public internal::HyperParameterInterface
{
public:
  // Ensure that only supported types can instantiate this class
  static_assert(internal::is_supported<ValueType_>::value, "ValueType_ argument is not supported.");

  // Alias
  using ValueType = ValueType_;
  
  /*!
   * @brief Constructs using only a name, i.e. not requiring a value range for bounds checking.
   * @param value The initial value for the specific parameter instance.
   * @param name The scoped (contextual) name of the parameter instance.
   */
  HyperParameter(ValueType value, const std::string& name):
    HyperParameterInterface(typeid(ValueType).name())
  {
    this->setName(name);
    this->setValuePtr<ValueType>(std::make_shared<internal::ParameterValue<ValueType>>(value));
  }

  /*!
   * @brief Constructs an instance using the full parameter specification with value range for bounds checking.
   * @param value The initial value for the specific parameter instance.
   * @param name The scoped (contextual) name of the parameter instance.
   */
  HyperParameter(ValueType value, const std::string& name, const std::vector<ValueType>& range, bool range_is_discrete=false):
    HyperParameterInterface(typeid(ValueType).name())
  {
    this->setName(name);
    this->setValuePtr<ValueType>(std::make_shared<internal::ParameterValue<ValueType>>(value, range, range_is_discrete));
    this->checkRange(value);
  }
  
  /*!
   * @brief Constructs an alias of an existing parameter.
   * @param value_ptr The shared pointer to the underlying parameter value instance.
   * @param name The name of the source parameter instance.
   */
  HyperParameter(std::shared_ptr<internal::ParameterValue<ValueType_>> value_ptr, const std::string& name):
    HyperParameterInterface(typeid(ValueType).name())
  {
    this->setName(name);
    this->setValuePtr<ValueType>(value_ptr);
  }
  
  /*!
   * @brief Default destructor.
   */
  ~HyperParameter() override = default;
  
  /*!
   * @brief Access operator for retrieving the currently held value of the parameter.
   * @note This performs an explicit casting operation.
   * @note This method is thread-safe, as a mutex is used in the base class to protect access to the underlying value container.
   *
   * @code
   * // Create an integer parameter initialized to 42, but with permissible range of [0, 100]
   * HyperParameter<int> myInt(42, "myInt", {0, 100})
   * // Retrieve value
   * auto intValue = myInt;
   * // Print value
   * std::cout << "Int value: " << intValue << std::endl;
   * @endcode
   *
   * @return The current parameter value.
   */
  operator const ValueType&() const noexcept {
    return HyperParameterInterface::getValue<ValueType>();
  }
  
  /*!
   * @brief Access operator for updating the value of the parameter.
   * @note This performs an overload of the assignment operator.
   * @note This method is thread-safe, as a mutex is used in the base class to protect access to the underlying value container.
   *
   * @code
   * // Create an integer parameter initialized to 42, but with permissible range of [0, 100]
   * HyperParameter<int> myInt(42, "myInt", {0, 100})
   * // Set new value
   * myInt = 37;
   * // Print value
   * std::cout << "Int value: " << myInt << std::endl;
   * @endcode
   *
   * @return The current parameter value.
   */
  HyperParameter& operator=(const ValueType& value) noexcept {
    HyperParameterInterface::setValue<ValueType>(value);
    return *this;
  }
  
  /*!
   * @brief Retrieves the configured permissible range for values of the hyper-parameter.
   * @return Returns a vector of ValueType containing the range.
   */
  std::vector<ValueType> range() {
    return HyperParameterInterface::getRange<ValueType>();
  }
  
  /*!
   * @brief Convert parameter internals into a description as single XML element.
   * @return Returns a TiXmlElement with the parameter description.
   */
  TiXmlElement* toXml(bool simplified=false) {
    return HyperParameterInterface::toXmlElement<ValueType>(simplified);
  }

  /*!
   * @brief Parse an XML element description to extract parameter values.
   * @param element The TiXmlElement to extract the parameter values from.
   */
  void fromXml(TiXmlElement* element) {
    HyperParameterInterface::fromXmlElement<ValueType>(element);
  }

  /*!
   * @brief Convert parameter internals into a description as single XML element string.
   * @return Returns a std::string with the parameter description.
   */
  std::string toXmlStr(bool simplified=false) {
    return HyperParameterInterface::toXmlElementStr<ValueType>(simplified);
  }
};

} // namespace hyperparam
} // namespace noesis

/*
 * HyperParameter stream operators
 */

template <typename ValueType_>
std::enable_if_t<std::is_same<ValueType_, bool>::value, std::ostream>&
operator<<(std::ostream& os, noesis::hyperparam::HyperParameter<ValueType_>& m) {
  return os << m.getName() << ": "<< std::boolalpha << static_cast<ValueType_>(m);
}

template <typename ValueType_>
std::enable_if_t<noesis::hyperparam::internal::is_supported_nonbool_scalar<ValueType_>::value, std::ostream>&
operator<<(std::ostream& os, noesis::hyperparam::HyperParameter<ValueType_>& m) {
  return os << m.getName() << ": "<< static_cast<ValueType_>(m);
}

template <typename ValueType_>
std::enable_if_t<std::is_same<ValueType_, std::vector<bool>>::value, std::ostream>&
operator<<(std::ostream& os, noesis::hyperparam::HyperParameter<ValueType_>& m) {
  std::stringstream ss;
  ss << "[";
  for (auto value: static_cast<ValueType_>(m)) {
    ss << std::boolalpha << value << ", ";
  }
  std::string out = ss.str();
  out.pop_back();
  out.pop_back();
  out += "]";
  return os << m.getName() << ": "<< out;
}

template <typename ValueType_>
std::enable_if_t<noesis::hyperparam::internal::is_supported_numeric_vector<ValueType_>::value, std::ostream>&
operator<<(std::ostream& os, noesis::hyperparam::HyperParameter<ValueType_>& m) {
  std::string out = "[";
  for (auto& value : static_cast<ValueType_>(m))
    out += std::to_string(value) + ", ";
  out.pop_back();
  out.pop_back();
  out += "]";
  return os << m.getName() << ": "<< out;
}

template <typename ValueType_>
std::enable_if_t<std::is_same<ValueType_, std::vector<std::string>>::value, std::ostream>&
operator<<(std::ostream& os, noesis::hyperparam::HyperParameter<ValueType_>& m) {
  std::string out = "[";
  for (auto& value : static_cast<ValueType_>(m))
    out += value + ", ";
  out.pop_back();
  out.pop_back();
  out += "]";
  return os << m.getName() << ": "<< out;
}

#endif // NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETER_HPP_

/* EOF */
