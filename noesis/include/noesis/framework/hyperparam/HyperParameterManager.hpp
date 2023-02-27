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
#ifndef NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETER_MANAGER_HPP_
#define NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETER_MANAGER_HPP_

// C/C++
#include <unordered_map>

// Noesis
#include "noesis/framework/hyperparam/HyperParameter.hpp"
#include "noesis/framework/hyperparam/HyperParameterTuple.hpp"

namespace noesis {
namespace hyperparam {

/*!
 * @brief The Hyper-Parameter Manger class.
 *
 * This class provides an object which can manage parameters defined by other modules.
 * Parameters are defined and constructed in a distributed manner, and then added to
 * the manager so that configurations can be set centrally.
 *
 * TODO: Describe the get/set API
 * TODO: Describe the load/store API
 */
class HyperParameterManager
{
public:
  //! Default constructor
  HyperParameterManager() = default;

  //! Default destructor
  ~HyperParameterManager() = default;

  /*!
   * @brief Adds a new parameter to be managed.
   * @param parameter The parameter to be added via an generic interface type.
   */
  void addParameter(internal::HyperParameterInterface& parameter);
  
  /*!
   * @brief Remove the parameter tuple from the manager.
   * @param parameter The parameter tuple to be removed.
   */
  void removeParameter(const internal::HyperParameterInterface& parameter);
  
  /*!
   * @brief Adds all parameters in a hyper-parameter tuple to be managed.
   * @param parameter The parameter tuple to be added.
   */
  void addParameterTuple(const HyperParameterTuple& tuple);
  
  /*!
   * @brief Remove the parameter tuple from the manger.
   * @param tuple The parameter to be removed.
   */
  void removeParameterTuple(const HyperParameterTuple& tuple);
  
  /*!
   * @brief Remove the parameter tuple by name.
   * @param name The (scoped) name of the parameter tuple to be removed.
   */
  void removeParameterTuple(const std::string& name);
  
  /*!
   * @brief Checks if a hyper-parameter is managed.
   * @note The look-up is peformed using the scoped-name of the hyper-parameter.
   * @param parameter A reference to the interface of the hyper-parameter.
   * @return True if the hyper-parameter is managed by the manager.
   */
  bool exists(const internal::HyperParameterInterface& parameter) const;
  
  /*!
   * @brief Checks if a hyper-parameter tuple is managed.
   * @note The look-up is peformed using the scoped-name of the tuple.
   * @param tuple A reference to the hyper-parameter tuple instance to be checked.
   * @return True if the hyper-parameter tuple is managed by the manager.
   */
  bool exists(const HyperParameterTuple& tuple) const;
  
  /*!
   * @brief Retrieves a handle to a parameter currently known by the manager.
   * @tparam ValueType_ The data type of the target parameter.
   * @param name The (scoped) name of the parameter to be retrieved.
   * @return Returns a HyperParameter instance using the same underlying storage as the oringial instantiation.
   */
  template<typename ValueType_>
  HyperParameter<ValueType_> getParameter(const std::string& name) {
    boost::shared_lock<boost::shared_mutex> lock(mutex_);
    std::shared_ptr<internal::ParameterValue<ValueType_>> valuePtr;
    bool found = false;
    for (auto& param: parameters_) {
      if (param.first.getName() == name) {
        found = true;
        const std::string val_type = internal::ParameterTypeMap().getTypeFromTypeInfo(typeid(ValueType_).name());
        NFATAL_IF(val_type != param.first.getType(), "[HyperParameterManager]: HyperParameter is of type '"
          << param.first.getType() << "', not '" << val_type << "'.");
        valuePtr = param.first.getValuePtr<ValueType_>();
      }
    }
    NFATAL_IF(!found, "[HyperParameterManager]: HyperParameter '" << name << "' is not managed.");
    return HyperParameter<ValueType_>(valuePtr, name);
  }
  
  /*!
   * @brief Retrieves the value held currently by a specific parameter.
   * @tparam ValueType_ The data type of the target parameter.
   * @param name The (scoped) name of the parameter to be retrieved.
   * @param value The value held currently by the parameter.
   */
  template<typename ValueType_>
  ValueType_ getParameterValue(const std::string& name) const {
    ValueType_ value;
    boost::shared_lock<boost::shared_mutex> lock(mutex_);
    bool found = false;
    for (auto& param: parameters_) {
      if (param.first.getName() == name) {
        found = true;
        const std::string val_type = internal::ParameterTypeMap().getTypeFromTypeInfo(typeid(ValueType_).name());
        NFATAL_IF(val_type != param.first.getType(), "[HyperParameterManager]: HyperParameter is of type '"
                                                        << param.first.getType() << "', not '" << val_type << "'.");
        value = param.first.getValue<ValueType_>();
      }
    }
    NFATAL_IF(!found, "[HyperParameterManager]: HyperParameter '" << name << "' is not managed.");
    return value;
  }

  /*!
   * @brief Sets a new value into a target parameter.
   * @note This function is not thread safe! Make sure it is only called from a single thread.
   * @tparam ValueType_ The data type of the target parameter.
   * @param name The (scoped) name of the parameter to be set.
   * @param value The value to be set.
   */
  template<typename ValueType_>
  void setParameterValue(const std::string& name, const ValueType_& value) {
    bool found = false;
    for (auto& param: parameters_) {
      if (param.first.getName() == name) {
        found = true;
        const std::string val_type = internal::ParameterTypeMap().getTypeFromTypeInfo(typeid(ValueType_).name());
        NFATAL_IF(val_type != param.first.getType(), "[HyperParameterManager]: HyperParameter is of type '"
                                                        << param.first.getType() << "', not '" << val_type << "'.");
        param.first.setValue<ValueType_>(value);
      }
    }
    NFATAL_IF(!found, "[HyperParameterManager]: HyperParameter '" << name << "' is not managed.");
  }

  /*!
   * @brief Outputs all currently managed parameters to an XML file.
   * @param filepath The output file containing a hierarchy of all currently managed parameters.
   */
  void saveParametersToXmlFile(const std::string& filepath, bool simplified=false);

  /*!
   * @brief Outputs all currently managed parameters to an XML element.
   * @param element The output TiXmlElement containing a hierarchy of all currently managed parameters.
   */
  void saveParametersToXmlElement(TiXmlElement* element, bool simplified=false);

  /*!
   * @brief Load parameter values from a provided XML file.
   * @param filepath The file with full path from where parameter values will be parsed.
   */
  void loadParametersFromXmlFile(const std::string& filepath);

  /*!
   * @brief Load parameter values from a provided XML element.
   * @param element The TiXmlElement to be parsed for paramter values.
   */
  void loadParametersFromXmlElement(const TiXmlElement* element);

  /*!
   * @brief Removes any parameters which have gone out-of-scope due to their original owners being destructed.
   * @param verbose Enables more descriptive output for the user.
   */
  void removeUnusedParameters(bool verbose=false);

  /*!
   * @brief Removes all parameters from the manager.
   * @param verbose Enables more descriptive output for the user.
   */
  void removeAllParameters(bool verbose=false);

  /*!
   * @brief Prints the name and value of all parameters currently managed.
   */
  void printParameters();

private:
  /*!
   * @brief Parses an XML element recursively along the the specified name-scope.
   * @param scope The name-scope which guides the parses.
   * @param element The TiXmlElement structure to be parsed.
   */
  void parseXmlElement(const std::vector<std::string>& scope, const TiXmlElement* element);

  /*!
   * @brief Sets the value of a managed parameter by parsing an XML element.
   * @param element The TiXmlElement containing the parameter values to be retrieved.
   * @param parameter The parameter interface which provides access to the managed parameter.
   */
  void setParameterFromXmlElement(const TiXmlElement* element, internal::HyperParameterInterface& parameter);

  /*!
   * @brief Parses the scope of the hyper parameter, converts is to a TiXmlElement and appends it to 'element'.
   * @note This method does not consider hyper parameters coming from hyper parameter tuples.
   *       Use appendHyperParameterTupleToScopedElement instead
   * @param hyperParameter The hyper parameter to add to 'element'
   * @param element The target element to append to.
   */
  void appendHyperParameterToScopedElement(internal::HyperParameterInterface& hyperParameter, TiXmlElement* element, bool simplified=false);

  /*!
   * @brief Parses the scope of the hyper parameter tuple, converts is to a TiXmlElement and appends it to 'element'.
   * @param tuple The hyper parameter tuple to add to 'element'
   * @param element The target element to append to.
   */
  void appendHyperParameterTupleToScopedElement(const HyperParameterTuple& tuple, TiXmlElement* element, bool simplified=false);

private:
  //! @brief The set of simple parameters managed by the manager instance.
  std::list<std::pair<internal::HyperParameterInterface, bool>> parameters_;
  //! @brief The set of (compound) parameter tuples managed by the manager instance.
  std::list<std::pair<HyperParameterTuple, std::string>> tuples_;
  //! @brief A mutex for ensuring thread-safety of manager operations.
  mutable boost::shared_mutex mutex_;
};

} // namespace hyperparam
} // namespace noesis

#endif // NOESIS_FRAMEWORK_HYPERPARAM_HYPER_PARAMETER_MANAGER_HPP_

/* EOF */
