/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// C/C++
#include <cstdlib>
#include <fstream>

// Noesis
#include <noesis/framework/hyperparam/HyperParameterManager.hpp>

namespace noesis {
namespace hyperparam {

void HyperParameterManager::addParameter(internal::HyperParameterInterface& parameter) {
  boost::unique_lock<boost::shared_mutex> lock(mutex_);
  // Check if the parameter has already been added
  bool exists = false;
  for (auto& param: parameters_) {
    if (parameter.getName() == param.first.getName()) {
      exists = true;
      auto type = parameter.getType();
      if (type == "bool") {
        parameter.setValuePtr<bool>(param.first.getValuePtr<bool>());
      } else if (type == "int") {
        parameter.setValuePtr<int>(param.first.getValuePtr<int>());
      } else if (type == "float") {
        parameter.setValuePtr<float>(param.first.getValuePtr<float>());
      } else if (type == "double") {
        parameter.setValuePtr<double>(param.first.getValuePtr<double>());
      } else if (type == "string") {
        parameter.setValuePtr<std::string>(param.first.getValuePtr<std::string>());
      } else if (type == "bools") {
        parameter.setValuePtr<std::vector<bool>>(param.first.getValuePtr<std::vector<bool>>());
      } else if (type == "ints") {
        parameter.setValuePtr<std::vector<int>>(param.first.getValuePtr<std::vector<int>>());
      } else if (type == "floats") {
        parameter.setValuePtr<std::vector<float>>(param.first.getValuePtr<std::vector<float>>());
      } else if (type == "doubles") {
        parameter.setValuePtr<std::vector<double>>(param.first.getValuePtr<std::vector<double>>());
      } else if (type == "strings") {
        parameter.setValuePtr<std::vector<std::string>>(param.first.getValuePtr<std::vector<std::string>>());
      } else {
        NFATAL("[HyperParameterManager]: Unsupported type: '" << type << "' in parameter '" << parameter.getName() << "'. Cannot attach!");
      }
      break;
    }
  }
  // Add the parameter to the manager only if it does not already exist
  if (!exists) {
    parameters_.emplace_back(parameter, true);
  }
}

void HyperParameterManager::removeParameter(const internal::HyperParameterInterface& parameter) {
  boost::unique_lock<boost::shared_mutex> lock(mutex_);
  bool found = false;
  for (auto it = parameters_.begin(); it != parameters_.end(); it++) {
    if (it->first.getName() == parameter.getName()) {
      found = true;
      it = parameters_.erase(it);
      break;
    }
  }
  NWARNING_IF(!found, "[HyperParameterManager]: HyperParameter '" << parameter.getName() << "' has not been added to manager.");
}

bool HyperParameterManager::exists(const internal::HyperParameterInterface& parameter) const {
  boost::shared_lock<boost::shared_mutex> lock(mutex_);
  bool found = false;
  for (auto& param: parameters_) {
    if (param.first.getName() == parameter.getName()) {
      found = true;
      break;
    }
  }
  return found;
}

bool HyperParameterManager::exists(const HyperParameterTuple& tuple) const {
  boost::shared_lock<boost::shared_mutex> lock(mutex_);
  bool found = false;
  for (auto& tup: tuples_) {
    if (tup.first.getName() == tuple.getName()) {
      found = true;
      break;
    }
  }
  return found;
}

void HyperParameterManager::addParameterTuple(const HyperParameterTuple& tuple) {
  boost::unique_lock<boost::shared_mutex> lock(mutex_);
  // Check if the parameter has already been added
  for (auto& parameter: parameters_) {
    for (auto& param: tuple.get()) {
      NFATAL_IF((param.second->getName() == parameter.first.getName()),
                "[HyperParameterManager]: HyperParameter '" << param.second->getName() << "' has already been added to the manager.");
    }
  }
  // Keep track of the tuples that were added
  tuples_.emplace_back(tuple, tuple.namescope());
  // Add the parameters of the tuple to the manager
  for (auto& param: tuple.get()) {
    parameters_.emplace_back(*param.second, false);
  }
}

void HyperParameterManager::removeParameterTuple(const HyperParameterTuple& tuple) {
  removeParameterTuple(tuple.namescope());
}

void HyperParameterManager::removeParameterTuple(const std::string& name) {
  boost::unique_lock<boost::shared_mutex> lock(mutex_);
  bool found = false;
  // Find the location of the tuple in the list
  auto iter = tuples_.end();
  for (auto it = tuples_.begin(); it != tuples_.end(); it++) {
    if (it->second == name) {
      iter = it;
      found = true;
      break;
    }
  }
  NFATAL_IF(!found, "[HyperParameterManager]: HyperParameterTuple '" << name << "' has not been added to manager.");
  // Remove parameters belonging to the tuple
  for (auto& param: iter->first.get()) {
    for (auto it = parameters_.begin(); it != parameters_.end(); it++) {
      if (it->first.getName() == param.second->getName()) {
        it = parameters_.erase(it);
        break;
      }
    }
  }
  // Erase the tuple from the list
  for (auto it = tuples_.begin(); it != tuples_.end(); it++) {
    if (it->second == name) {
      it = tuples_.erase(it);
      break;
    }
  }
}

void HyperParameterManager::saveParametersToXmlFile(const std::string& filepath, bool simplified) {
  auto* declaration = new TiXmlDeclaration("1.0", "UTF-8", "");
  auto* element = new TiXmlElement("HyperParameters");
  saveParametersToXmlElement(element, simplified);
  auto* document = new TiXmlDocument;
  document->LinkEndChild(declaration);
  document->LinkEndChild(element);
  document->SaveFile(filepath);
  delete document;
}

void HyperParameterManager::saveParametersToXmlElement(TiXmlElement* element, bool simplified) {
  boost::shared_lock<boost::shared_mutex> lock(mutex_);
  NFATAL_IF((parameters_.empty()), "[HyperParameterManager]: There are no parameters to save.");
  // Iterate over all parameters and create their individual nodes
  for (auto& tuple: tuples_) {
    appendHyperParameterTupleToScopedElement(tuple.first, element, simplified);
  }
  for (auto& param: parameters_) {
    // Check that the parameter is standalone (i.e. not from a tuple)
    if (param.second) {
      appendHyperParameterToScopedElement(param.first, element, simplified);
    }
  }
}

void HyperParameterManager::loadParametersFromXmlFile(const std::string& filepath) {
  NFATAL_IF(!boost::filesystem::exists(filepath), "[HyperParameterManager]: File not found: " << filepath);
  boost::unique_lock<boost::shared_mutex> lock(mutex_);
  // Load the parameter file from disk
  TiXmlDocument document(filepath);
  document.LoadFile(TiXmlEncoding::TIXML_ENCODING_UTF8);
  TiXmlHandle handle(&document);
  std::string rootElementName = handle.FirstChildElement().ToElement()->Value();
  NFATAL_IF(rootElementName != "HyperParameters",
    "[HyperParameterManager]: The root XML element must be <HyperParameters> ... </HyperParameters>");
  parseXmlElement({}, handle.FirstChildElement().ToElement());
}

void HyperParameterManager::loadParametersFromXmlElement(const TiXmlElement* element) {
  boost::unique_lock<boost::shared_mutex> lock(mutex_);
  // Check for the by-convention root element name
  std::string rootElementName = element->Value();
  NFATAL_IF(rootElementName != "HyperParameters",
    "[HyperParameterManager]: The root XML element must be <HyperParameters> ... </HyperParameters>");
  // Iterate over top-level elements in the XML file and progress recursively through each
  for (auto* e = element->FirstChildElement(); e != nullptr; e = e->NextSiblingElement()) {
    std::vector<std::string> scope;
    scope.emplace_back(e->Value());
    parseXmlElement(scope, e);
  }
}

void HyperParameterManager::removeUnusedParameters(bool verbose) {
  boost::unique_lock<boost::shared_mutex> lock(mutex_);
  if (parameters_.empty()) {
    NWARNING("[HyperParameterManager]: Cannot remove unused parameters. There are no parameters to remove ...");
    return;
  }
  for (auto it = parameters_.begin(); it != parameters_.end();) {
    bool used = true;
    // Retrieve use count by type
    auto type = it->first.getType();
    if (type == "bool") {
      used = it->first.getValuePtr<bool>()->isUsed();
    } else if (type == "int") {
      used = it->first.getValuePtr<int>()->isUsed();
    } else if (type == "float") {
      used = it->first.getValuePtr<float>()->isUsed();
    } else if (type == "double") {
      used = it->first.getValuePtr<double>()->isUsed();
    } else if (type == "string") {
      used = it->first.getValuePtr<std::string>()->isUsed();
    } else if (type == "bools") {
      used = it->first.getValuePtr<std::vector<bool>>()->isUsed();
    } else if (type == "ints") {
      used = it->first.getValuePtr<std::vector<int>>()->isUsed();
    } else if (type == "floats") {
      used = it->first.getValuePtr<std::vector<float>>()->isUsed();
    } else if (type == "doubles") {
      used = it->first.getValuePtr<std::vector<double>>()->isUsed();
    } else if (type == "strings") {
      used = it->first.getValuePtr<std::vector<std::string>>()->isUsed();
    } else {
      NFATAL("[HyperParameterManager]: Unsupported type: '" << type << "' in parameter '" << it->first.getName() << "'. Cannot remove.");
    }
    // Check parameter usage and remove if necessary
    if (!used) {
      if (verbose) NINFO("[HyperParameterManager]: Removing parameter: " << it->first.getName());
      it = parameters_.erase(it);
    } else {
      it++;
    }
  }
}

void HyperParameterManager::removeAllParameters(bool verbose) {
  boost::unique_lock<boost::shared_mutex> lock(mutex_);
  if (parameters_.empty()) {
    NWARNING("[HyperParameterManager]: Cannot remove parameters. There are no parameters to remove ...");
    return;
  }
  NINFO_IF(verbose, "[HyperParameterManager]: Removing all parameters ...");
  parameters_.clear();
}

void HyperParameterManager::printParameters() {
  boost::shared_lock<boost::shared_mutex> lock(mutex_);
  if (parameters_.empty()) {
    NWARNING("[HyperParameterManager]: There are no parameters to print ...");
    return;
  }
  for (auto& param: parameters_) {
    auto type = param.first.getType();
    if (type == "bool") {
      param.first.print<bool>();
    } else if (type == "int") {
      param.first.print<int>();
    } else if (type == "float") {
      param.first.print<float>();
    } else if (type == "double") {
      param.first.print<double>();
    } else if (type == "string") {
      param.first.print<std::string>();
    } else if (type == "bools") {
      param.first.print<std::vector<bool>>();
    } else if (type == "ints") {
      param.first.print<std::vector<int>>();
    } else if (type == "floats") {
      param.first.print<std::vector<float>>();
    } else if (type == "doubles") {
      param.first.print<std::vector<double>>();
    } else if (type == "strings") {
      param.first.print<std::vector<std::string>>();
    } else {
      NFATAL("[HyperParameterManager]: Unsupported type: '" << type << "'");
    }
  }
}

void HyperParameterManager::parseXmlElement(const std::vector<std::string>& scope, const TiXmlElement* element) {
  if ((element->Attribute("value") != nullptr) && (element->Attribute("type") != nullptr)) {
    // Parameters have the "type" and "value" attributes
    std::string name = utils::make_namescope(scope);
    // Check if parameter exits
    bool found = false;
    for (auto& param: parameters_) {
      if (param.first.getName() == name) {
        // Check parameter type
        const std::string type = element->Attribute("type");
        NFATAL_IF((param.first.getType() != type), "[HyperParameterManager]: HyperParameter '"
          << name << "' is of type '" << param.first.getType() << "' not '" << type << "'.");
        setParameterFromXmlElement(element, param.first);
        found = true;
        break;
      }
    }
    NWARNING_IF(!found, "[HyperParameterManager]: HyperParameter '" << name << "' does not exist.");
  } else if (element->Attribute("type") != nullptr && element->Attribute("name") != nullptr) {
    // Parameter tuples have the "type" and "name" attributes
    // Check if parameter tuple exits
    std::string name = utils::make_namescope(scope) + "/" + element->Attribute("name");
    bool found = false;
    for (auto& tuple: tuples_) {
      if (tuple.first.namescope() == name) {
        if (tuple.first.getType() != element->Attribute("type")) {
          for (auto& param: tuple.first.get()) {
            for (auto it = parameters_.begin(); it != parameters_.end(); it++) {
              if (it->first.getName() == param.second->getName()) {
                it = parameters_.erase(it);
                break;
              }
            }
          }
          // Clear all current contents and re-create from the XML
          tuple.first.clear();
          tuple.first.setType(element->Attribute("type"));
          tuple.first.createParametersFromXml(*element);
          for (auto& param: tuple.first.get()) {
            parameters_.emplace_back(*param.second, false);
          }
        } else {
          tuple.first.fromXml(*element);
        }
        found = true;
        break;
      }
    }
    NWARNING_IF(!found, "[HyperParameterManager]: HyperParameterTuple '" << name << "' does not exist. No action taken.");
  } else {
    // Non-param node - proceed recursively over all child nodes of the current
    for (auto* e = element->FirstChildElement(); e != nullptr; e = e->NextSiblingElement()) {
      std::vector<std::string> subscope = scope;
      subscope.emplace_back(e->Value());
      parseXmlElement(subscope, e);
    }
  }
}

void HyperParameterManager::setParameterFromXmlElement(const TiXmlElement* element, internal::HyperParameterInterface& parameter) {
  if (strcmp(element->Attribute("type"), "bool") == 0) {
    this->setParameterValue<bool>(parameter.getName(), utils::xml::readValueFromXmlElement<bool>(element));
  } else if (strcmp(element->Attribute("type"), "int") == 0) {
    this->setParameterValue<int>(parameter.getName(), utils::xml::readValueFromXmlElement<int>(element));
  } else if (strcmp(element->Attribute("type"), "float") == 0) {
    this->setParameterValue<float>(parameter.getName(), utils::xml::readValueFromXmlElement<float>(element));
  } else if (strcmp(element->Attribute("type"), "double") == 0) {
    this->setParameterValue<double>(parameter.getName(), utils::xml::readValueFromXmlElement<double>(element));
  } else if (strcmp(element->Attribute("type"), "string") == 0) {
    this->setParameterValue<std::string>(parameter.getName(), utils::xml::readValueFromXmlElement<std::string>(element));
  } else if (strcmp(element->Attribute("type"), "bools") == 0) {
    this->setParameterValue<std::vector<bool>>(parameter.getName(), utils::xml::readValuesFromXmlElement<bool>(element));
  } else if (strcmp(element->Attribute("type"), "ints") == 0) {
    this->setParameterValue<std::vector<int>>(parameter.getName(), utils::xml::readValuesFromXmlElement<int>(element));
  } else if (strcmp(element->Attribute("type"), "floats") == 0) {
    this->setParameterValue<std::vector<float>>(parameter.getName(), utils::xml::readValuesFromXmlElement<float>(element));
  } else if (strcmp(element->Attribute("type"), "doubles") == 0) {
    this->setParameterValue<std::vector<double>>(parameter.getName(), utils::xml::readValuesFromXmlElement<double>(element));
  } else if (strcmp(element->Attribute("type"), "strings") == 0) {
    this->setParameterValue<std::vector<std::string>>(parameter.getName(), utils::xml::readValuesFromXmlElement<std::string>(element));
  } else {
    NFATAL("[HyperParameterManager]: Unknown or unsupported hyper-parameter type: " << element->Value() << "\n");
  }
}

void HyperParameterManager::appendHyperParameterToScopedElement(internal::HyperParameterInterface& hyperParameter, TiXmlElement* element, bool simplified) {
  // Find the corresponding sub-element to append to
  TiXmlElement* currentParent = element;
  std::vector<std::string> scopeVector = hyperParameter.getScope();
  for (std::size_t i = 0; i < scopeVector.size()-1; i++) {
    // Check if the scope already exists, create new element if false
    TiXmlElement* scopedElement = currentParent->FirstChildElement(scopeVector[i]);
    if (scopedElement == nullptr) {
      auto* temp = new TiXmlElement(scopeVector[i]);
      currentParent->LinkEndChild(temp);
      currentParent = temp;
    } else {
      currentParent = scopedElement;
    }
  }
  // Convert the hyper-parameter to an XML element
  auto type = hyperParameter.getType();
  TiXmlElement* hyperParamElement = nullptr;
  if (type == "bool") {
    hyperParamElement = hyperParameter.toXmlElement<bool>(simplified);
  } else if (type == "int") {
    hyperParamElement = hyperParameter.toXmlElement<int>(simplified);
  } else if (type == "float") {
    hyperParamElement = hyperParameter.toXmlElement<float>(simplified);
  } else if (type == "double") {
    hyperParamElement = hyperParameter.toXmlElement<double>(simplified);
  } else if (type == "string") {
    hyperParamElement = hyperParameter.toXmlElement<std::string>(simplified);
  } else if (type == "bools") {
    hyperParamElement = hyperParameter.toXmlElement<std::vector<bool>>(simplified);
  } else if (type == "ints") {
    hyperParamElement = hyperParameter.toXmlElement<std::vector<int>>(simplified);
  } else if (type == "floats") {
    hyperParamElement = hyperParameter.toXmlElement<std::vector<float>>(simplified);
  } else if (type == "doubles") {
    hyperParamElement = hyperParameter.toXmlElement<std::vector<double>>(simplified);
  } else if (type == "strings") {
    hyperParamElement = hyperParameter.toXmlElement<std::vector<std::string>>(simplified);
  } else {
    NFATAL("[HyperParameterManager]: Unsupported type: '" << type << "'");
  }
  currentParent->LinkEndChild(hyperParamElement);
}

void HyperParameterManager::appendHyperParameterTupleToScopedElement(const HyperParameterTuple& tuple, TiXmlElement* element, bool simplified) {
  if (tuple.size() > 0) {
    // Generate the parent elements if they do not exist yet
    std::vector<std::string> scopeVector;
    boost::split(scopeVector, tuple.getScope(), [](char c){return c == '/';});
    TiXmlElement* currentParent = element;
    for (auto& scope: scopeVector) {
      // Check if the scope already exists, create new element if false
      TiXmlElement* scopedElement = currentParent->FirstChildElement(scope);
      if (scopedElement == nullptr) {
        auto* temp = new TiXmlElement(scope);
        currentParent->LinkEndChild(temp);
        currentParent = temp;
      } else {
        currentParent = scopedElement;
      }
    }
    currentParent->LinkEndChild(tuple.toXml(simplified));
  }
}

} // namespace hyperparam
} // namespace noesis

/* EOF */
