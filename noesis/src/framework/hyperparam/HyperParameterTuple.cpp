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

#include "noesis/framework/hyperparam/HyperParameterTuple.hpp"

namespace noesis {
namespace hyperparam {

HyperParameterTuple::HyperParameterTuple(std::string scope, std::string name, std::string category, std::vector<std::string> range):
  range_(std::move(range)),
  scope_(std::move(scope)),
  name_(std::move(name)),
  category_(std::move(category)),
  type_(std::make_shared<std::string>()),
  parameters_(std::make_shared<TupleType>())
{
}

HyperParameterTuple::HyperParameterTuple(std::string scope, std::string name, std::string category):
  range_(),
  scope_(std::move(scope)),
  name_(std::move(name)),
  category_(std::move(category)),
  type_(std::make_shared<std::string>()),
  parameters_(std::make_shared<TupleType>())
{
}

const std::string& HyperParameterTuple::getScope() const {
  return scope_;
}

const std::string& HyperParameterTuple::getName() const {
  return name_;
}

const std::string& HyperParameterTuple::getCategory() const {
  return category_;
}

std::string HyperParameterTuple::namescope() const {
  return utils::make_namescope({scope_, category_, name_});
}

std::string HyperParameterTuple::getType() const {
  return *type_;
}

void HyperParameterTuple::setType(const std::string& type) {
  NFATAL_IF(!this->checkType(type), "Invalid or unsupported optimizer type '" << type << "'.");
  *type_ = type;
}

bool HyperParameterTuple::exists(const std::string& name) const {
  bool result = false;
  for (const auto& param: *parameters_) {
    if (name == param.first) {
      result = true;
    }
  }
  return result;
}

size_t HyperParameterTuple::size() const {
  return parameters_->size();
}

void HyperParameterTuple::clear() {
  parameters_->clear();
}

const HyperParameterTuple::TupleType& HyperParameterTuple::get() const {
  return *parameters_;
}

void HyperParameterTuple::fromXml(const TiXmlElement& element) {
  NFATAL_IF(this->getName() != element.Attribute("name"),
            "[" << this->namescope() << "]: XML element does not contain tuple name.");
  for (auto* e = element.FirstChildElement(); e != nullptr; e = e->NextSiblingElement()) {
    std::string name = e->Value();
    std::string type = e->Attribute("type");
    for (auto &param: *parameters_) {
      if (name == param.first && type == param.second->getType()) {
        if (type == "bool") {
          param.second->setValue<bool>(noesis::utils::xml::readValueFromXmlElement<bool>(e));
        } else if (type == "int") {
          param.second->setValue<int>(utils::xml::readValueFromXmlElement<int>(e));
        } else if (type == "float") {
          param.second->setValue<float>(utils::xml::readValueFromXmlElement<float>(e));
        } else if (type == "double") {
          param.second->setValue<double>(utils::xml::readValueFromXmlElement<double>(e));
        } else if (type == "string") {
          param.second->setValue<std::string>(utils::xml::readValueFromXmlElement<std::string>(e));
        } else if (type == "bools") {
          param.second->setValue<std::vector<bool>>(utils::xml::readValuesFromXmlElement<bool>(e));
        } else if (type == "ints") {
          param.second->setValue<std::vector<int>>(utils::xml::readValuesFromXmlElement<int>(e));
        } else if (type == "floats") {
          param.second->setValue<std::vector<float>>(utils::xml::readValuesFromXmlElement<float>(e));
        } else if (type == "doubles") {
          param.second->setValue<std::vector<double>>(utils::xml::readValuesFromXmlElement<double>(e));
        } else if (type == "strings") {
          param.second->setValue<std::vector<std::string>>(utils::xml::readValuesFromXmlElement<std::string>(e));
        }
      }
    }
  }
}

TiXmlElement* HyperParameterTuple::toXml(bool simplified) const {
  TiXmlElement* element = nullptr;
  element = new TiXmlElement(category_);
  element->SetAttribute("name", name_);
  element->SetAttribute("type", *type_);
  for (const auto& param: *parameters_) {
    std::string type = param.second->getType();
    std::string name = param.first;
    if (type == "bool") {
      element->LinkEndChild(param.second->toXmlElement<bool>(simplified));
    } else if (type == "int") {
      element->LinkEndChild(param.second->toXmlElement<int>(simplified));
    } else if (type == "float") {
      element->LinkEndChild(param.second->toXmlElement<float>(simplified));
    } else if (type == "double") {
      element->LinkEndChild(param.second->toXmlElement<double>(simplified));
    } else if (type == "string") {
      element->LinkEndChild(param.second->toXmlElement<std::string>(simplified));
    }  else if (type == "bools") {
      element->LinkEndChild(param.second->toXmlElement<std::vector<bool>>(simplified));
    }  else if (type == "ints") {
      element->LinkEndChild(param.second->toXmlElement<std::vector<int>>(simplified));
    }  else if (type == "floats") {
      element->LinkEndChild(param.second->toXmlElement<std::vector<float>>(simplified));
    }  else if (type == "doubles") {
      element->LinkEndChild(param.second->toXmlElement<std::vector<double>>(simplified));
    } else if (type == "strings") {
      element->LinkEndChild(param.second->toXmlElement<std::vector<std::string>>(simplified));
    } else {
      auto tupleName = this->namescope();
      NFATAL("[" << tupleName << "]: Cannot convert parameter '" << name << "' to XML. Type '" << type << "' is not supported.");
    }
  }
  return element;
}

std::string HyperParameterTuple::toXmlStr(bool simplified) {
  TiXmlPrinter printer;
  printer.SetIndent("  ");
  TiXmlHandle handle = this->toXml(simplified);
  handle.ToNode()->Accept(&printer);
  return printer.Str();
}

void HyperParameterTuple::createParametersFromXml(const TiXmlElement& element) {
  category_ = element.Value();
  *type_ = element.Attribute("type");
  // Iterate over all nodes and construct parameter for each one
  for (auto* e = element.FirstChildElement(); e != nullptr; e = e->NextSiblingElement()) {
    std::string name = e->Value();
    NFATAL_IF(exists(name), "[" << this->namescope() << "]: '" << name << "' already exists in this tuple.");
    std::string type = e->Attribute("type");
    if (type == "bool") {
      this->addParameter<bool>(utils::xml::readValueFromXmlElement<bool>(e), name);
    } else if (type == "int") {
      this->addParameter<int>(utils::xml::readValueFromXmlElement<int>(e), name,
        utils::xml::readRangeFromXmlElement<int>(e));
    } else if (type == "float") {
      this->addParameter<float>(utils::xml::readValueFromXmlElement<float>(e), name,
        utils::xml::readRangeFromXmlElement<float>(e));
    } else if (type == "double") {
      this->addParameter<double>(utils::xml::readValueFromXmlElement<double>(e), name,
        utils::xml::readRangeFromXmlElement<double>(e));
    } else if (type == "string") {
      this->addParameter<std::string>(utils::xml::readValueFromXmlElement<std::string>(e), name,
        utils::xml::readRangeFromXmlElement<std::string>(e));
    } else if (type == "bools") {
      this->addParameter<std::vector<bool>>(utils::xml::readValuesFromXmlElement<bool>(e), name);
    } else if (type == "ints") {
      this->addParameter<std::vector<int>>(utils::xml::readValuesFromXmlElement<int>(e), name,
        utils::xml::readRangesFromXmlElement<int>(e));
    } else if (type == "floats") {
      this->addParameter<std::vector<float>>(utils::xml::readValuesFromXmlElement<float>(e), name,
        utils::xml::readRangesFromXmlElement<float>(e));
    } else if (type == "doubles") {
      this->addParameter<std::vector<double>>(utils::xml::readValuesFromXmlElement<double>(e), name,
        utils::xml::readRangesFromXmlElement<double>(e));
    } else if (type == "strings") {
      this->addParameter<std::vector<std::string>>(utils::xml::readValuesFromXmlElement<std::string>(e), name,
        utils::xml::readRangesFromXmlElement<std::string>(e));
    } else {
      auto tupleName = this->namescope();
      NFATAL("[" << tupleName << "]: Cannot import parameter '" << name << "' from XML element. Type '" << type << "' is not supported.");
    }
  }
}

bool HyperParameterTuple::checkType(const std::string& type) const {
  if (range_.empty()) {
    return true;
  }
  for (auto& element: range_) {
    if (element == type) {
      return true;
    }
  }
  return false;
}

} // namespace hyperparam
} // namespace noesis

/*!
 * @brief HyperParameterTuple: stream operator
 * @param os Target output stream to output the tuple description.
 * @param tuple The source hyper-parameter tuple.
 * @return The augmented output stream.
 */
std::ostream& operator<<(std::ostream& os, const noesis::hyperparam::HyperParameterTuple& tuple) {
  os << tuple.namescope() << ": " << "\n";
  for (auto& param: tuple.get()) {
    auto type = param.second->getType();
    if (type == "bool") {
      param.second->print<bool>(os);
    } else if (type == "int") {
      param.second->print<int>(os);
    } else if (type == "float") {
      param.second->print<float>(os);
    } else if (type == "double") {
      param.second->print<double>(os);
    } else if (type == "string") {
      param.second->print<std::string>(os);
    } else if (type == "bools") {
      param.second->print<std::vector<bool>>(os);
    } else if (type == "ints") {
      param.second->print<std::vector<int>>(os);
    } else if (type == "floats") {
      param.second->print<std::vector<float>>(os);
    } else if (type == "doubles") {
      param.second->print<std::vector<double>>(os);
    } else if (type == "strings") {
      param.second->print<std::vector<std::string>>(os);
    }
  }
  return os;
}

/* EOF */
