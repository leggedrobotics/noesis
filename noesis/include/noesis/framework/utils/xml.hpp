/*!
 * @author    David Hoeller
 * @email     dhoeller@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_UTILS_XML_HPP_
#define NOESIS_FRAMEWORK_UTILS_XML_HPP_

// C/C++
#include <string>
#include <vector>

// Boost
#include <boost/lexical_cast.hpp>

// TinyXML
#include <tinyxml.h>

// Noesis
#include "noesis/framework/log/message.hpp"

namespace noesis {
namespace utils {
namespace xml {

/**
 * Overloads to set an attribute of specific type
 */

template<typename ValueType_>
inline void setXmlAttribute(const std::string& name, ValueType_ value, TiXmlElement* element) {
  element->SetAttribute(name, value);
}

template <>
inline void setXmlAttribute<float>(const std::string& name, float value, TiXmlElement* element) {
  element->SetDoubleAttribute(name, value);
}

template <>
inline void setXmlAttribute<double>(const std::string& name, double value, TiXmlElement* element) {
  element->SetDoubleAttribute(name, value);
}

template <>
inline void setXmlAttribute<bool>(const std::string& name, bool value, TiXmlElement* element) {
  if (value) {
    element->SetAttribute(name, "true");
  } else {
    element->SetAttribute(name, "false");
  }
}

/**
 * Parsing values and ranges
 */

template<typename ValueType_>
inline ValueType_ readValueFromXmlElement(const TiXmlElement* element) {
  ValueType_ value;
  if (element->QueryValueAttribute("value", &value) != TIXML_SUCCESS) {
    NFATAL("Could not find the attribute value in the xml element.");
  }
  return value;
}

template<>
inline std::string readValueFromXmlElement(const TiXmlElement* element) {
  std::string value;
  if (element->QueryStringAttribute("value", &value) != TIXML_SUCCESS) {
    NFATAL("Could not find the attribute value in the xml element.");
  }
  return value;
}

template<>
inline bool readValueFromXmlElement(const TiXmlElement* element) {
  bool value;
  if (element->QueryBoolAttribute("value", &value) != TIXML_SUCCESS) {
    NFATAL("Could not find the attribute value in the xml element.");
  }
  return value;
}

template<typename ValueType_>
inline std::vector<ValueType_> readValuesFromXmlElement(const TiXmlElement* element) {
  std::vector<ValueType_> values;
  for (const TiXmlElement *e = element->FirstChildElement(); e != nullptr; e = e->NextSiblingElement()) {
    values.push_back(readValueFromXmlElement<ValueType_>(e));
  }
  return values;
}

template<typename ValueType_>
inline void parseNumericRange(const std::string& range_string, std::vector<ValueType_>& range) {
  // Convert the space separated range into a vector of strings containing each range element
  std::istringstream ss{range_string};
  std::vector<std::string> rangeVec{std::istream_iterator<std::string>{ss}, std::istream_iterator<std::string>{}};
  
  if (rangeVec.size() == 2) {
    range.push_back(std::numeric_limits<ValueType_>::lowest());
    range.push_back(std::numeric_limits<ValueType_>::max());
    if (rangeVec.front() == "eps") {
      range.front() = std::numeric_limits<ValueType_>::min();
    }
    if (rangeVec.front() != "min" && rangeVec.front() != "eps") {
      range.front() = boost::lexical_cast<ValueType_>(rangeVec.front());
    }
    if (rangeVec.back() != "max") {
      range.back() = boost::lexical_cast<ValueType_>(rangeVec.back());
    }
  } else {
    for (auto& element: rangeVec) {
      range.push_back(boost::lexical_cast<ValueType_>(element));
    }
  }
}

template<typename ValueType_>
inline std::vector<ValueType_> readRangeFromXmlElement(const TiXmlElement* element) {
  std::vector<ValueType_> ranges;
  std::string rangeString;
  if (element->QueryStringAttribute("range", &rangeString) != TIXML_SUCCESS) {
    NFATAL("Could not find the attribute range in the xml element.");
  }
  std::vector<ValueType_> range;
  parseNumericRange(rangeString, range);
  return range;
}

template<>
inline std::vector<std::string> readRangeFromXmlElement(const TiXmlElement* element) {
  std::string rangeString;
  if (element->QueryStringAttribute("range", &rangeString) != TIXML_SUCCESS) {
    NFATAL("Could not find the attribute range in the xml element.");
  }
  std::istringstream ss{rangeString};
  std::vector<std::string> range{std::istream_iterator<std::string>{ss}, std::istream_iterator<std::string>{}};
  return range;
}


template<typename ValueType_>
inline std::vector<std::vector<ValueType_>> readRangesFromXmlElement(const TiXmlElement* element) {
  std::vector<std::vector<ValueType_>> ranges;
  for (const TiXmlElement *e = element->FirstChildElement(); e != nullptr; e = e->NextSiblingElement()) {
    ranges.push_back(readRangeFromXmlElement<ValueType_>(e));
  }
  return ranges;
}

//! Print a node and its children in xml format
inline void printNode(TiXmlNode* node, int depth=0) {
  if (!node) return;
  std::string blank(static_cast<size_t>(2*depth), ' ');
  switch (node->Type()) {
    case TiXmlNode::TINYXML_ELEMENT: {
      std::cout << blank << "<" << node->Value();
      TiXmlAttribute *attribute = dynamic_cast<TiXmlElement *>(node)->FirstAttribute();
      while (attribute) {
        std::cout  << " " << attribute->Name() << "=\"" << attribute->Value() << "\"";
        attribute = attribute->Next();
      }
      if (node->FirstChild() != nullptr) {
        std::cout << ">" << std::endl;
        for (TiXmlNode* child = node->FirstChild(); child != nullptr; child = child->NextSibling()) {
          printNode(child, depth+1);
        }
        std::cout << blank << "</" << node->Value() << ">" << std::endl;
      } else {
        std::cout << " /> " << std::endl;
      }
      break;
    }
    case TiXmlNode::TINYXML_TEXT: {
      TiXmlText *text = node->ToText();
      std::cout << blank << text->Value() << std::endl;
      break;
    }
    default:
      break;
  }
}

} // namespace xml
} // namespace utils
} // namespace noesis

#endif // NOESIS_FRAMEWORK_UTILS_XML_HPP_

/* EOF */
