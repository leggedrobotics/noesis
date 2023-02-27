/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Noesis
#include <noesis/noesis.hpp>
#include <noesis/framework/hyperparam/hyper_parameters.hpp>


void printIntParameter(int value) {
  NINFO("Int parameter value: " << value);
}

void printStringParameter(std::string value) {
  NINFO("String parameter value: " << value);
}

int main(int argc, const char* argv[])
{
  using namespace noesis::hyperparam;
  
  /*
   * NOTE: we do not initialize this process with noesis::init() since we do not want to create any logs. In general, it is recommended
   * to always do a noesis::init() call once to create log directories.
   */
  noesis::init("noesis_hyperparams_example");
  
  /*
   * It is possible to define hyper-parameters from various fundamental C++ data-types as well as STL vector data structure of them. The
   * hyper-parameters can be initialized with any of the noesis::hyperparam::HyperParameter() constructor call. The first argument is typically
   * the default value to which the hyper-parameter is set to.
   */
  NNOTIFY("Defining some parameters: ");
  HyperParameter<bool> myBool(true, "TestAgent/myBool");
  HyperParameter<int> myInt(42, "TestAgent/MyFunc/myInt", {0, 100});
  HyperParameter<std::string> myString("Value0", "TestAgent/MyFunc/myString", {"Value0", "Value1", "Value2"});

  NINFO(myBool);
  NINFO(myInt);
  NINFO(myString);
  
  /*
   * Hyperparameter manager helps setting and getting of parameters which may be globally distributed. It also allows loading and dumping
   * parameters into an XML file. This makes it easy to tune and configure algorithms and environments during training.
   */
  NNOTIFY("Adding parameters to the global manager: ");
  manager->addParameter(myBool);
  manager->addParameter(myInt);
  manager->addParameter(myString);
  manager->printParameters();
  
  // It can later be configured via the noesis::hyperparam::manager by loading XML file or through the overloaded operator '='.
  NNOTIFY("Setting parameters at source scope using operators ...");
  myBool = true;
  myInt = 72;
  myString = "Value2";
  NINFO(myBool);
  NINFO(myInt);
  NINFO(myString);
  manager->printParameters();
  
  NNOTIFY("Accessing parameters at source scope using operators ...");
  bool boolVal = myBool;
  int intVal = myInt;
  std::string strVal = myString;
  NINFO("Bool value: " << boolVal);
  NINFO("Int value: " << intVal);
  NINFO("String value: " << strVal);
  
  NNOTIFY("Setting parameters via the manager ...");
  manager->setParameterValue<bool>("TestAgent/myBool", true);
  manager->setParameterValue<int>("TestAgent/MyFunc/myInt", 47);
  manager->setParameterValue<std::string>("TestAgent/MyFunc/myString", "Value2");
  NINFO(myBool);
  NINFO(myInt);
  NINFO(myString);
  manager->printParameters();
  
  NNOTIFY("Implicit casting ...");
  printIntParameter(myInt);
  printStringParameter(myString);
  
  if (myInt > 50) {
    NWARNING("'myInt' has value greater than 50: " << static_cast<size_t>(myInt));
  } else {
    NWARNING("'myInt' has value: " << static_cast<size_t>(myInt));
  }
  
  NNOTIFY("Full XML output: ");
  NINFO("\n" << myBool.toXmlStr());
  NINFO("\n" << myInt.toXmlStr());
  NINFO("\n" << myString.toXmlStr());

  NNOTIFY("Simplified XML output: ");
  NINFO("\n" << myBool.toXmlStr(true));
  NINFO("\n" << myInt.toXmlStr(true));
  NINFO("\n" << myString.toXmlStr(true));

  // An example on saving the added hyper-parameters into an XML file using hyper-parameter manager
  NNOTIFY("Manager XML export: ");
  TiXmlPrinter printerFull;
  printerFull.SetIndent("  ");
  TiXmlElement parameter_description_full("ExampleParametersFull");
  manager->saveParametersToXmlElement(&parameter_description_full);
  parameter_description_full.Accept(&printerFull);
  NINFO("FULL:\n" << printerFull.Str());
  TiXmlPrinter printerSimple;
  printerSimple.SetIndent("  ");
  TiXmlElement parameter_description_simple("ExampleParametersSimple");
  manager->saveParametersToXmlElement(&parameter_description_simple, true);
  parameter_description_simple.Accept(&printerSimple);
  NINFO("SIMPLE:\n" << printerSimple.Str());
}

/* EOF */
