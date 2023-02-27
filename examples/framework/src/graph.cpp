/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */

// Boost
#include <boost/filesystem.hpp>

// Noesis
#include <noesis/noesis.hpp>
#include <noesis/framework/core/Graph.hpp>

int main(int argc, char** argv)
{
  noesis::init("noesis_graph_example");
  
  // Define paths to the files
  std::string srcFile = noesis::rootpath() + "/examples/framework/python/graph.py";
  std::string buildFile = noesis::logpath() + "/graph/graph.py";
  std::string protoFile = noesis::logpath() + "/graph/protos/graph.pb";
  
  // We copy the builder file into the current
  // log directory and call it from there
  boost::filesystem::create_directories(noesis::logpath() + "/graph");
  boost::filesystem::copy(srcFile, buildFile);
  
  // Step 1: Create an instance of a C++ graph front-end
  auto graph = std::make_unique<noesis::core::Graph>();
  
  // Step 2: We execute the Python script which will build the graph
  // and generate the *.pb protobuf file containing the MetaGraphDef
  graph->generateFrom(buildFile);
  
  // Step 3: We load the MetaGraphDef from the generated *.pb file.
  graph->loadFrom(protoFile);
  
  // Step 4: We launch the graph in order to be able to execute operations.
  graph->startup();
  
  // TODO: Step 5: Execute an example operation
  
  // Success
  return 0;
}

/* EOF */
