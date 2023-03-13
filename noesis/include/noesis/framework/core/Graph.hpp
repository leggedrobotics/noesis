/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_FRAMEWORK_CORE_GRAPH_HPP_
#define NOESIS_FRAMEWORK_CORE_GRAPH_HPP_

// C/C++
#include <memory>
#include <string>
#include <vector>

// TensorFlow
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

// Noesis
#include "noesis/framework/core/Object.hpp"
#include "noesis/framework/core/TensorTuple.hpp"
#include "noesis/framework/system/tensorflow.hpp"

namespace noesis {
namespace core {

/*!
 * @brief Noesis graphs provide easy-to-use generic wrappers for TensorFlow sessions.
 */
class Graph: public ::noesis::core::Object
{
public:
  
  /*!
   * @brief Configuration helper struct for Noesis Graph instances.
   */
  struct Config {
    tf::SessionConfig session;
    std::string name{"Graph"};
    std::string scope{"/"};
    bool verbose{false};
  };
  
  /*
   * Instantiation
   */

  //! @note Copy and copy assignment construction should not be permitted.
  //! @note This behavior is already enforced by using std::unique_ptr to wrap the session pointer.
  Graph(const Graph& other) = delete;
  Graph& operator=(const Graph& other) = delete;
  
  //! @note Move and move-assignment construction is permissible for this class.
  Graph(Graph&& other) = default;
  Graph& operator=(Graph&& other) = default;
  
  /*!
   * @brief
   * @param session Explicit graph constructor fully specifying the session and object configurations.
   * @param name The name of the graph instance. Should be unique within owning scope.
   * @param scope The scope within which the object is defined.
   * @param verbose Set to true to enable verbose console output.
   */
  explicit Graph(
    tf::SessionConfig session=tf::SessionConfig(),
    const std::string& name="Graph",
    const std::string& scope="/",
    bool verbose=false);
  
  /*!
   * @brief Explicit graph constructor using the configuration helper.
   * @param config Configurations for the session and noesis::Object instance.
   */
  explicit Graph(const Config& config);
  
  /*!
   * @brief Default graph destructor.
   * @note This releases all session resources.
   */
  ~Graph() override;
  
  /*
   * Configurations
   */
  
  /*!
   * @brief Adds a target device from which the session should allocate resources.
   * @param name The name of the device to be added, such as CPU0, GPU0 etc.
   * @note The device name is NOT the fully qualified TF target such as "/device:CPU:0".
   */
  void addDevice(const std::string& name);
  
  /*
   * Properties
   */
  
  /*!
   * @brief Retrieves the list of devices for which the session will allocate resources.
   * @return The vector of string device names.
   * @note The device names are NOT fully qualified TF devices.
   */
  std::vector<std::string> devices() const { return devices_; }
  
  /*!
   * @brief Indicates if the graph currently has an active session for performing operations.
   * @return Returns `true` only if a session is active, else `false`.
   */
  bool isActive() const { return ((session_) ? true : false); }
  
  /*
   * Operations
   */

  /*!
   * @brief Executes a Python script which should generate to MetaGraphDef protobuf file to later load.
   * @param pyscript The path (i.e. absolute filename) to the target Python script to be executed.
   * @param virtualenv The optional virtualenv within which to execute the target Python script.
   */
  void generateFrom(const std::string& pyscript, const std::string& virtualenv="noesis");
  
  /*!
   * @brief Loads an existing graph in TensorFlow's MetaGraphDef format from the specified file.
   * @param filename The filename (must be a fully-qualified absolute path) from which to load the graph.
   */
  void loadFrom(const std::string& filename);
  
  /*!
   * @brief Launches a new TensorFlow session and initializes all in-graph global resources.
   * @note This assumes that a MetaGraphDef has already been loaded from file.
   */
  void startup();
  
  /*!
   * @brief Terminates active sessions and releases all allocated device resources.
   * @note This operation also releases the MetaGraphDef so new ones have to be loaded.
   */
  void shutdown();
  
  /*!
   * @brief Initializes all internal global in-graph variables.
   */
  void initialize();
  
  /*!
   * @brief Restores the graph state from TensorFlow checkpoint format at the specified path.
   * @param path The target path from which to load the graph checkpoint.
   */
  void restoreFrom(const std::string& path, const std::string& saver="");
  
  /*!
   * @brief Saves the current graph state to TensorFlow checkpoint format at the specified path.
   * @param path The target path at which to generate the graph checkpoint.
   */
  void saveTo(const std::string& path);
  
  /*!
   * @brief Resets (to zero) the built-in in-graph global-step variable.
   */
  void resetGlobalStep();
  
  /*!
   * @brief Increments the built-in in-graph global-step variable.
   */
  void incrementGlobalStep();
  
  /*
   * Low-level graph execution interface - operations on raw TensorFlow Tensor containers.
   */
  
  void run(
    const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
    const std::vector<std::string>& output_tensor_names,
    const std::vector<std::string>& target_node_names,
    std::vector<tensorflow::Tensor>& outputs
  );
  
  void run(
    const std::vector<std::string>& output_tensor_names,
    const std::vector<std::string>& target_node_names,
    std::vector<tensorflow::Tensor>& outputs
  );
  
  void run(
    const std::vector<std::string>& output_tensor_names,
    std::vector<tensorflow::Tensor>& outputs
  );
  
  void run(
    const std::string& output_tensor_name,
    tensorflow::Tensor& output
  );
  
  void run(
    const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
    const std::vector<std::string>& target_node_names
  );
  
  void run(
    const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
    const std::string& target_node_name
  );
  
  void run(const std::vector<std::string>& target_node_names);
  
  void run(const std::string& target_node_name);
  
  template<typename Scalar_>
  inline void run(
    const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
    const std::vector<std::string>& output_tensor_names,
    const std::vector<std::string>& target_node_names,
    std::vector<Tensor<Scalar_>>& outputs
  );
  
  template<typename Scalar_>
  inline void run(
    const std::vector<Tensor<Scalar_>>& inputs,
    const std::vector<std::string>& output_tensor_names,
    const std::vector<std::string>& target_node_names,
    std::vector<Tensor<Scalar_>>& outputs
  );
  
  template<typename Scalar_>
  inline void run(
    const std::vector<Tensor<Scalar_>>& inputs,
    const std::string& output_tensor_name,
    const std::vector<std::string>& target_node_names,
    Tensor<Scalar_>& output
  );
  
  template<typename Scalar_>
  inline void run(
    const std::vector<std::string>& output_tensor_names,
    const std::vector<std::string>& target_node_names,
    std::vector<Tensor<Scalar_>>& outputs
  );
  
  template<typename Scalar_>
  inline void run(
    const std::vector<std::string>& output_tensor_names,
    std::vector<Tensor<Scalar_>>& outputs
  );
  
  template<typename Scalar_>
  inline void run(
    const std::string& output_tensor_name,
    Tensor<Scalar_>& output
  );
  
  /*
   * Mid-level graph execution interface - operations on noesis Tensor and TensorTuple containers.
   */
  
  template<typename Scalar_>
  inline void run(
    const std::vector<Tensor<Scalar_>>& inputs,
    const std::vector<std::string>& operations,
    std::vector<Tensor<Scalar_>>& outputs
  );
  
  template<typename Scalar_>
  inline void run(
    const std::vector<Tensor<Scalar_>>& inputs,
    const std::string& operation,
    std::vector<Tensor<Scalar_>>& outputs
  );
  
  template<typename Scalar_>
  inline void run(
    const std::vector<Tensor<Scalar_>>& inputs,
    std::vector<Tensor<Scalar_>>& outputs
  );
  
  template<typename Scalar_>
  inline void run(
    const std::vector<Tensor<Scalar_>>& inputs,
    Tensor<Scalar_>& output
  );
  
  template<typename Scalar_>
  inline void run(
    const std::vector<Tensor<Scalar_>>& inputs,
    const std::vector<std::string>& operations
  );
  
  template<typename Scalar_>
  inline void run(
    const std::vector<Tensor<Scalar_>>& inputs,
    const std::string& operation
  );
  
  template<typename Scalar_>
  inline void run(
    const Tensor<Scalar_>& input,
    const std::string& operation
  );
  
  /*
   * High-level graph execution interface - operations on noesis Tensor-tuples
   */
  
  template<typename Scalar_>
  inline void run(const TensorTuple<Scalar_>& inputs, TensorTuple<Scalar_>& outputs);
  
  template<typename Scalar_>
  inline void run(const TensorTuple<Scalar_>& inputs, Tensor<Scalar_>& output);
  
  template<typename Scalar_>
  inline void run(const TensorTuple<Scalar_>& inputs, std::vector<std::string>& operations);
  
  template<typename Scalar_>
  inline void run(const TensorTuple<Scalar_>& inputs, std::string& operation);
  
  /*
   * Scalar non-trainable variable interface
   */

  template<typename Scalar_>
  inline Scalar_ readModifiable(const std::string& name);
  
  template<typename Scalar_>
  inline void writeModifiable(const std::string& name, Scalar_ value);
  
  template<typename Scalar_>
  inline void writeModifiable(const std::string& name, tensorflow::DataType dtype, Scalar_ value);
  
private:

  /*!
   * @brief Helper function for session configuration and creation.
   */
  void createSession();

private:
  //! @brief The MetaGraphDef contains the definition of graph operations.
  tensorflow::MetaGraphDef metaGraphDef_;
  //! @brief List of devices from which the session will allocate resources.
  //! @note The resources consists of memory and session thread.
  std::vector<std::string> devices_{{"CPU0"}};
  //! @brief Session configuration struct helper.
  tf::SessionConfig sessionConfig_;
  //! @brief Pointer to the TensorFlow session created to execute graph operations.
  std::unique_ptr<tensorflow::Session> session_{nullptr};
};

} // namespace core
} // namespace noesis

// Include the template implementations
#include "noesis/framework/core/Graph.tpp"

#endif // NOESIS_FRAMEWORK_CORE_GRAPH_HPP_

/* EOF */
