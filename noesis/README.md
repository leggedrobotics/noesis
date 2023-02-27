Noesis
----

### Package Modules

* [__Framework__](include/noesis/framework): lies at the center of the Noesis framework and comprises of the following:      
    * __core__: defines the basic data structures, such as the tensor `noesis::Tensor` and several functions that can be used to 
    manipulate these data structures
    * __graph__: implements the interface for generating, loading and running TensorFlow-based computation graphs and sub-graphs
    * __hyperparam__: globally manages the parameters defined by other modules
    * __log__: includes tools for logging using [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) and console 
    output
    * __math__: provides several mathematics operations, including kernels, random generators, and common probability distributions
    * __system__: deals with filesystem operations and system signal handling
    * __utils__: includes utilities for parsing and loading datasets
* [__Estimator__](include/noesis/estimator): defines abstract class and its implementation for supervised learning
* [__Agent__](include/noesis/agent): defines the abstract class and its implementation for creating a reinforcement learning agent
* [__Environment__](include/noesis/environment): defines the abstract classes required for creating an environment in Noesis
* [__Task__](include/noesis/task): complements the __environment__ module by providing an easy-to-use interface to handle reward terms and 
termination criteria for the environment 
* [__Algorithm__](include/noesis/algorithm): implements the learning algorithms, including policy evaluation, policy gradient estimation, 
and regression
* [__Function__](include/noesis/function): complements the `graph` submodule in __framework__ to create computation graphs for a particular
purpose, such as policy or Q-value estimation networks 
* [__Memory__](include/noesis/memory): provides task-dependent buffer implementations to store Tensors
* [__Runner__](include/noesis/runner): acts as a middleware between the agent and the environment instances to train or evaluate an RL 
algorithm

### API Concepts

#### noesis Namespace

All the Noesis classes and functions are placed in the `noesis` namespace. In order to access this functionality, use the specifier `noesis::`
or the directive `using namespace noesis`.

#### Tensors

A rather brief introduction to the `noesis::Tensor` class. Tensors are multi-dimensional array of elements. These can be interpreted as 
generalized matrices. A 1-D and 2-D tensor are actually a vector and a matrix respectively. The dimension of a tensor is called its rank.

In Noesis, the tensors are managed using [Eigen TensorMap](https://eigen.tuxfamily.org/dox/unsupported/classEigen_1_1TensorMap.html). They 
are stored in a column-major order (opposed to the`tensorflow::Tensor` which is in row-major form). 

Since the core design of the framework accounts for the need of RL algorithms, we provide methods to explicitly consider the two last two 
dimensions of the tensor as the maximum time-steps and batch sizes respectively. These type of tensors are called _batched_ tensors and typically
look like __[datum shape, time, batch]__. 

As an example of where batched tensor can be used, consider an agent acting in the environment. It observes RGB images (size HxWx3). We want
to store all the images it observes for a specified number of trajectories (B) with each trajectory lasting for some preset number of 
timesteps (T). Then, the shape of the batched tensor may look like: [H, W, 3, T, B].

To account for cases where one might not be interested in sequential data (such as in supervised learning), it is possible to create a _normal_ 
tensor, which would look like __[datum shape]__, by setting the tensor as not-batched at the time of creation.

```objectivec
#include <noesis/framework/core/Tensor.hpp>
...
// creating a batched tensor named "batched_tensor" with data size = [4, 4], timesteps = 3, batches = 2 
noesis::Tensor<double> tensor1("batched_tensor", {4,4,3,2}, true);
...
// creating a normal tensor named "normal_tensor" with data size = [4, 4, 3, 2] 
noesis::Tensor<double> tensor2("normal_tensor", {4,4,3,2}, false);
...
```
