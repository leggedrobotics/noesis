# Noesis C++

**DISCLAIMER**: Doxygen documentation is currently under construction.

### Library Components

* [__framework__](include/noesis/framework):      
    * __core__: basic data structures, such as the tensor `noesis::Tensor`, `noesis::TensorTuple` and `noesis::Graph`
    * __hyperparam__: interface to manage hyper-parameters
    * __log__: tools for logging using [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) and console output
    * __math__: mathematics operations, statistics helpers for tensors and pseudo-random number generators
    * __system__: interfaces for filesystem operations and system signal handling
    * __utils__: utilities for parsing and loading datasets
* [__MDP__](include/noesis/mdp): abstract interface for generalized MDP problems
* [__gym__](include/noesis/gym): 
    * __core__: environment base class and relevant decorators
    * __envs__: environment implementations and physics engines interfaces
    * __train__: Interface definitions for common objects used in training
* [__rl__](include/noesis/rl):
    * __agent__: DRL agent classes
    * __algorithm__: DRL algorithms
    * __function__: interfaces to parameterized functions
    * __memory__: classes for data-set collection
    * __sample__: sampling algorithms
    * __train__: high-level classes used in training programs

### API

#### Namespaces

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

```c++
#include <noesis/framework/core/Tensor.hpp>
...
// creating a batched tensor named "batched_tensor" with data size = [4, 4], timesteps = 3, batches = 2 
noesis::Tensor<double> tensor1("batched_tensor", {4,4,3,2}, true);
...
// creating a normal tensor named "normal_tensor" with data size = [4, 4, 3, 2] 
noesis::Tensor<double> tensor2("normal_tensor", {4,4,3,2}, false);
...
```
