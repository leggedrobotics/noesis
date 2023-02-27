# Developers Notes

### Tensor
* [ ] Operation in Tensor to shift samples in time by a specified number of positions up to the max time-step (in one batch)
* [ ] Pre-allocate for flattened tensors in trajmem and have option to not re-allocate on every call to `getFlattenedBatches()`
* [ ] `Tensor getFlattenedBatches(const size_t validTimeSteps = 1) const`: Why do we need the validTimeSteps? Can we have version without it?
* [ ] `Tensor getFlattenedBatches(const size_t validTimeSteps = 1) const`: we need
  - One version which returns reshaped Tensor via TensorMap (i.e. all time-steps are valid)
  - One version which allocates new store to copy to and returns that --> maybe we need move operation here?
  - One version which accepts as argument where to store flattened copies of data.

### Determinism
* [ ] See TensorFlow issues for details:
  - [ ] https://github.com/openai/baselines/issues/805
  - [ ] https://github.com/tensorflow/tensorflow/pull/10636
  - [ ] https://github.com/tensorflow/tensorflow/issues/16889
  - [ ] https://github.com/tensorflow/tensorflow/issues/12871
  - [ ] https://github.com/NVIDIA/tensorflow-determinism
  
### Various
* [ ] Fix how names name-scopes are handled --> use make_namescope internally instead of users using it
* [ ] Double integrator environment (trivial classic control problem)
* [ ] Cleanup of `blocks.py`
* [ ] Use the test runner in the training loop to evaluate performance in a canonical test-train pattern:
* [ ] Evaluate and check the need to use `tf.convert_to_tensor()`
* [ ] Add TF style guide in the documentation.
* [ ] Adopt some things from [TF Agents](https://github.com/tensorflow/agents/blob/master/STYLE_GUIDE.md)
* [ ] Adopt and setup [semantic versioning ](https://github.com/tensorflow/agents/blob/master/tf_agents/version.py) in C++ and Python packages 
* [ ] Design a learner thread into the agent class for online execution.
* [ ] Re-design the RNG to not be templated class but have templated methods so to allow type deduction to determine type
* [ ] Use Python logging in the noesis message util (https://docs.python.org/3/howto/logging.html#logging-basic-tutorial)
* [ ] MPI multi-machine parallelization.
* [ ] Doxygen documentation.
* [ ] ReadTheDocs documentation.
* [ ] Implement DQN.
* [ ] Implement DDPG.
* [ ] Implement SAC.

### Efficiency Improvements
* [ ] Review run-time critical (common) computations and reduce new allocations of `Tensor` and other heavy objects 
* [ ] Same as above, but in general for dynamic allocations, especially of large objects
* [ ] Revise code that has multiple return statements --> prefer using buffer and a single final move
* [ ] Consider using std::move() in CompGraph::run functions
* [ ] Add unit test for Tensor to test efficiency of allocation and assignment ops

### Features

* [ ] Use Google sanitizers to check code quality:
  - https://github.com/google/sanitizers/wiki/AddressSanitizer
  - https://gist.github.com/kwk/4171e37f4bcdf7705329
  - https://stackoverflow.com/questions/37970758/how-to-use-addresssanitizer-in-gcc
  - https://lemire.me/blog/2016/04/20/no-more-leaks-with-sanitize-flags-in-gcc-and-clang/
  - https://github.com/google/sanitizers/wiki/MemorySanitizer

* [ ] Use google benchmark:
  - https://github.com/google/benchmark

* [ ] Check FP operation issues:
  - https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
  - https://gcc.gnu.org/wiki/FloatingPointMath
  - https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
  - `#include <fenv.h>` + `feenableexcept(FE_DIVBYZERO | FE_INEXACT | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);`

* [ ] Try to use `tf.stop_gradient()` together with `tf.control_dependencies()` in order to avoid getting value targets in c++ side:
  - https://www.tensorflow.org/api_docs/python/tf/Graph#control_dependencies
  - https://www.tensorflow.org/api_docs/python/tf/make_template
  - https://blog.metaflow.fr/tensorflow-mutating-variables-and-control-flow-2181dd238e62
  - https://stackoverflow.com/questions/41780655/what-is-the-difference-between-tf-group-and-tf-control-dependencies
  - https://www.tensorflow.org/api_guides/python/control_flow_ops#Control_Flow_Operations
  - https://www.tensorflow.org/api_docs/python/tf/group
  - https://www.tensorflow.org/api_docs/python/tf/tuple

* [ ] TensorFlow 2.0 and API improvements:
  - [ ] Consider using `tf.layers.flatten()` instead of `tf.reshape()`
  - [ ] PD types:
      ```python
      def make_pdtype(ac_space):
          from gym import spaces
          if isinstance(ac_space, spaces.Box):
              assert len(ac_space.shape) == 1
              return DiagGaussianPdType(ac_space.shape[0])
          elif isinstance(ac_space, spaces.Discrete):
              return CategoricalPdType(ac_space.n)
          elif isinstance(ac_space, spaces.MultiDiscrete):
              return MultiCategoricalPdType(ac_space.nvec)
          elif isinstance(ac_space, spaces.MultiBinary):
              return BernoulliPdType(ac_space.n)
          else:
              raise NotImplementedError
      ```
  - [ ] Output of NN arch is `latent` --> fed into the ProbDist to create `distribution`
  - [ ] Built-in observation processing:
    ```python
     def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
         rms = RunningMeanStd(shape=x.shape[1:])
         norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
         return norm_x, rms
    ```
  - [ ] Look at `gym/common/input.py` for ideas about heterogeneous data types in observations
  - [ ] Look at `class TfRunningMeanStd` for implementation ideas
  
  
----
