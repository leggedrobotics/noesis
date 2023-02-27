# Noesis Gym

The noesis environments are implemented as close as possible to [OpenAI Gym](https://gym.openai.com/) in order to make it consistent to compare and 
develop new algorithms.

Currently we support three possibilities to integrate new environments:     
* __Classic:__ the bare minimum implementation for creating your own environment from scratch. Includes cartpole, pendulum, and acrobot
* __Mujoco:__ integrates the [MuJoCo](http://www.mujoco.org/) physics engine. Includes ant, halfcheetah, humanoid, hopper, and walker-2d
* __RaiSim:__ integrates RaiSim pjysics engine. Includes capler

### Environments

Each environment has the following common components:     
* __Actions:__ Inherits from the `noesis::Actions` class. Used to define the type of actions that the agent can take.
* __Observations:__ Inherits from the `noesis::Observations` class. Used to define the type of observations that the agent receives from the environment.
* __Environment:__ Provides methods to reset and step through the environment. Normally, allows multiple-instances to be used.
* __Visualizer:__ Performs asynchornous or synchronous rendering of the environment. Can also be used to record videos for inspection.
 
