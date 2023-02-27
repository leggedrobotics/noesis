#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import tensorflow as tf
import noesis as ns


# Reduce the verbosity of the console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# Main function
def main():
    # Set destination for output
    dest = os.path.dirname(os.path.abspath(__file__))
    print("Generating graph in path: ", dest)

    # Create a noesis graph
    graph = ns.Graph()

    # Define Tensor shapes for observations and actions
    obs_specs = [
        tf.TensorSpec(name="motion", shape=[3], dtype=tf.float32),
    ]

    # Discrete actions
    act_specs = [tf.TensorSpec(name="torque", shape=[1], dtype=tf.float32)]

    # Append all operations to the specified tensorflow graph
    with graph.as_default():
        with tf.device("/device:CPU:0"):

            with tf.name_scope('Example/Agent'):

                # Create placeholders
                with tf.name_scope('Observations'):
                    obs = ns.placeholders_from_specs(obs_specs, batched=True)

                # Define actor network
                policy_units = [16, 8]
                policy_activations = [tf.nn.relu, tf.nn.relu]
                policy_net = ns.nn.MultiLayerPerceptron(inputs=obs, units=policy_units, activations=policy_activations)
                # Define a policy function
                policy = ns.func.DiagonalGaussianPolicy(
                    latent=policy_net,
                    actions_specs=act_specs,
                    name="Policy",
                    init_stddev=1.0
                )
                # Create a policy optimizer
                policy_opt = ns.rl.ProximalPolicyOptimization(
                    policy=policy,
                    name="PolicyOptimizer",
                    lr_decay_rate=0.9,
                    lr_decay_steps=500
                )

                # Define critic network
                critic_units = [16, 8]
                critic_activations = [tf.nn.relu, tf.nn.relu]
                critic_net = ns.nn.MultiLayerPerceptron(inputs=obs, units=critic_units, activations=critic_activations)
                # # Create a policy
                critic = ns.func.StateValueFunction(latent=critic_net, name="Value")
                # Create a critic optimizer
                critic_opt = ns.rl.ClippedPolicyEvaluation(
                    critic=critic,
                    name="PolicyEvaluator",
                    lr_decay_rate=0.9,
                    lr_decay_steps=500
                )

    # Finalize the graph in order to add all necessary global initialization ops
    graph.finalize()

    # Generate protobuf file
    graph.to_event(path=dest + "/events")

    # Generate protobuf file
    graph.to_protobuf(path=dest)


# Main program entry-point
if __name__ == '__main__':
    main()

# EOF
