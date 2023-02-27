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
    # dest = os.path.expanduser("~/.noesis/tmp/graph")
    print("Generating graph in path: ", dest)

    # Create a noesis graph
    graph = ns.Graph()

    # Define Tensor shapes for observations and actions
    # TODO: Get info from C++ --> Maybe create them from TensorShape protos????
    obs_specs = [
        tf.TensorSpec(name="target", shape=[1], dtype=tf.float32),
        tf.TensorSpec(name="motion", shape=[6], dtype=tf.float32),
        tf.TensorSpec(name="history", shape=[2], dtype=tf.float32)
    ]
    act_specs = [
        tf.TensorSpec(name="commands", shape=[2], dtype=tf.float32)
    ]

    # Append all operations to the specified tensorflow graph
    with graph.as_default():
        with tf.device("/device:CPU:0"):

            # Create placeholders
            with tf.name_scope('Observations'):
                obs = ns.placeholders_from_specs(obs_specs, batched=True)

            with tf.name_scope('Actor'):
                # Define actor network
                policy_units = [64, 64]
                policy_activations = [tf.nn.tanh, tf.nn.tanh]
                policy_net = ns.nn.MultiLayerPerceptron(inputs=obs, units=policy_units, activations=policy_activations)
                # Create a policy
                policy = ns.func.DiagonalGaussianPolicy(latent=policy_net, actions_specs=act_specs)
                # Create a policy optimizer
                policy_opt = ns.rl.TrustRegionPolicyOptimization(policy=policy)

            with tf.name_scope('Critic'):
                # Define critic network
                critic_units = [64, 64]
                critic_activations = [tf.nn.tanh, tf.nn.tanh]
                critic_net = ns.nn.MultiLayerPerceptron(inputs=obs, units=critic_units, activations=critic_activations)
                # # Create a policy
                critic = ns.func.StateValueFunction(latent=critic_net)
                # Create a critic optimizer
                critic_opt = ns.rl.DampedPolicyEvaluation(critic=critic)

    # Finalize the graph in order to add all necessary global initialization ops
    graph.finalize()

    # Generate a summary event in order to inspect the graph with tensorboard.
    graph.to_event(path=dest + "/events")

    # Generate protobuf file containing the MetaGraphDef.
    graph.to_protobuf(path=dest + "/protos")


# Main program entry-point
if __name__ == '__main__':
    main()

# EOF
