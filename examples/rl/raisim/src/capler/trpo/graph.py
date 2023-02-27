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

            with tf.name_scope('Example/Agent'):

                # Create placeholders
                with tf.name_scope('Observations'):
                    obs = ns.placeholders_from_specs(obs_specs, batched=True)

                # Define actor network
                policy_net = ns.nn.MultiLayerPerceptron(
                    inputs=obs,
                    units=[64, 64],
                    activations=[tf.nn.tanh, tf.nn.tanh]
                )
                # Create a policy
                policy = ns.func.DiagonalGaussianPolicy(
                    latent=policy_net,
                    actions_specs=act_specs,
                    name="Policy",
                    init_stddev=5.0,
                    min_stddev=0.0005,
                    stddev_from_latent=False
                )
                # Create a policy optimizer
                policy_opt = ns.rl.TrustRegionPolicyOptimization(policy=policy, name="PolicyOptimizer")

                # Define critic network
                critic_net = ns.nn.MultiLayerPerceptron(
                    inputs=obs,
                    units=[64, 64],
                    activations=[tf.nn.tanh, tf.nn.tanh]
                )
                # Create a critic
                critic = ns.func.StateValueFunction(latent=critic_net, name="Value")
                # Create a critic optimizer
                critic_opt = ns.rl.DampedPolicyEvaluation(critic=critic, name="PolicyEvaluator")

    # Finalize the graph in order to add all necessary global initialization ops
    graph.finalize()

    # Generate protobuf file
    dest = os.path.dirname(os.path.abspath(__file__))
    print("Generating graph in path: ", dest)
    graph.to_event(path=dest + "/events")
    graph.to_protobuf(path=dest)


# Main program entry-point
if __name__ == '__main__':
    main()

# EOF
