# Copyright 2023 The Noesis Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Graph-level implementation of the Trust-Region Policy Optimization (TRPO) policy-gradient algorithm."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.compat.v1 as tf1

# from noesis.new.parameter import get_modifiable_parameter
from noesis.core.tensor import max_dtype, placeholders_from_specs, flatten_and_concatenate, modifiable
from noesis.core.gradient import compute_flat_gradient, conjugate_gradient_descent


class TrustRegionPolicyOptimization(tf.Module):
    """Implements a policy gradient estimate using Trust Region Policy Optimization [1].

    This implementation is based on the algorithms described in [1] and [2], and draws
    inspiration from the implementations provided in [3].

    [1] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz,
        "Trust Region Policy Optimization",
        In International conference on machine learning, pp. 1889-1897, 2015.

    [2] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel,
        "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
        arXiv:1506.02438, 2018

    [3] OpenAI Baselines. (https://github.com/openai/baselines)
    """
    def __init__(self,
                 policy,
                 name="TrustRegionPolicyOptimization",
                 cg_damping=0.1,
                 cg_tol=1e-15,
                 cg_max_iter=40,
                 entropy_weight=0.0):
        super(TrustRegionPolicyOptimization, self).__init__(name=name)
        assert isinstance(policy, tf.Module)
        # TODO: assert isinstance(policy, ns.func.Policy)
        # Retrieve architecture information
        self.__theta = policy.parameters
        # Retrieve actions specifications and outputs from the policy function
        self.__actions_specs = policy.actions_specs
        self.__dtype = max_dtype(policy.actions_specs)
        self.__actions_sample_shape = policy.sample_shape
        # Retrieve the action distributions required by the CPI loss.
        self.__actions_pd = policy.pd
        self.__actions_pd_old = policy.pd_fixed
        # Add the graph operations for this class
        with tf.name_scope(self.name_scope.name):
            # Define variables and operations for the algorithm configurations
            with tf.name_scope("configurations"):
                self.__cg_damping, _ = modifiable(cg_damping, dtype=self.__dtype, name="cg_damping")
                self.__cg_tolerance, _ = modifiable(cg_tol, dtype=self.__dtype, name="cg_tolerance")
                self.__cg_max_iterations, _ = modifiable(cg_max_iter, dtype=tf.int64, name="cg_max_iterations")
                self.__entropy_weight, _ = modifiable(entropy_weight, dtype=self.__dtype, name="entropy_weight")
            # Define the algorithm sample and gradient inputs
            with tf.name_scope("inputs"):
                # Input policy gradient
                policy_gradient_in = tf1.placeholder(self.__dtype, shape=[1, None], name='policy_gradient')
                # Input advantage
                advantage_in = tf1.placeholder(dtype=self.__dtype, name='advantages')
                advantage = tf.reshape(tensor=advantage_in, shape=[-1])
                # Input action samples from previous distribution
                with tf.name_scope("actions"):
                    actions_samples_old_in = placeholders_from_specs(self.__actions_specs, batched=True)
                    actions_samples_old = flatten_and_concatenate(actions_samples_old_in, name='flat')
                    actions_samples_old = tf.reshape(actions_samples_old, shape=self.__actions_sample_shape)
            # Algorithm loss function
            with tf.name_scope("loss"):
                # Probability ratio
                with tf.name_scope("probability_ratio"):
                    # The probability of the action samples originating from the next policy - parametric qty
                    log_prob = self.__actions_pd.log_prob(actions_samples_old)
                    log_prob_op = tf.reshape(tensor=log_prob, shape=[-1, 1], name="log_prob_op")
                    log_prob_old = self.__actions_pd_old.log_prob(actions_samples_old)
                    log_prob_old_op = tf.reshape(tensor=log_prob_old, shape=[-1, 1], name="log_prob_old_op")
                    log_prob_ratio = log_prob - log_prob_old
                    prob_ratio = tf.exp(log_prob_ratio, name="prob_ratio_op")
                # Entropy
                with tf.name_scope("entropy"):
                    entropy = self.__actions_pd.entropy()
                    mean_entropy_op = tf.reduce_mean(entropy, name="mean_entropy_op")
                # KL divergence
                with tf.name_scope("kl_divergence"):
                    kld = self.__actions_pd_old.kl_divergence(self.__actions_pd)
                    mean_kld_op = tf.reduce_mean(kld, name="mean_kld_op")
                # Surrogate loss, a.k.a the conservative policy iteration loss (L^CPI)
                with tf.name_scope("surrogate"):
                    cpi = prob_ratio * advantage
                    cpi_op = tf.reduce_mean(cpi, name="cpi_op")
                # Total loss
                with tf.name_scope("total"):
                    # Set the base loss to the Conservative Policy Iteration loss (L^CPI)
                    loss_op = cpi_op
                    # Add the entropy regularization term (H[pi])
                    loss_op = loss_op + tf.multiply(self.__entropy_weight, mean_entropy_op)
                    # Total loss operation
                    loss_op = tf.identity(loss_op, name="loss_op")

            # Compute policy gradient estimate
            with tf.name_scope("gradient"):
                # Compute policy gradient estimate
                policy_gradient_op, _ = compute_flat_gradient(
                    loss=loss_op,
                    variables=self.__theta,
                    name="policy_gradient_op"
                )
                # Compute the KL-divergence constraint gradient estimate
                kld_gradient_op, _ = compute_flat_gradient(
                    loss=mean_kld_op,
                    variables=self.__theta,
                    name="kld_gradient_op"
                )

                # Fisher-vector products
                def hessian_vector_product(tangent_op):
                    gp = tf.reduce_sum(kld_gradient_op * tangent_op)
                    gradient, _ = compute_flat_gradient(loss=gp, variables=self.__theta)
                    return gradient

                # Compute natural gradient estimate
                ng, cge = conjugate_gradient_descent(
                    dtype=self.__dtype,
                    eval=hessian_vector_product,
                    b=policy_gradient_in,
                    max_iter=self.__cg_max_iterations,
                    tolerance=self.__cg_tolerance,
                    damping=self.__cg_damping
                )
                # Gradient accessor operations
                natural_gradient_op = tf.identity(ng, name='natural_gradient_op')
                conjugate_gradient_error_op = tf.identity(cge, name='conjugate_gradient_error_op')

# EOF
