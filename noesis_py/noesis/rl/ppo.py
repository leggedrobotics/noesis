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

"""
Graph-level implementation of the Proximal Policy Optimization (PPO) policy-gradient algorithm.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from noesis.core.tensor import max_dtype, placeholders_from_specs, flatten_and_concatenate, modifiable, step_counter


class ProximalPolicyOptimization(tf.Module):
    """Implements a policy gradient estimate using Proximal Policy Optimization [1].

    This implementation is based on those described in [1] and [2], and draws
    from the implementations provided in [3].

    [1] John Schulman, Filipp Wolski, Prafulla Dhariwal, ALex Radford, Oleg Klimov.
        Proximal Policy Optimization.
        arXiv pre-print arXiv:1707.06347, 2017, (https://arxiv.org/abs/1707.06347)

    [2] Nicolas Heess, Dhruva TB, Srinivasan Sriram, Jay Lemmon, Josh Merel, Greg Wayne,
        Yuval Tassa, Tom Erez, Ziyu Wang, S. M. Ali Eslami, Martin Riedmiller, David Silver.
        Emergence of Locomotion Behaviours in Rich Environments.
        arXiv pre-print arXiv:1707.06347, 2017. (https://arxiv.org/abs/1707.02286)

    [3] OpenAI Baselines. (https://github.com/openai/baselines)
    """
    def __init__(self,
                 policy,
                 name="ProximalPolicyOptimization",
                 learning_rate=1e-3,
                 lr_decay_rate=1.0,
                 lr_decay_steps=1,
                 kld_penalty=0.0,
                 clipping=0.2,
                 cl_decay_rate=1.0,
                 cl_decay_steps=1,
                 entropy_weight=0.0,
                 max_grad_norm=1.0,
                 use_clipping=True):
        super(ProximalPolicyOptimization, self).__init__(name=name)
        assert isinstance(policy, tf.Module)
        # TODO: assert isinstance(policy, ns.func.Policy)
        # Retrieve architecture information
        self.__params = policy.parameters
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
                learning_rate, _ = modifiable(learning_rate, dtype=self.__dtype, name="learning_rate")
                clipping, _ = modifiable(clipping, dtype=self.__dtype, name="clipping")
                kld_penalty, _ = modifiable(kld_penalty, dtype=self.__dtype, name="kld_penalty")
                entropy_weight, _ = modifiable(entropy_weight, dtype=self.__dtype, name="entropy_weight")
                max_grad_norm, _ = modifiable(max_grad_norm, dtype=self.__dtype, name="max_grad_norm")
            # A local counter used for the decays
            local_step, _ = step_counter(name="decays/step_counter")
            # Wrap the configured initial value with a decay operation
            learning_rate = tf1.train.exponential_decay(
                learning_rate,
                global_step=local_step,
                decay_rate=lr_decay_rate,
                decay_steps=lr_decay_steps,
                name="decays/learning_rate"
            )
            learning_rate = tf.identity(learning_rate, name="learning_rate/value")
            # Wrap the configured initial value with a decay operation
            clipping = tf1.train.exponential_decay(
                clipping,
                global_step=local_step,
                decay_rate=cl_decay_rate,
                decay_steps=cl_decay_steps,
                name="decays/clipping"
            )
            clipping = tf.identity(clipping, name="clipping/value")
            # Configure a default optimizer if has not been defined
            optimizer = tf1.train.AdamOptimizer(learning_rate=learning_rate, name="Optimizer")
            # Define the algorithm sample inputs
            with tf.name_scope("inputs"):
                advantage_in = tf1.placeholder(dtype=self.__dtype, name='advantages')
                advantage = tf.reshape(tensor=advantage_in, shape=[-1])
                log_prob_old_in = tf1.placeholder(dtype=self.__dtype, name='logprobs')
                log_prob_old = tf.reshape(tensor=log_prob_old_in, shape=[-1])
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
                    log_prob_ratio = log_prob - log_prob_old
                    prob_ratio = tf.exp(log_prob_ratio, name="prob_ratio_op")
                # Entropy
                with tf.name_scope("entropy"):
                    entropy = self.__actions_pd.entropy()
                    mean_entropy_op = tf.reduce_mean(entropy, name="mean_entropy_op")
                # KL divergence
                with tf.name_scope("kl_divergence"):
                    kld = 0.5 * tf.square(-log_prob_ratio)
                    mean_kld_op = tf.reduce_mean(kld, name="mean_kld_op")
                # Surrogate loss, a.k.a the conservative policy iteration loss (L^CPI)
                with tf.name_scope("surrogate"):
                    cpi = prob_ratio * advantage
                    cpi_op = tf.reduce_mean(cpi, name="cpi_op")
                # Clipped surrogate loss (L^CLIP)
                with tf.name_scope("clip"):
                    cpi_clipped = advantage * tf.clip_by_value(prob_ratio, 1.0-clipping, 1.0+clipping)
                    cpi_clipped_op = tf.reduce_mean(cpi_clipped, name="cpi_clipped_op")
                    clip = tf.minimum(cpi, cpi_clipped)
                    clip_op = tf.reduce_mean(clip, name="clip_op")
                    clip_fraction = tf.cast(tf.greater(tf.abs(prob_ratio-1.0), clipping), dtype=self.__dtype)
                    clip_fraction_op = tf.reduce_mean(clip_fraction, name="clip_fraction_op")
                # Total loss
                with tf.name_scope("total"):
                    # Set the base loss to the either the clipping-based loss (L^CLIP) or basic CPI loss (L^CPI)
                    loss_op = clip_op if use_clipping else cpi_op
                    # Add the KL-Divergence penalty term (L^KLPEN)
                    loss_op = loss_op - tf.multiply(kld_penalty, mean_kld_op)
                    # Add the entropy regularization term (H[pi])
                    loss_op = loss_op + tf.multiply(entropy_weight, mean_entropy_op)
                    # We must invert the sign because PPO maximizes the loss term, while Adam minimizes it.
                    loss_op = tf.identity(-loss_op, name="loss_op")
            # Compute policy gradient estimate
            with tf.name_scope("gradient"):
                grads_and_params = optimizer.compute_gradients(loss=loss_op, var_list=self.__params)
                grads, _ = zip(*grads_and_params)
                grads_norm_op = tf.identity(tf1.global_norm(grads), name="norm_op")
                clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_grad_norm, use_norm=grads_norm_op)
                clipped_gradient_norm_op = tf.identity(tf1.global_norm(clipped_grads), name="clipped_norm_op")
            # Algorithm training operation
            with tf.name_scope('train'):
                train_step, _ = step_counter(name="step_counter")
                grads_and_params = list(zip(clipped_grads, self.__params))
                train_op = optimizer.apply_gradients(
                    grads_and_vars=grads_and_params,
                    global_step=train_step,
                    name="train_op"
                )

# EOF
