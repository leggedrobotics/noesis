# Copyright 2020 The Noesis Authors. All Rights Reserved.
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
Graph-level implementation of a temporal-difference learning algorithm
using TD(1) and clipped value-error objective.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from noesis.core.tensor import size_of_tensor, step_counter, modifiable
from noesis.core.loss import squared_clipped


class ClippedPolicyEvaluation(tf.Module):
    """Implements the Clipped Policy Evaluation algorithm based on [1].

    This class implements the computation of gradients with respect to a
    Supervised-Learning error. Values are those output by the function
    approximator, while the TD-error targets must be provided by and another
    implementation.

    In addition to the TD-error, the total loss also incorporates a clipping term
    according to [1], which, provides an upper-bound on on the un-clipped loss.

    [1] John Schulman, Filipp Wolski, Prafulla Dhariwal, ALex Radford, Oleg Klimov,
        "Proximal Policy Optimization",
        arXiv:1707.06347, 2017
    """
    def __init__(self,
                 critic,
                 name="ClippedPolicyEvaluation",
                 learning_rate=1e-3,
                 lr_decay_rate=1.0,
                 lr_decay_steps=1,
                 clipping=0.2,
                 cl_decay_rate=1.0,
                 cl_decay_steps=1,
                 ve_weight=1.0,
                 l2r_weight=0.0,
                 max_grad_norm=1.0):
        super(ClippedPolicyEvaluation, self).__init__(name=name)
        # Set the value computation head of the critic.
        self.__value = critic.value
        # Retrieve architecture information.
        self.__params = critic.parameters
        # Set the DType for the operations of this class based on that of the critic's value operation.
        dtype = critic.value.dtype
        # Add the graph operations for this class
        with tf.name_scope(self.name_scope.name):
            # Define variables and operations for the algorithm configurations
            with tf.name_scope("configurations"):
                learning_rate, _ = modifiable(learning_rate, dtype=dtype, name="learning_rate")
                clipping, _ = modifiable(clipping, dtype=dtype, name="clipping")
                ve_weight, _ = modifiable(ve_weight, dtype=dtype, name="ve_weight")
                l2r_weight, _ = modifiable(l2r_weight, dtype=dtype, name="l2_regularization")
                max_grad_norm, _ = modifiable(max_grad_norm, dtype=dtype, name="max_grad_norm")
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
            # Algorithm inputs
            with tf.name_scope("values"):
                value_predicted = tf.identity(self.__value, name='predicted')
                value_previous_in = tf1.placeholder(dtype=dtype, shape=[None, 1], name='previous')
                value_target_in = tf1.placeholder(dtype=dtype, shape=[None, 1], name='targets')
            # Algorithm loss function
            with tf.name_scope("loss"):
                with tf.name_scope("value_prediction"):
                    value_error_loss_op = squared_clipped(
                        target=value_target_in,
                        value=value_predicted,
                        value_prev=value_previous_in,
                        clip_range=clipping,
                        name='loss_op'
                    )
                with tf.name_scope("l2_regularization"):
                    weights = [params for params in self.__params if 'bias' not in params.name]
                    num_weights = tf.constant(value=sum([size_of_tensor(w) for w in weights]), dtype=dtype)
                    l2_regularization_op_in = [tf.nn.l2_loss(w) for w in weights]
                    l2_regularization_op = tf.add_n(l2_regularization_op_in)
                    l2_regularization_op = tf.divide(x=l2_regularization_op, y=num_weights, name='loss_op')
                with tf.name_scope("total"):
                    # Apply the value error weighting coefficient
                    loss_op = tf.multiply(ve_weight, value_error_loss_op)
                    # Add the L2-norm regularization penalty term
                    loss_op = loss_op + tf.multiply(l2r_weight, l2_regularization_op)
                    # Define the accessible operation for the total loss
                    loss_op = tf.identity(input=loss_op, name='loss_op')
            # Gradient computation operations
            with tf.name_scope("gradient"):
                grads_and_params = optimizer.compute_gradients(loss=loss_op, var_list=self.__params)
                grads, _ = zip(*grads_and_params)
                grads_norm_op = tf.identity(tf1.global_norm(grads), name="norm_op")
                clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_grad_norm, use_norm=grads_norm_op)
                clipped_gradient_norm_op = tf.identity(tf1.global_norm(clipped_grads), name="clipped_norm_op")
            # Algorithm training operation
            with tf.name_scope('train'):
                train_step, _ = step_counter(name="step_counter")
                grads_and_params = zip(clipped_grads, self.__params)
                train_op = optimizer.apply_gradients(
                    grads_and_vars=grads_and_params,
                    global_step=train_step,
                    name="train_op"
                )

# EOF
