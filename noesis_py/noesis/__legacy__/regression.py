# Copyright 2018 The Noesis Authors. All Rights Reserved.
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


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from noesis.core.base import Algorithm
from noesis.core.registration import register
from noesis.function.deterministic_model import DeterministicModel
from noesis.common.tensor import flatten_and_concatenate, number_of_elements
from noesis.common.configuration import *
from noesis.common.gradient import clip_gradient_by_max_norm
from noesis.common.train import get_optimizer, parameter_decay
from noesis.common.parameter import get_modifiable_parameter
from noesis.common.loss import squared


class NonLinearRegression(Algorithm):
    """Provides a graph-level backend for non-linear regression with deterministic models.
    """
    def __init__(self, name, algo_config, algo_funcs, dtype, device, verbose):
        super(NonLinearRegression, self).__init__(name, algo_config, algo_funcs, dtype, device, verbose)
        # Configure the loss function
        # TODO: self._loss_type = get_loss_type_from_xml_config(algo_config)
        self._loss_type = squared
        # Configure the optimizer
        self._optimizer, self._learning_rate = get_optimizer_from_xml_config(algo_config)
        # Check whether to enable learning-rate decay
        self._lr_decay = get_parameter_decay_from_xml_config(algo_config, "lr_decay")
        # Check whether to enable gradient clipping
        self._max_grad_norm = float(get_parameter_value_from_xml_config(algo_config, 'max_grad_norm'))
        # Configure the L2-norm parameter regularization
        self._l2_regularization = float(get_parameter_value_from_xml_config(algo_config, 'l2_regularization'))

    @staticmethod
    def dependencies():
        return [DeterministicModel.__name__]

    def add_subgraph(self, graph):

        # Retrieve outputs specifications from the model function
        outputs_in = []
        outputs_dims = []
        outputs_names = []
        for output in self.inputs.specs:
            outputs_in.append(self.get_input_node_by_name(output.name))
            outputs_dims.append(self.get_input_dim_by_name(output.name))
            outputs_names.append(output.name)

        # Append all operations to the specified tensorflow graph
        with graph.as_default():
            with tf.device(self.device):

                # Retrieve parameter lists
                all_params, trainable_params = self.get_parameters_by_function_type(graph, DeterministicModel)

                # Sub-graph operations
                with tf.name_scope(self.operations_namescope):
                    # Configure L2-norm regularization parameter
                    l2_regularization, _ = get_modifiable_parameter(initial=self._l2_regularization,
                                                                    dtype=self.dtype,
                                                                    name="l2_regularization",
                                                                    param_scope=self.parameters_namescope)
                    # Configure gradient clipping parameters
                    max_grad_norm, _ = get_modifiable_parameter(initial=self._max_grad_norm,
                                                                dtype=self.dtype,
                                                                name="max_grad_norm",
                                                                param_scope=self.parameters_namescope)
                    # Configure learning-rate decay
                    learning_rate, _ = parameter_decay(config=self._lr_decay,
                                                       initial=self._learning_rate,
                                                       dtype=self.dtype,
                                                       name="learning_rate",
                                                       param_scope=self.parameters_namescope,
                                                       global_step=self.global_step)

                    # Create the gradient-based solver algorithm
                    optimizer = get_optimizer(config=self._optimizer, learning_rate=self._learning_rate)

                    # Algorithm inputs
                    with tf.name_scope("inputs"):
                        with tf.name_scope("predictions"):
                            predictions_in = []
                            for k in range(0, len(outputs_in)):
                                predictions_in.append(tf.identity(input=outputs_in[k], name=outputs_names[k]))
                        with tf.name_scope("targets"):
                            targets_in = []
                            for k in range(0, len(outputs_in)):
                                output_shape = [-1] + outputs_dims[k]
                                targets_in.append(tf.reshape(tf.placeholder(dtype=self.dtype, name=outputs_names[k]), shape=output_shape))
                        with tf.name_scope("total"):
                            predictions = flatten_and_concatenate(predictions_in, outputs_dims, 'predictions')
                            labels = flatten_and_concatenate(targets_in, outputs_dims, 'targets')

                    # Algorithm loss function
                    with tf.name_scope("loss"):
                        with tf.name_scope("supervised_learning"):
                            sl_loss_op = self._loss_type(target=labels, value=predictions, name='loss_op')
                        with tf.name_scope("l2_regularization"):
                            weights = [params for params in trainable_params if 'bias' not in params.name]
                            num_weights = tf.constant(value=sum([number_of_elements(w) for w in weights]), dtype=self.dtype)
                            l2_regularization_op_in = [tf.nn.l2_loss(w) for w in weights]
                            l2_regularization_op = tf.add_n(l2_regularization_op_in)
                            l2_regularization_op = tf.divide(x=l2_regularization_op, y=num_weights, name='loss_op')
                            l2_regularization_op = l2_regularization_op * l2_regularization
                        with tf.name_scope("total"):
                            loss_op = tf.add(x=sl_loss_op, y=l2_regularization_op, name='loss_op')

                    # Algorithm gradients
                    with tf.name_scope("gradients"):
                        grads_and_vars_op = optimizer.compute_gradients(loss=loss_op, colocate_gradients_with_ops=True)
                        gradients, variables = zip(*grads_and_vars_op)
                        gradients, grads_norm_op = clip_gradient_by_max_norm(gradients, max_grad_norm)
                        grads_norm_op = tf.identity(input=grads_norm_op, name="grad_norm_op")
                        grads_and_vars_op = zip(gradients, variables)

                    # Algorithm training operation
                    with tf.name_scope('train'):
                        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars_op,
                                                             global_step=self.global_step,
                                                             name="train_op")

        # Store inputs
        self.add_input_node(predictions)
        self.add_input_node(labels)

        # Store outputs
        self.add_output_node(loss_op)
        self.add_output_node(grads_and_vars_op)
        self.add_output_node(grads_norm_op)
        self.add_output_node(train_op)


# Register architecture implementation
register("ALGO", NonLinearRegression)

# EOF
