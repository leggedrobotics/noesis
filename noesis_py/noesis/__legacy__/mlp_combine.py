# Copyright 2019 The Noesis Authors. All Rights Reserved.
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

"""An implementations of multi-layer perceptron (feed-forward) neural-network architecture for function
approximation."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, xavier_initializer

from noesis.common.configuration import *
from noesis.core.base import Architecture
from noesis.core.registration import register
from noesis.common.tensor import flatten_and_concatenate, split_and_fold, flatten_dimensions

import os
from tensorflow.python import pywrap_tensorflow


# TODO: rename class because this MLP is flat and multi-IO
class MultiLayerPerceptron(Architecture):
    """
    Generic multi-layer perceptron architecture.
    """
    def __init__(self, namescope, inputs_spec, outputs_spec, arch_config, dtype, device, verbose):
        super(MultiLayerPerceptron, self).__init__(namescope, inputs_spec, outputs_spec, arch_config, dtype, device,
                                                   verbose)
        # Retrieve the layer activations list
        self._layer_activations, self._layer_units, self._max_initial_value = get_architecture_from_xml_config(arch_config)

    def add_subgraph(self, graph, verbose=False):
        # Retrieve arch input-output specifications
        inputs = self.get_input_nodes()
        inputs_dims = self.get_input_dimensions()
        outputs_names = self.get_output_names()
        outputs_dims = self.get_output_dimensions()

        llc_inputs = inputs[:4]
        llc_inputs_dims = inputs_dims[:4]
        hlc_inputs = inputs[4:11]
        hlc_inputs_dims = inputs_dims[4:11]
        door_inputs = inputs[11:]
        door_inputs_dims = inputs_dims[11:]

        llc_trainable = False

        # Ensure input dimensions are flat:
        for dims in inputs_dims:
            assert (len(dims) <= 2), "Input tensors must be at most rank 2 (i.e. matrices)."
        for dims in outputs_dims:
            assert (len(dims) <= 2), "Output tensors must be at most rank 2 (i.e. matrices)."

        # Compute flattened output dimensions
        flat_output_dims = flatten_dimensions(outputs_dims)

        # Append all operations to the specified tensorflow graph
        with graph.as_default():
            with tf.device(self.device):

                # Construct sub-graph inputs
                with tf.name_scope(self.operations_namescope) as opscope:
                    with tf.name_scope('inputs'):
                        # Check for multiple inputs and concatenate if true
                        input_layer_in = flatten_and_concatenate(inputs, inputs_dims)
                        llc_input_layer_in = flatten_and_concatenate(llc_inputs, llc_inputs_dims)
                        hlc_input_layer_in = flatten_and_concatenate(hlc_inputs, hlc_inputs_dims)
                        door_input_layer_in = flatten_and_concatenate(door_inputs, door_inputs_dims)
                        # Fully-connected input layer
                        input_layer = tf.identity(input_layer_in, name='all_input')
                        llc_input_layer = tf.identity(llc_input_layer_in, name='llc_input')
                        hlc_input_layer = tf.identity(hlc_input_layer_in, name='hlc_input')
                        door_input_layer = tf.identity(door_input_layer_in, name='door_input')

                # Construct sub-graph operations
                # High-Level Controller and Low-Level Controller (only for StochasticPolicy)
                if self.namescope == "StochasticPolicy":
                    torque_outputs_names = outputs_names[0:1]
                    torque_outputs_dims = outputs_dims[0:1]
                    hlc_outputs_names = outputs_names[1:4]
                    hlc_outputs_dims = outputs_dims[1:4]

                    flat_torque_output_dims = flatten_dimensions(torque_outputs_dims)
                    flat_hlc_output_dims = flatten_dimensions(hlc_outputs_dims)

                    # High-Level Controller
                    with tf.variable_scope(self.parameters_namescope + "/HLC"):
                        # Door state extraction
                        with tf.variable_scope("DSE"):
                            layer_num = -1
                            layer_units = [64, 64]
                            dse_top = door_input_layer
                            if len(layer_units) > 0:
                                for layer_dim in layer_units:
                                    layer_num += 1
                                    layer_name = 'layer_'+repr(layer_num+1)
                                    param_namescope = 'hidden/'+layer_name
                                    dse_top = fully_connected(inputs=dse_top,
                                                              num_outputs=layer_dim,
                                                              activation_fn=tf.nn.relu,
                                                              weights_initializer=xavier_initializer(),
                                                              trainable=True,
                                                              scope=param_namescope)
                        # General state extraction
                        with tf.variable_scope("GSE"):
                            layer_num = -1
                            layer_units = [64, 64]
                            gse_top = hlc_input_layer
                            if len(layer_units) > 0:
                                for layer_dim in layer_units:
                                    layer_num += 1
                                    layer_name = 'layer_'+repr(layer_num+1)
                                    param_namescope = 'hidden/'+layer_name
                                    gse_top = fully_connected(inputs=gse_top,
                                                              num_outputs=layer_dim,
                                                              activation_fn=tf.nn.relu,
                                                              weights_initializer=xavier_initializer(),
                                                              trainable=True,
                                                              scope=param_namescope)

                        hlc_concat = tf.concat([dse_top, gse_top], axis=-1)
                        hlc_output = fully_connected(inputs=hlc_concat,
                                                     num_outputs=sum(flat_hlc_output_dims),
                                                     activation_fn=tf.nn.tanh,
                                                     weights_initializer=xavier_initializer(),
                                                     trainable=True,
                                                     scope='outputs')

                        grip, position_command, orientation_command = split_and_fold(hlc_output, flat_hlc_output_dims,
                                                                                     hlc_outputs_dims, hlc_outputs_names)

                        # Quaternions need to have norm 1
                        orientation_command_normalized = tf.nn.l2_normalize(orientation_command)

                    # Low-Level Controller for which weights will be loaded from checkpoint
                    # TODO implement loading the part of graph from .pb file
                    temp_scope = "/LLC"
                    with tf.variable_scope(self.parameters_namescope + temp_scope):
                        # path = "/home/wojciech/.noesis/proc/train_kinova_ppo_pd/2019-04-09-17-11-43/" \
                        #        "graphs/kinova_ppo_graph/graph.pb"
                        # saver = tf.train.import_meta_graph(path)
                        # saver_def = saver.as_saver_def()
                        # print(saver_def.filename_tensor_name)
                        # print(saver_def.restore_op_name)
                        #
                        # for op in graph.get_operations():
                        #     print(op.name)

                        layer_units = [128, 128]
                        layer_num = -1
                        llc_top = tf.concat([position_command, orientation_command_normalized, llc_input_layer], axis=-1)
                        if len(layer_units) > 0:
                            for layer_dim in layer_units:
                                layer_num += 1
                                layer_name = 'layer_'+repr(layer_num+1)
                                param_namescope = 'hidden/'+layer_name
                                llc_top = fully_connected(inputs=llc_top,
                                                          num_outputs=layer_dim,
                                                          activation_fn=tf.nn.tanh,
                                                          weights_initializer=xavier_initializer(),
                                                          trainable=llc_trainable,
                                                          scope=param_namescope)

                        llc_output = fully_connected(inputs=llc_top,
                                                     num_outputs=sum(flat_torque_output_dims),
                                                     activation_fn=None,
                                                     weights_initializer=xavier_initializer(),
                                                     trainable=llc_trainable,
                                                     scope='outputs')

                        output_layer = tf.concat([llc_output, grip, position_command, orientation_command], axis=-1)

                    # Loading weights from previously saved checkpoint
                    # (automatically separating weights for StochasticPolicy function)
                    ckpt_path = "/home/wojciech/Documents/Code/src/noesis_physx/kinova_physx/pretrained_graphs/" \
                                "llc_pd/kinova_ppo_graph/checkpoints/graph"
                    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
                    var_to_shape_map = reader.get_variable_to_shape_map()
                    # Do not load variables connected to optimizer
                    new_map = {k: v for k, v in var_to_shape_map.items()
                               if self.parameters_namescope in k and "Optimizer" not in k}
                    new_keys = list(new_map.keys())

                    # Creating mapping from names saved in checkpoint to names of operations defined in this graph
                    graph_vars = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    new_dict = {k: v for k in new_keys for v in graph_vars
                                if v.name.replace(':0', '').replace(temp_scope, '') in k
                                or k in v.name.replace(':0', '').replace(temp_scope, '')}

                    # Creating saver
                    saver = tf.train.Saver(new_dict, name=self.operations_namescope + "/save")
                    saver_def = saver.as_saver_def()

                    # Tensor name and operation name used from C++ side to execute weights restoration
                    # print(saver_def.filename_tensor_name)
                    # print(saver_def.restore_op_name)

                # simple MLP for StateValue
                else:
                    with tf.variable_scope(self.parameters_namescope):
                        layer_units = [128, 128]
                        layer_num = -1
                        top = input_layer
                        if len(layer_units) > 0:
                            for layer_dim in layer_units:
                                layer_num += 1
                                layer_name = 'layer_'+repr(layer_num+1)
                                param_namescope = 'hidden/'+layer_name
                                top = fully_connected(inputs=top,
                                                      num_outputs=layer_dim,
                                                      activation_fn=tf.nn.relu,
                                                      weights_initializer=xavier_initializer(),
                                                      trainable=True,
                                                      scope=param_namescope)

                        output_layer = fully_connected(inputs=top,
                                                       num_outputs=sum(flat_output_dims),
                                                       activation_fn=None,
                                                       weights_initializer=xavier_initializer(),
                                                       trainable=True,
                                                       scope='outputs')

                with tf.name_scope(opscope):
                    # Fully-connected output layer
                    with tf.name_scope('outputs'):
                        # Check for multiple outputs and split if true
                        outputs = split_and_fold(output_layer, flat_output_dims, outputs_dims, outputs_names)

        # Store inputs
        self.add_input_node(inputs)
        # Store outputs
        self.add_output_node(outputs)


# Register architecture implementation
register("ARCH", MultiLayerPerceptron)

# EOF
