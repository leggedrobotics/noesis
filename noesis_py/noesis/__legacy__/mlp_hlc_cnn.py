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
from tensorflow.contrib.layers import fully_connected, xavier_initializer, flatten

from noesis.common.configuration import *
from noesis.core.base import Architecture
from noesis.core.registration import register
from noesis.common.tensor import flatten_and_concatenate, split_and_fold, flatten_dimensions


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

        gse_inputs = inputs[:-3]
        gse_inputs_dims = inputs_dims[:-3]
        image = inputs[-3]
        image_dims = inputs_dims[-3]
        smoothed_h_pos = inputs[-2]
        smoothed_h_pos_dims = inputs_dims[-2]
        smoothed_h_ori = inputs[-1]
        smoothed_h_ori_dims = inputs_dims[-1]

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
                        gse_input_layer_in = flatten_and_concatenate(gse_inputs, gse_inputs_dims)
                        # Fully-connected input layer
                        gse_input_layer = tf.identity(gse_input_layer_in, name='gse_input')
                        image_input_layer = tf.expand_dims(image, 1, name='image_input')

                with tf.variable_scope(self.parameters_namescope):

                    # Door state extraction (load pre-trained CNN)
                    with tf.variable_scope("DSE"):
                        graph_path = "/home/wojciech/Documents/Code/src/noesis_physx/kinova_physx/training/" \
                                     "unet/graphs/20190527_122302/model_frozen_full.pb"
                        with tf.gfile.FastGFile(graph_path, 'rb') as f:
                            graph_def = tf.GraphDef()
                            graph_def.ParseFromString(f.read())

                        input_map = {"dataset/IteratorGetNext:0": image_input_layer}
                        ret_elems = ["down_output:0", "handle_position:0", "handle_orientation:0",
                                     "handle_angle:0", "door_angle:0", "end_effector_position:0",
                                     "output:0"]
                        down, h_pos, h_ori, h_ang, d_ang, ee_pos, seg = tf.import_graph_def(graph_def,
                                                                                            input_map=input_map,
                                                                                            return_elements=ret_elems,
                                                                                            name="CNN")
                        print("Loaded CNN graph from ", graph_path)

                        # Which features from CNN to use
                        # "concat", "h_pos_h_ori", "smoothed_h_pos_h_ori", "h_pos", "smoothed_h_pos", "latent"
                        output_mode = "latent"

                        if output_mode == "concat":
                            dse_top = tf.concat([h_pos, h_ori, h_ang, d_ang], axis=1)
                        elif output_mode == "h_pos_h_ori":
                            dse_top = tf.concat([h_pos, h_ori], axis=1)
                        elif output_mode == "smoothed_h_pos_h_ori":
                            dse_top = tf.concat([smoothed_h_pos, smoothed_h_ori], axis=1)
                        elif output_mode == "h_pos":
                            dse_top = h_pos
                        elif output_mode == "smoothed_h_pos":
                            dse_top = smoothed_h_pos
                        elif output_mode == "latent":
                            dse_top = flatten(down)
                            layer_num = -1
                            if len(self._layer_units) > 1:
                                for layer_dim in self._layer_units[:-1]:
                                    layer_num += 1
                                    layer_name = 'layer_'+repr(layer_num+1)
                                    param_namescope = 'hidden/'+layer_name
                                    dse_top = fully_connected(inputs=dse_top,
                                                              num_outputs=layer_dim,
                                                              activation_fn=self._layer_activations[layer_num],
                                                              weights_initializer=xavier_initializer(),
                                                              trainable=True,
                                                              scope=param_namescope)
                        else:
                            print("Unsupported output mode!")
                            raise

                    # General state extraction
                    with tf.variable_scope("GSE"):
                        layer_num = -1
                        gse_top = gse_input_layer
                        if len(self._layer_units) > 1:
                            for layer_dim in self._layer_units[:-1]:
                                layer_num += 1
                                layer_name = 'layer_'+repr(layer_num+1)
                                param_namescope = 'hidden/'+layer_name
                                gse_top = fully_connected(inputs=gse_top,
                                                          num_outputs=layer_dim,
                                                          activation_fn=self._layer_activations[layer_num],
                                                          weights_initializer=xavier_initializer(),
                                                          trainable=True,
                                                          scope=param_namescope)

                    # DSE and GSE concatenation
                    with tf.variable_scope("concat"):
                        output_concat = tf.concat([dse_top, gse_top], axis=-1)
                        if len(self._layer_units) > 0:
                            layer_dim = self._layer_units[-1]
                            layer_num += 1
                            output_concat = fully_connected(inputs=output_concat,
                                                            num_outputs=layer_dim,
                                                            activation_fn=self._layer_activations[layer_num],
                                                            weights_initializer=xavier_initializer(),
                                                            trainable=True)

                    with tf.variable_scope('outputs'):
                        dim_in = layer_dim
                        wo = tf.get_variable(name='weights',
                                             initializer=tf.random_uniform(dtype=self.dtype,
                                                                           shape=[dim_in,
                                                                                  sum(flat_output_dims)],
                                                                           minval=-self._max_initial_value,
                                                                           maxval=self._max_initial_value))
                        bo = tf.get_variable(name='biases',
                                             initializer=tf.random_uniform(dtype=self.dtype,
                                                                           shape=[sum(flat_output_dims)],
                                                                           minval=-self._max_initial_value,
                                                                           maxval=self._max_initial_value))

                with tf.name_scope(opscope):
                    # Fully-connected output layer
                    with tf.name_scope('outputs'):
                        # Configure the output layer
                        if self._layer_activations[-1] is None:
                            output_layer = tf.add(tf.matmul(output_concat, wo), bo)
                        else:
                            output_layer = self._layer_activations[-1](tf.matmul(output_concat, wo) + bo)
                        # Check for multiple outputs and split if true
                        outputs = split_and_fold(output_layer, flat_output_dims, outputs_dims, outputs_names)
                        output_seg = tf.identity(seg, name="img_seg")
                        output_h_pos = tf.identity(h_pos, name="h_pos")
                        output_h_ori = tf.identity(h_ori, name="h_ori")
                        output_h_ang = tf.identity(h_ang, name="h_ang")
                        output_d_ang = tf.identity(d_ang, name="d_ang")
                        output_ee_pos = tf.identity(ee_pos, name="ee_pos")

        # Store inputs
        self.add_input_node(inputs)
        # Store outputs
        self.add_output_node(outputs)
        self.add_output_node(output_seg)
        self.add_output_node(output_h_pos)
        self.add_output_node(output_h_ori)
        self.add_output_node(output_h_ang)
        self.add_output_node(output_d_ang)
        self.add_output_node(output_ee_pos)


# Register architecture implementation
register("ARCH", MultiLayerPerceptron)

# EOF
