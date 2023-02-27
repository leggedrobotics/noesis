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

"""Graph-level implementations of common policy function-approximators.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from noesis.core.base import Function
from noesis.core.registration import register


class DiagonalGaussianPolicy(Function):
    """
    Stochastic Policy with learned (diagonal) covariance.
    """
    def __init__(self, name, inputs_spec, outputs_spec, func_config, arch_type, arch_config, dtype, device, verbose):
        super(DiagonalGaussianPolicy, self).__init__(name, inputs_spec, outputs_spec, func_config, arch_type, arch_config, dtype, device,
                                                     verbose=verbose)
        # Allow all inputs to be fed into the function architecture
        self.arch.inputs.specs = self.inputs.specs
        # Only action mean is to be output by the function architecture
        self.arch.outputs.specs = [out for out in self.outputs.specs if '_mean' in out.name]

    def add_subgraph(self, graph):

        try:
            seg_in = self.get_input_node_by_name("img_seg")
            h_pos_in = self.get_input_node_by_name("h_pos")
            h_ori_in = self.get_input_node_by_name("h_ori")
            h_ang_in = self.get_input_node_by_name("h_ang")
            d_ang_in = self.get_input_node_by_name("d_ang")
            ee_pos_in = self.get_input_node_by_name("ee_pos")
        except ValueError as err:
            print(err)
            seg_in = None
            h_pos_in = None
            h_ori_in = None
            h_ang_in = None
            d_ang_in = None
            ee_pos_in = None

        # Configure sub-graph inputs
        action_mean_in = []
        action_dims = []
        action_name = []
        for action in self.arch.outputs.specs:
            action_mean_in.append(self.get_input_node_by_name(action.name))
            action_dims.append(self.get_output_dim_by_name(action.name))
            action_name.append(action.name.replace('_mean', ''))

        # Append all operations to the specified tensorflow graph
        with graph.as_default():
            with tf.device(self.device):

                # Sub-graph parameters
                with tf.variable_scope(self.parameters_namescope):
                    # Action distribution output parametrization
                    with tf.variable_scope('action_distribution'):
                        action_stddev_w = []
                        for k in range(0, len(action_dims)):
                            with tf.variable_scope(action_name[k] + '_stddev'):
                                action_stddev_w.append(tf.get_variable(name='weights',
                                                                       shape=action_dims[k],
                                                                       dtype=self.dtype,
                                                                       initializer=tf.ones_initializer,
                                                                       trainable=True))

                # Sub-graph operations
                with tf.name_scope(self.operations_namescope):
                    if seg_in is not None:
                        seg_bg = tf.identity(seg_in[:, 0, :, :], name="img_seg_bg")
                        seg_handle = tf.identity(seg_in[:, 1, :, :], name="img_seg_handle")
                        seg_door = tf.identity(seg_in[:, 2, :, :], name="img_seg_door")
                        seg_arm = tf.identity(seg_in[:, 3, :, :], name="img_seg_arm")
                        h_pos_out = tf.identity(h_pos_in, name="h_pos")
                        h_ori_out = tf.identity(h_ori_in, name="h_ori")
                        h_ang_out = tf.identity(h_ang_in, name="h_ang")
                        d_ang_out = tf.identity(d_ang_in, name="d_ang")
                        ee_pos_out = tf.identity(ee_pos_in, name="ee_pos")
                    # Action distribution input parametrization
                    with tf.name_scope('action_distribution'):
                        with tf.name_scope('stddev'):
                            action_stddev_in = []
                            action_stddev_set_op = []
                            action_stddev = []
                            for k in range(0, len(action_dims)):
                                action_stddev_in.append(tf.placeholder(dtype=self.dtype,
                                                                       shape=action_dims[k],
                                                                       name=action_name[k]+'_input'))
                                action_stddev_set_op.append(tf.assign(ref=action_stddev_w[k],
                                                                      value=tf.log(action_stddev_in[-1]),
                                                                      name=action_name[k]+'_set_op'))
                                action_stddev.append(tf.identity(input=tf.exp(action_stddev_w[k]), name=action_name[k]))

                        # Action distribution output operations
                        with tf.name_scope('mean'):
                            action_mean = []
                            for k in range(0, len(action_dims)):
                                action_mean.append(tf.identity(input=action_mean_in[k], name=action_name[k]))

        # Store inputs
        self.add_input_node(action_stddev_in)
        # Store outputs
        self.add_output_node(action_mean)
        self.add_output_node(action_stddev)
        if seg_in is not None:
            self.add_output_node(seg_bg)
            self.add_output_node(seg_handle)
            self.add_output_node(seg_door)
            self.add_output_node(seg_arm)
            self.add_output_node(h_pos_out)
            self.add_output_node(h_ori_out)
            self.add_output_node(h_ang_out)
            self.add_output_node(d_ang_out)
            self.add_output_node(ee_pos_out)


# Register architecture implementation
register("FUNC", DiagonalGaussianPolicy)

# EOF
