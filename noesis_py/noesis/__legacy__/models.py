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

"""Sub-graphs of supported model classes."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from noesis.core.base import Function
from noesis.core.registration import register
from noesis.common.configuration import *

class DeterministicModel(Function):
    """
    Deterministic model function implementation.
    """
    def __init__(self, name, inputs_spec, outputs_spec, func_config, arch_type, arch_config, dtype, device, verbose):
        super(DeterministicModel, self).__init__(name, inputs_spec, outputs_spec, func_config,
                                                 arch_type, arch_config, dtype, device, verbose)
        # Allow all inputs and outputs to be mapped the function architecture
        self.arch.inputs.specs = self.inputs.specs
        self.arch.outputs.specs = self.outputs.specs

    def add_subgraph(self, graph):

        # Configure sub-graph outputs
        outputs_in = []
        outputs_dims = []
        outputs_names = []
        for output in self.arch.outputs.specs:
            outputs_in.append(self.get_input_node_by_name(output.name))
            outputs_dims.append(self.get_output_dim_by_name(output.name))
            outputs_names.append(output.name)

        # Append all operations to the specified tensorflow graph
        with graph.as_default():
            with tf.device(self.device):

                # Sub-graph operations
                with tf.name_scope(self.operations_namescope):
                    with tf.name_scope("outputs"):
                        outputs = []
                        for k in range(0, len(outputs_in)):
                            outputs.append(tf.identity(input=outputs_in[k], name=outputs_names[k]))

        # Store inputs
        # NONE
        # Store outputs
        self.add_output_node(outputs)


# Register architecture implementation
register("FUNC", DeterministicModel)

class DiagonalGaussianModel(Function):
    """
    Deterministic model function implementation.
    """
    def __init__(self, name, inputs_spec, outputs_spec, func_config, arch_type, arch_config, dtype, device, verbose):
        super(DiagonalGaussianModel, self).__init__(name, inputs_spec, outputs_spec, func_config,
                                                    arch_type, arch_config, dtype, device, verbose)
        # Allow all inputs and outputs to be mapped the function architecture
        self.arch.inputs.specs = self.inputs.specs
        self.arch.outputs.specs = self.outputs.specs
        self._mean_activation = 'linear'
        self._mean_filters = self.outputs.specs[0].dims[-1]
        self._mean_kernel_size = 4
        self._cov_activation ='exponential'
        self._cov_filters = self.outputs.specs[1].dims[-1]
        self._cov_kernel_size = 4
        # read func config
        print(func_config)
        self._mean_activation, self._mean_kernel_size, self._cov_activation, self._cov_kernel_size = self.read_xml_config(func_config)

    def read_xml_config(self, config_element):
        mean_activation = config_element.iter("mean_activation").__next__().attrib["value"]
        mean_kernel_size = int(config_element.iter("mean_kernel_size").__next__().attrib["value"])

        cov_activation = config_element.iter("cov_activation").__next__().attrib["value"]
        cov_kernel_size = int(config_element.iter("cov_kernel_size").__next__().attrib["value"])
        return mean_activation, mean_kernel_size, cov_activation, cov_kernel_size

    def add_subgraph(self, graph):

        # Configure sub-graph outputs
        output_mean_name = self.arch.outputs.specs[0].name
        output_cov_name = self.arch.outputs.specs[1].name
        outputs_in = self.get_input_node_by_name("TCN_outputs")

        # Append all operations to the specified tensorflow graph
        with graph.as_default():
            with tf.device(self.device):

                # Split into mean and covariance
                with tf.variable_scope(self.parameters_namescope):
                    init = tf.keras.initializers.glorot_normal()
                    outputs_mean = tf.keras.layers.Conv1D(filters=self._mean_filters, kernel_size=self._mean_kernel_size,
                                                          padding="causal", kernel_initializer=init)(outputs_in)
                    outputs_mean = tf.keras.layers.Activation(self._mean_activation)(outputs_mean)

                    outputs_cov = tf.keras.layers.Conv1D(filters=self._cov_filters, kernel_size=self._cov_kernel_size,
                                                         padding="causal", kernel_initializer=init)(outputs_in)
                    outputs_cov = tf.keras.layers.Activation(self._cov_activation)(outputs_cov)
                    # add correct names
                with tf.name_scope(self.operations_namescope):
                    with tf.name_scope("outputs"):
                        time_dim = outputs_mean.shape[1].value
                        half_dim = int(time_dim / 2)
                        outputs_mean=tf.identity(input=outputs_mean[:,half_dim:,:], name=output_mean_name)
                        outputs_cov=tf.identity(input=outputs_cov[:,half_dim:,:], name=output_cov_name)

        # Store inputs
        # NONE
        # Store outputs
        self.add_output_node(outputs_mean)
        self.add_output_node(outputs_cov)


# Register architecture implementation
register("FUNC", DiagonalGaussianModel)

# EOF
