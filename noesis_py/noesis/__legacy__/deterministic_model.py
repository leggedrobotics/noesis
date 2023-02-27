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

"""Sub-graphs of supported model classes."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from noesis.core.base import Function
from noesis.core.registration import register


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

# EOF
