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

"""Classes and functions used to construct tensorflow-based computation graphs."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from noesis.core.device import check_device


class Graph:
    """
    Class for construction of TensorFlow-based computation graphs.
    """
    def __init__(self, name="graph", seed=0, verbose=False):
        # Initialize internals
        self.__name = name
        # Set the global operation device
        self.__device = check_device('/device:CPU:0')
        # Allocate the tensorflow computation graph
        self.__graph = tf.Graph()
        # Graph-level configurations
        with self.__graph.as_default():
            with tf.device(self.__device):
                tf1.set_random_seed(int(seed))
                self.__global_step = tf1.train.create_global_step()
        # Initialize internals to be added at graph finalization
        self.__local_vars_init_op = None
        self.__global_vars_init_op = None
        self.__global_saver_op = None
        self.__global_saver_def = None

    @property
    def name(self):
        return self.__name

    @property
    def graph(self):
        return self.__graph

    def as_default(self):
        return self.__graph.as_default()

    def finalize(self):
        # Create global initialization operation
        with self.__graph.as_default():
            with tf.device(self.__device):
                with tf.name_scope("Global"):
                    # Global operations
                    with tf.name_scope("global_step"):
                        tf1.assign(self.__global_step, 0, name='reset_op', use_locking=True)
                        tf1.assign(self.__global_step, self.__global_step + 1, name='increment_op', use_locking=True)
                    self.__global_vars_init_op = tf1.variables_initializer(name='global_vars_init_op',
                                                                           var_list=tf1.global_variables())
                    self.__local_vars_init_op = tf1.variables_initializer(name='local_vars_init_op',
                                                                          var_list=tf1.local_variables())
                    self.__global_saver_op = tf1.train.Saver(max_to_keep=None,
                                                             name="global_saver",
                                                             restore_sequentially=False,
                                                             pad_step_number=True,
                                                             save_relative_paths=True,
                                                             filename=self.name)
                    self.__global_saver_def = self.__global_saver_op.as_saver_def()
        # Finalize the internal TF graph, setting it to read only.
        # NOTE: This protects the graph from accidentally adding new ops at run-time.
        self.__graph.finalize()

    def to_event(self, path=None):
        assert self.__global_saver_def, "Graph has not been finalized using `.finalize()`!."
        writer = tf1.summary.FileWriter(logdir=path, graph=self.__graph)
        writer.flush()

    def to_protobuf(self, path=None, as_text=False):
        assert self.__global_saver_def, "Graph has not been finalized using `.finalize()`!."
        filename = os.path.join(path if path else os.path.curdir, self.name + '.pb')
        with tf1.Session(graph=self.__graph) as session:
            tf1.train.export_meta_graph(filename=filename,
                                        graph_def=session.graph_def,
                                        saver_def=self.__global_saver_def,
                                        as_text=as_text,
                                        graph=session.graph)
        return filename

# EOF
