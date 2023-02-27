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

"""Graph-level implementations of common policy function-approximators.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp

from tensorflow.contrib.layers import fully_connected, xavier_initializer
from noesis.core.tensor import is_specs, flatten_specs, flatten_and_concatenate, split_and_fold, variables_accessors


# TODO: Define base class for Policy
class DiagonalGaussianPolicy(tf.Module):
    """
    TODO
    """
    def __init__(self,
                 latent,
                 actions_specs,
                 name="DiagonalGaussianPolicy",
                 init_stddev=1.0,  # TODO
                 min_stddev=0.0,  # TODO
                 stddev_from_latent=False):
        super(DiagonalGaussianPolicy, self).__init__(name=name)
        # TODO: Define base class for Model/Arch to assume existance of latent.output
        assert isinstance(latent, tf.Module)
        assert is_specs(actions_specs)
        # TODO
        self.__latent = latent.output
        self.__actions_specs = actions_specs
        self.__theta = tf1.get_collection(scope=latent.name_scope.name, key=tf1.GraphKeys.TRAINABLE_VARIABLES)
        # TODO
        self.__pd = None
        self.__mode_op = None
        self.__sample_op = None
        self.__pd_fixed = None
        # Create lists for action specs
        flat_actions_specs = flatten_specs(self.__actions_specs)
        self.__dtype = flat_actions_specs.dtype
        # TODO
        with tf.name_scope(self.name_scope.name):
            # Define flat action distribution
            with tf.name_scope("latent"):
                # TODO
                with tf.name_scope("mean"):
                    # TODO: Make configurable
                    mw0 = tf.cast(x=float(1e-3), dtype=self.__dtype)
                    mb0 = 0.0
                    mfn = None
                    flat_mean = fully_connected(
                        inputs=self.__latent,
                        num_outputs=int(sum(flat_actions_specs.shape.as_list())),
                        activation_fn=mfn,
                        weights_initializer=tf.random_uniform_initializer(-mw0, mw0),
                        biases_initializer=tf.initializers.constant(mb0),
                        scope=self.name_scope.name + "latent/mean",
                        trainable=True
                    )
                # TODO
                with tf.name_scope("stddev"):
                    if stddev_from_latent:
                        # Add state-dependent stddev
                        flat_stddev = fully_connected(
                            inputs=self.__latent,
                            num_outputs=int(sum(flat_actions_specs.shape.as_list())),
                            activation_fn=tf.exp,
                            weights_initializer=tf.zeros_initializer(),
                            biases_initializer=tf.constant_initializer(np.log(init_stddev)),
                            scope=self.name_scope.name + "latent/stddev",
                            trainable=True
                        )
                    else:
                        # Add state-independent stddev
                        log_stddev = [
                            tf1.get_variable(
                                name=self.name_scope.name + "latent/stddev/" + spec.name + "/parameter",
                                shape=spec.shape,
                                dtype=spec.dtype,
                                initializer=tf.constant_initializer(np.log(init_stddev)),
                                trainable=True
                            ) for spec in self.__actions_specs
                        ]
                        stddev = []
                        stddev_in = []
                        stddev_set_op = []
                        for k, w in enumerate(log_stddev):
                            spec = self.__actions_specs[k]
                            name = spec.name
                            dtype = spec.dtype
                            shape = spec.shape
                            with tf.name_scope(name):
                                stddev.append(tf.exp(w, name='value'))
                                stddev_in.append(tf1.placeholder(name='input', dtype=dtype, shape=shape))
                                stddev_set_op.append(tf1.assign(ref=w, value=tf1.log(stddev_in[k]), name='set_op'))
                        flat_stddev = flatten_and_concatenate(stddev, name="flat")
                        min_stddevp_param = tf.constant(value=min_stddev, shape=flat_stddev.shape.as_list(), dtype=self.__dtype)
                        flat_stddev = tf.maximum(flat_stddev, min_stddevp_param)

            # Define the action distribution
            with tf.name_scope("action_distribution"):
                self.__pd = tfp.distributions.MultivariateNormalDiag(loc=flat_mean, scale_diag=flat_stddev)

                with tf.name_scope("sample"):
                    flat_sample = self.__pd.sample()
                    self.__sample_op = split_and_fold(flat_sample, self.__actions_specs)

                with tf.name_scope("mode"):
                    flat_mode = self.__pd.mode()
                    self.__mode_op = split_and_fold(flat_mode, self.__actions_specs)

                self.__sample_shape = flat_sample.shape.as_list()
                self.__sample_shape = [-1 if not x else int(x) for x in self.__sample_shape]

                with tf.name_scope("fixed"):
                    self.__pd_fixed = tfp.distributions.MultivariateNormalDiag(
                        loc=tf.stop_gradient(flat_mean),
                        scale_diag=tf.stop_gradient(flat_stddev)
                    )

            # Store the list of trainable variables
            self.__theta += tf1.get_collection(scope=self.name_scope.name, key=tf1.GraphKeys.TRAINABLE_VARIABLES)

            # Define the action distribution
            with tf.name_scope("parameters"):
                var_ops = variables_accessors(var_list=self.__theta, dtype=self.__dtype)
                self.__theta_size_op = var_ops[0]
                self.__theta_in = var_ops[1]
                self.__theta_in_set_op = var_ops[2]
                self.__theta_in_get_op = var_ops[3]

    @property
    def actions_specs(self):
        return self.__actions_specs

    @property
    def pd(self):
        return self.__pd

    @property
    def mode(self):
        return self.__mode_op

    @property
    def sample(self):
        return self.__sample_op

    @property
    def sample_shape(self):
        return self.__sample_shape

    @property
    def pd_fixed(self):
        return self.__pd_fixed

    @property
    def parameters(self):
        return self.__theta


class CategoricalPolicy(tf.Module):
    """
    TODO
    """
    def __init__(self,
                 latent,
                 actions_specs,
                 actions_categories,
                 name="CategoricalPolicy"):
        super(CategoricalPolicy, self).__init__(name=name)
        assert isinstance(latent, tf.Module)
        assert is_specs(actions_specs)
        assert len(actions_specs) == len(actions_categories)
        # TODO
        self.__latent = latent.output
        self.__actions_specs = actions_specs
        self.__actions_categories = actions_categories[0]
        self.__theta = tf1.get_collection(scope=latent.name_scope.name, key=tf1.GraphKeys.TRAINABLE_VARIABLES)
        self.__dtype = self.__actions_specs[0].dtype
        self.__action_name = self.__actions_specs[0].name
        # TODO
        self.__pd = None
        self.__mode_op = None
        self.__sample_op = None
        self.__sample_shape = None
        self.__pd_fixed = None

        # Construct sub-graph operations
        with tf.name_scope(self.name_scope.name):
            # Add a fully connected layer defining the latent output of the model
            with tf.name_scope("latent"):
                with tf.name_scope("logits"):
                    actions_logits = fully_connected(
                        inputs=self.__latent,
                        num_outputs=int(len(self.__actions_categories)),
                        activation_fn=None,
                        trainable=True,
                        scope=self.name_scope.name + "latent/logits"
                    )
                    print("POLICY: actions_logits: ", actions_logits)

            # Define the policy distribution
            with tf.name_scope("action_distribution"):
                self.__pd = tfp.distributions.Categorical(logits=actions_logits)

                with tf.name_scope("mode"):
                    action_mode = self.__pd.mode()
                    self.__mode_op = [tf.reshape(tf.cast(action_mode, dtype=self.__dtype), shape=[-1, 1], name=self.__action_name)]

                with tf.name_scope("sample"):
                    action_sample = self.__pd.sample()
                    self.__sample_op = [tf.reshape(tf.cast(action_sample, dtype=self.__dtype), shape=[-1, 1], name=self.__action_name)]

                self.__sample_shape = action_sample.shape.as_list()
                self.__sample_shape = [-1 if not x else int(x) for x in self.__sample_shape]

                with tf.name_scope("fixed"):
                    self.__pd_fixed = tfp.distributions.Categorical(logits=tf.stop_gradient(actions_logits))

            # Store the list of trainable variables
            self.__theta += tf1.get_collection(scope=self.name_scope.name, key=tf1.GraphKeys.TRAINABLE_VARIABLES)

            # Define the action distribution
            with tf.name_scope("parameters"):
                var_ops = variables_accessors(var_list=self.__theta, dtype=self.__dtype)
                self.__theta_size_op = var_ops[0]
                self.__theta_in = var_ops[1]
                self.__theta_in_set_op = var_ops[2]
                self.__theta_in_get_op = var_ops[3]

    @property
    def actions_specs(self):
        return self.__actions_specs

    @property
    def pd(self):
        return self.__pd

    @property
    def mode(self):
        return self.__mode_op

    @property
    def sample(self):
        return self.__sample_op

    @property
    def sample_shape(self):
        return self.__sample_shape

    @property
    def pd_fixed(self):
        return self.__pd_fixed

    @property
    def parameters(self):
        return self.__theta


# EOF
