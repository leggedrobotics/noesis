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

"""Gradient helper functions."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from operator import mul
from functools import reduce
from noesis.core.tensor import size_of_tensor


# TODO: document
def clip_gradient_by_max_norm(grads, max_norm, norm=None, name=None):
    gradients, global_norm = tf.clip_by_global_norm(grads, use_norm=norm, clip_norm=max_norm, name=name)
    return gradients, global_norm


# TODO: document
def flatten_gradient(grads, variables, name=None):
    flat_grads = tf.concat([tf.reshape(grad, [size_of_tensor(v)]) for (v, grad) in zip(variables, grads)], axis=0)
    flat_grads = tf.reshape(tensor=flat_grads, shape=[1, -1], name=name)
    return flat_grads


# TODO: document
def compute_flat_gradient(loss, variables, max_norm=None, name=None):
    grads = tf.gradients(loss, variables)
    norm = tf1.global_norm(grads)
    # Optionally apply gradient clipping
    if max_norm is not None:
        grads, _ = clip_gradient_by_max_norm(grads=grads, max_norm=max_norm, norm=norm)
    # Flatten gradients
    flat_grads = tf.reshape(
        tensor=tf.concat([tf.reshape(grad, [size_of_tensor(v)]) for (v, grad) in zip(variables, grads)], axis=0),
        shape=[1, -1],
        name=name)
    return flat_grads, norm


# TODO: document
def merge_flat_gradient_with_vars(gradient, variables):
    split_gradients = tf.split(
        value=gradient,
        num_or_size_splits=[reduce(mul, param.get_shape().as_list(), 1) for param in variables],
        axis=1)
    grads = []
    for grad, param in zip(split_gradients, variables):
        grads += [tf.reshape(grad, tf.shape(param))]
    return zip(grads, variables)


# TODO: document
def conjugate_gradient_descent(dtype, eval, b, max_iter, tolerance, damping):
    """
    NOTE: Not applicable to recurrent network(dynamic rnn)
    """

    # Internal condition callback
    def condition(i, p, r, x, rdotr):
        return tf.logical_and(tf.less(tol, rdotr), tf.less(i, max_iter))

    # Internal step callback
    def iteration(i, p, r, x, rdotr):
        Ap = eval(p)
        Ap += damping * p  # cg damping
        a = tf.divide(rdotr, tf.reduce_sum(tf.matmul(p, tf.transpose(Ap))))
        x += a * p
        r -= a * Ap
        newrdotr = tf.reduce_sum(tf.matmul(r, tf.transpose(r)))
        mu = newrdotr / rdotr
        p = r + mu * p
        return i + 1, p, r, x, newrdotr
    # Define loop variables
    r = tf.identity(b)
    p = tf.identity(b)
    x = tf.zeros(shape=tf.shape(b), dtype=dtype)
    tol = tf.cast(tolerance, dtype=dtype)
    initial_i = tf.constant(0, dtype=tf.int64)
    rdotr = tf.reduce_sum(tf.matmul(r, tf.transpose(r)))  # vector dot product
    # Define looping operation
    i, p, r, x, rdotr = tf.while_loop(
        condition, iteration, [initial_i, p, r, x, rdotr],
        shape_invariants=[initial_i.get_shape(), b.get_shape(), b.get_shape(), b.get_shape(), rdotr.get_shape()])
    return x, rdotr


# EOF
