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

"""A collection of common loss functions."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.compat.v1 as tf1


def huber(target, value, extra_cost=0, delta=1e-1, name=None):
    with tf.name_scope("huber"):
        cutoff_line = tf.constant(value=delta, dtype=target.dtype)
        residual = tf.sqrt(tf.reduce_sum(tf.square(target - value), axis=1))
        condition = tf.less(residual, delta)
        small_res = 0.5 * tf.square(residual)
        large_res = cutoff_line * residual - 0.5 * tf.square(cutoff_line)
        return tf.add(tf.reduce_mean(tf.where(condition, small_res, large_res), 0), extra_cost, name=name)


def infimum(target, value, extra_cost=0, under_est_error=1e-3, name=None):
    with tf.name_scope("infimum"):
        residual = target - value
        condition = tf.less(residual, 0)
        small_res = 0.5 * tf.square(residual)
        large_res = under_est_error * residual
        return tf.add(tf.reduce_mean(tf.where(condition, small_res, large_res), 0), extra_cost, name=name)


def squared(target, value, extra_cost=0, name=None):
    with tf.name_scope("squared"):
        return tf.add(tf.reduce_mean(tf.square(target - value)), extra_cost, name=name)


def squared_clipped(target, value, value_prev, extra_cost=0, clip_range=0.2, name=None):
    """
     Clipping-based trust region loss
     (https://github.com/openai/baselines/blob/master/baselines/pposgd/pposgd_simple.py)
    """
    with tf.name_scope("squared_clipped"):
        clipped_value = value_prev + tf.clip_by_value(value - value_prev, -clip_range, clip_range)
        loss1 = tf.square(value - target)
        loss2 = tf.square(clipped_value - target)
        loss = tf.add(0.5 * tf.reduce_mean(tf.maximum(loss1, loss2)), extra_cost, name=name)
        return loss


# EOF
