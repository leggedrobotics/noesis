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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from functools import reduce
from operator import mul

import tensorflow as tf
import tensorflow.compat.v1 as tf1


# TODO: document
def max_dtype(items):
    dtypes = [item.dtype for item in items]
    dtype = tf.bool
    for dt in dtypes:
        dtype = dt if dt.size > dtype.size else dtype
    return dtype


# TODO: document
def size_of_tensor(tensor):
    assert tf.is_tensor(tensor)
    return reduce((lambda x, y: x * y), tensor.shape.as_list())


# TODO: document
def is_specs(specs):
    result = isinstance(specs, list) or isinstance(specs, dict)
    for spec in specs:
        result &= isinstance(spec, tf.TensorSpec)
    return result


# TODO: document
def placeholders_from_specs(specs, batched=False):
    assert is_specs(specs)
    nodes = []
    for spec in specs:
        assert isinstance(spec, tf.TensorSpec)
        shape = [None] + spec.shape if batched else spec.shape
        dtype = spec.dtype
        nodes.append(tf1.placeholder(name=spec.name, dtype=dtype, shape=shape))
    return nodes


# TODO: document
def specs_from_tensors(tensors, strip_batching=False, strip_scope=False):
    assert isinstance(tensors, list)
    specs = []
    for tensor in tensors:
        assert tf.is_tensor(tensor)
        shape = [dim for dim in tensor.shape.as_list() if not (strip_batching and dim is None)]
        name = tf.TensorSpec.from_tensor(tensor).name  # TODO: Use strip_scope
        specs.append(tf.TensorSpec(name=name, shape=shape, dtype=tensor.dtype))
    return specs


# TODO: document
def flatten_specs(specs, dtype=None, name=None):
    assert is_specs(specs)
    if dtype is None:
        dtype = max_dtype(specs)
    else:
        assert isinstance(dtype, tf.DType)
    flattened = []
    for spec in specs:
        flattened.append(reduce((lambda x, y: x * y), spec.shape))
    flat = reduce((lambda x, y: x + y), flattened)
    return tf.TensorSpec(name=name, shape=flat, dtype=dtype)


# TODO: document
def flatten_and_concatenate(tensors, dtype=None, name=None):
    assert isinstance(tensors, list)
    if dtype is None:
        dtype = max_dtype(tensors)
    else:
        assert isinstance(dtype, tf.DType)
    batched = False
    flattened = []
    for tensor in tensors:
        assert tf.is_tensor(tensor)
        batched = tensor.shape.as_list()[0] is None
        shape = tensor.shape.as_list() if not batched else tensor.shape.as_list()[1:]
        reshaped = [reduce((lambda x, y: x * y), shape)]
        reshaped = [-1] + reshaped if batched else reshaped
        flattened.append(tf.reshape(tf.cast(tensor, dtype=dtype), shape=reshaped))
    return tf.concat(flattened, axis=1 if batched else 0, name=name)


# TODO: document
def split_and_fold(tensor, specs):
    assert tf.is_tensor(tensor)
    assert is_specs(specs)
    batched = tensor.shape.as_list()[0] is None
    splits = [reduce((lambda x, y: x * y), spec.shape.as_list()) for spec in specs]
    split = tf.split(value=tensor, num_or_size_splits=splits, axis=-1)
    folded = []
    for k, spec in enumerate(specs):
        shape = spec.shape.as_list()
        reshaped = [-1] + shape if batched else shape
        folded.append(tf.reshape(split[k], shape=reshaped, name=spec.name))
    return folded


# TODO: document
def step_counter(name=None):
    name_scope = "step_counter" if name is None else name
    variable_scope = tf1.get_default_graph().get_name_scope()
    variable_scope = variable_scope + "/" + name_scope if variable_scope else name_scope
    step = tf1.get_variable(name=variable_scope + "/step",
                            shape=[],
                            dtype=tf.int64,
                            initializer=tf.zeros_initializer(),
                            trainable=False)
    with tf.name_scope(name_scope):
        reset_op = tf1.assign(step, 0, name='reset_op', use_locking=True)
        increment_op = tf1.assign(step, step + 1, name='increment_op', use_locking=True)
    # Collect all tensor nodes
    ops = [reset_op, increment_op]
    return step, ops


# TODO: document
def modifiable(initial, dtype, name):
    variable_scope = tf1.get_default_graph().get_name_scope()
    variable = tf1.get_variable(
        name=variable_scope + "/" + name + "/variable",
        dtype=dtype,
        initializer=tf.constant(initial, dtype=dtype),
        trainable=False
    )
    with tf.name_scope(name):
        variable_in = tf1.placeholder(name='input', shape=[], dtype=dtype)
        variable_set_op = tf1.assign(variable, tf.reshape(variable_in, []), name='set_op')
        variable_get_op = tf.identity(variable.read_value(), name='value')
    ops = [variable_in, variable_set_op, variable_get_op]
    return variable, ops


# TODO: document
def variables_accessors(var_list, dtype):
    # Retrieve lists of parameters
    var_size_list = [reduce(mul, param.get_shape().as_list(), 1) for param in var_list]
    var_size = sum(var_size_list)
    # Create operations for accessing sub-graph parameters
    var_size_op = tf.identity(tf.constant(value=var_size, dtype=tf.int32), name='get_size_op')
    var_in = tf1.placeholder(dtype=dtype, name='input')
    var_in_split = tf.split(value=var_in, num_or_size_splits=var_size_list, axis=1)
    var_nodes = [[tf.reshape(param, [-1])] for param in var_list]
    var_get_op = tf.reshape(tf.concat(var_nodes, name='get_op', axis=1), shape=[-1, 1])
    var_set_ops = []
    for idx, param in enumerate(var_list):
        reshaped_input_vector = tf.reshape(var_in_split[idx], shape=tf.shape(param))
        var_set_ops += [param.assign(reshaped_input_vector)]
    var_set_op = tf.group(*var_set_ops, name='set_op')
    return [var_size_op, var_in, var_get_op, var_set_op]

# EOF
