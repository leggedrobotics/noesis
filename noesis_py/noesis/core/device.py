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

"""Device availability and identification functions"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.python.client import device_lib


def available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


def available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def assert_device_exists(device):
    for dev in device_lib.list_local_devices():
        if device == dev.name:
            return
    # Fall-through means that no valid device matched
    raise ValueError("Device '%s' does not exist on current system. Please run available_devices()'." % device)


def check_device(device_name):
    if device_name == "CPU":
        device = available_cpus()[0]
    elif device_name == "GPU":
        gpus = available_gpus()
        if not gpus:
            raise ValueError("There are no GPUs available on the current system. Please use '/device:CPU:0' instead.")
        else:
            device = gpus[0]
    else:
        assert_device_exists(device_name)
        device = device_name
    return device

# EOF
