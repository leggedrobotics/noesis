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

"""Message logging and console output."""

import time
import inspect

from os.path import basename, splitext
from termcolor import colored


def __message(level, color, *msg):
    """Appends date-time to message and applies colored formatting."""
    module = inspect.getmodule(inspect.stack()[2][0])
    out = "[%s] [%s]: [%s]: " % (level, time.strftime("%Y.%m.%d::%H%M%S"), splitext(basename(module.__file__))[0])
    out += ''.join(msg[0])
    print(colored(out, color))
    # TODO use python.logging to actually log messages and output them to appropriate files


def info(*msg):
    """Output an INFO (general-purpose) message."""
    __message("INFO", None, msg)


def notify(*msg):
    """Output a NOTIFICATION message. Indicates correct operation, but should capture users attention."""
    __message("INFO", 'blue', msg)


def warn(*msg):
    """Output a WARNING message. Indicates possible problem or misbehavior."""
    __message("INFO", 'yellow', msg)


def error(*msg):
    """Output an ERROR message."""
    __message("INFO", 'red', msg)

# EOF
