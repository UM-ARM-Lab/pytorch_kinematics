# Copyright 2017-2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Various helper functions and classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import six

DEFAULT_ENCODING = sys.getdefaultencoding()


def to_binary_string(s):
    """Convert text string to binary."""
    if isinstance(s, six.binary_type):
        return s
    return s.encode(DEFAULT_ENCODING)


def to_native_string(s):
    """Convert a text or binary string to the native string format."""
    if six.PY3 and isinstance(s, six.binary_type):
        return s.decode(DEFAULT_ENCODING)
    elif six.PY2 and isinstance(s, six.text_type):
        return s.encode(DEFAULT_ENCODING)
    else:
        return s
