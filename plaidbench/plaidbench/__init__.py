# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pkg_resources
try:
    __version__ = pkg_resources.get_distribution("plaidbench").version
except pkg_resources.DistributionNotFound:
    __version__ = 'local'

SUPPORTED_NETWORKS = {
    'keras': [
        'densenet121',
        'densenet169',
        'densenet201',
        'imdb_lstm',
        'inception_resnet_v2',
        'inception_v3',
        'mobilenet',
        'mobilenet_v2',
        'nasnet_large',
        'nasnet_mobile',
        'resnet50',
        'vgg16',
        'vgg19',
        'xception',
    ],
}
