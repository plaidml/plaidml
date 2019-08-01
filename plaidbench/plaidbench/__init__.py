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

import argparse
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


def make_parser():
    # Create the parser outside of main() so the doc system can call this function
    # and thereby generate a web page describing these options. See docs/index.rst.
    parser = argparse.ArgumentParser()
    plaidargs = parser.add_mutually_exclusive_group()
    plaidargs.add_argument('--plaid', action='store_true', help="Use PlaidML as the backend.")
    plaidargs.add_argument('--caffe2', action='store_true', help="Use Caffe2 as the backend.")
    plaidargs.add_argument('--tf', action='store_true', help="Use TensorFlow as the backend.")
    plaidargs.add_argument(
        '--no-plaid',
        action='store_true',
        help="Use the non-PlaidML backend most appropriate to the chosen frontend")
    frontendargs = parser.add_mutually_exclusive_group()
    frontendargs.add_argument('--keras', action='store_true', help='Use Keras as the frontend')
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Use half-precision floats, setting floatx='float16'.")
    parser.add_argument('-v',
                        '--verbose',
                        action='count',
                        default=0,
                        help="Logging verbosity level (0..4).")
    parser.add_argument('--result',
                        default='/tmp/plaidbench_results',
                        help="Destination directory for results output.")
    parser.add_argument('--callgrind',
                        action='store_true',
                        help="Invoke callgrind during timing runs.")
    parser.add_argument('--no-warmup', action='store_true', help="Skip the warmup runs.")
    parser.add_argument('-n',
                        '--examples',
                        type=int,
                        default=None,
                        help="Number of examples to use.")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs per test.")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--train',
                        action='store_true',
                        help="Measure training performance instead of inference.")
    parser.add_argument('--blanket-run',
                        action='store_true',
                        help="Run all networks at a range of batch sizes, ignoring the "
                        "--batch-size and --examples options and the choice of network.")
    parser.add_argument('--print-stacktraces',
                        action='store_true',
                        help="Print a stack trace if an exception occurs.")
    all_supported_networks = set()
    for _, networks in SUPPORTED_NETWORKS.items():
        all_supported_networks = all_supported_networks.union(networks)
    parser.add_argument('module', choices=all_supported_networks, metavar='network')
    return parser
