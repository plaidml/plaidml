#!/usr/bin/env python
#
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
import os

import plaidbench.cli

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
        'resnet50_v2',
        'resnext50',
        'vgg16',
        'vgg19',
        'xception',
    ],
    'onnx': [
        'bvlc_alexnet',
        'densenet121',
        'inception_v1',
        'inception_v2',
        'resnet50',
        'shufflenet',
        'squeezenet',  # TODO: Fix inputs/outputs (only available as *.pb)
        'vgg16',
        'vgg19',
    ],
}


def make_parser():
    # Create the parser outside of main() so the doc system can call this function
    # and thereby generate a web page describing these options. See docs/index.rst.
    parser = argparse.ArgumentParser()
    plaidargs = parser.add_mutually_exclusive_group()
    plaidargs.add_argument('--plaid', action='store_true', help="Use PlaidML as the backend.")
    plaidargs.add_argument('--plaid-edsl',
                           action='store_true',
                           help="EXPERIMENTAL: Use PlaidML2 (EDSL) as the backend")
    plaidargs.add_argument('--caffe2', action='store_true', help="Use Caffe2 as the backend.")
    plaidargs.add_argument('--tf', action='store_true', help="Use TensorFlow as the backend.")
    plaidargs.add_argument(
        '--no-plaid',
        action='store_true',
        help="Use the non-PlaidML backend most appropriate to the chosen frontend")
    frontendargs = parser.add_mutually_exclusive_group()
    frontendargs.add_argument('--keras', action='store_true', help='Use Keras as the frontend')
    frontendargs.add_argument('--onnx', action='store_true', help='Use ONNX as the frontend')
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Use half-precision floats, setting floatx='float16'.")
    parser.add_argument('-v',
                        '--verbose',
                        action='count',
                        default=0,
                        help="Logging verbosity level (0..4).")
    parser.add_argument('--results',
                        default='/tmp/plaidbench_results',
                        help="Destination directory for results output.")
    parser.add_argument('--callgrind',
                        action='store_true',
                        help="Invoke callgrind during timing runs.")
    parser.add_argument('--no-warmup', action='store_true', help="Skip the warmup runs.")
    parser.add_argument('--no-kernel-timing', action='store_true', help="Skip the warmup runs.")
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
    parser.add_argument('--onnx-cpu',
                        action='store_true',
                        help='Use CPU instead of GPU (only used by ONNX)')
    parser.add_argument('--refresh-onnx-data',
                        action='store_true',
                        help='Download ONNX data even if cached')
    parser.add_argument('--tile', default=None, help='Export to this *.tile file')
    parser.add_argument(
        '--fix-learn-phase',
        action='store_true',
        help='Set the Keras learning_phase to an integer (rather than an input tensor)')
    all_supported_networks = set()
    for _, networks in SUPPORTED_NETWORKS.items():
        all_supported_networks = all_supported_networks.union(networks)
    parser.add_argument('module', choices=all_supported_networks, metavar='network')
    return parser


def main():
    exit_status = 0
    parser = make_parser()
    args = parser.parse_args()

    argv = []

    # plaidbench arguments
    if args.verbose:
        argv.append('-{}'.format('v' * args.verbose))
    if args.results:
        argv.append('--results={}'.format(args.results))
    if args.callgrind:
        argv.append('--callgrind')
    if args.examples:
        argv.append('--examples={}'.format(args.examples))
    if args.epochs:
        argv.append('--epochs={}'.format(args.epochs))
    if args.batch_size:
        argv.append('--batch-size={}'.format(args.batch_size))
    if args.blanket_run:
        argv.append('--blanket-run')
    if args.no_warmup:
        argv.append('--no-warmup')
    if args.no_kernel_timing:
        argv.append('--no-kernel-timing')
    if args.print_stacktraces:
        argv.append('--print-stacktraces')

    if args.onnx:
        # onnx arguments
        argv.append('onnx')
        if args.fp16:
            raise NotImplementedError(
                'With ONNX, --fp16 is defined by the model, not by the caller')
        if args.train:
            raise NotImplementedError('With ONNX, training vs. inference is model-specific')
        if args.tile:
            raise NotImplementedError(
                'Can\'t currently save Tile code with PlaidBench ONNX backend.')
        if args.onnx_cpu:
            argv.append('--cpu')
        if args.refresh_onnx_data:
            argv.append('--no-use-cached-data')
        if args.plaid_edsl:
            argv.append('--plaid-edsl')
        elif args.plaid or (not args.no_plaid and not args.caffe2 and not args.tf):
            argv.append('--plaid')
        elif args.caffe2:
            argv.append('--caffe2')
        else:
            argv.append('--tensorflow')
    else:
        # keras arguments
        argv.append('keras')
        if args.tile:
            argv.append('--tile={}'.format(args.tile))
        if args.fp16:
            argv.append('--fp16')
        if args.train:
            argv.append('--train')
        if args.onnx_cpu:
            raise NotImplementedError('--onnx_cpu is only meaningful with --onnx')
        if args.refresh_onnx_data:
            argv.append('--refresh-onnx-data is only meaningful with --onnx')
        if args.fix_learn_phase:
            argv.append('--fix-learn-phase')
        if args.plaid_edsl:
            argv.append('--plaid-edsl')
            os.environ["KERAS_BACKEND"] = "plaidml2.bridge.keras.__init__"
        elif args.plaid or (not args.no_plaid and not args.caffe2 and not args.tf):
            argv.append('--plaid')
        elif args.caffe2:
            raise ValueError('There is no Caffe2 backend for Keras')
        else:
            argv.append('--tensorflow')
            if args.tile:
                raise NotImplementedError('Can\'t save Tile code except in PlaidML')

    # Networks
    if args.module:
        argv.append(args.module)

    # Invoke plaidbench to do the actual benchmarking.
    plaidbench.cli.plaidbench(args=argv)


if __name__ == '__main__':
    main()
