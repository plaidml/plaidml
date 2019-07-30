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

from collections import namedtuple
import hashlib
import importlib
import os
import six
import sys
import tarfile

import click
import numpy as np
import plaidml
from six.moves.urllib.request import urlretrieve

from plaidbench import core


class DataNotFoundError(RuntimeError):
    pass


def download_onnx_data(model, filename, use_cached_data=True):
    # Uses the same directory structure as ONNX's backend test runner to reduce duplication
    expected_sha256 = {
        'bvlc_alexnet': '31202b69aa718b9d4aa67d0ed772efbe6886b69935cdb9224923c9ab8f36d01e',
        'densenet121': '6f3ec833eb27ef2170a407bc081c23893d2a1eb38f6374d36915e3ed1bba8242',
        'inception_v1': '3d934442c85cdeeb1cdceef83bd020dc19e3c8f6b3f5f97d5e7133aee0d41e40',
        'inception_v2': '1dba14b803bad006c591acbf8d5a9d459c139bc2067678a240fee4915d511bcf',
        'resnet50': '09076ac927e4a63730a02c3d40c9e1bb72fd88db942d203f713214a8b69cf09f',
        'shufflenet': 'a8d6339bf29c47d502cb8a11c3c753ec842f5b591f992b3af5590f06a437fd21',
        'squeezenet': '80dc32f8172209d139160258da093dc08af95e61ec4457f98b9061499020331c',
        'vgg16': '52634b4dabb1255dfc0f48a2927dd04e9abf07b43e13d011457d9032d8088080',
        'vgg19': '4ec42e15829d47c47c1f2cf00fd91b116c1e2b47a3e6bd2323c9be72593d69ec',
    }
    onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', os.path.join('~', '.onnx')))
    onnx_models = os.getenv('ONNX_MODELS', os.path.join(onnx_home, 'models'))
    model_dir = os.path.join(onnx_models, model)
    if not os.path.exists(model_dir) or not use_cached_data:
        compressed_file = os.path.join(onnx_models, '{}.tar.gz'.format(model))
        url = 'https://s3.amazonaws.com/download.onnx/models/{}.tar.gz'.format(model)
        if not os.path.exists(compressed_file):
            if not os.path.exists(onnx_models):
                os.makedirs(onnx_models)
            click.echo('Downloading {}...'.format(url), nl=False)
            try:
                urlretrieve(url, compressed_file)
            except:
                click.echo('Failed to download data {} for {}'.format(compressed_file, model))
                raise
            click.echo('Done')
        click.echo('Verifying checksum...', nl=False)
        with open(compressed_file, 'rb') as f:
            buffer_size = 65536
            hash = hashlib.sha256()
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                hash.update(data)
            if hash.hexdigest() != expected_sha256[model]:
                click.echo('Warning: unexpected checksum on file {}: {}'.format(
                    compressed_file, hash.hexdigest()))
        click.echo('Done')
        click.echo('Extracting {}...'.format(compressed_file), nl=False)
        try:
            with tarfile.open(compressed_file) as f:
                f.extractall(onnx_models)
        except:
            click.echo("Failed to extract data {} for {}".format(filename, model))
            raise
        click.echo('Done')
    if not os.path.exists(os.path.join(model_dir, filename)):
        msg = ('Successfully retrieved model data but did not find the file {}. ' +
               'Check the filename or try clearing the cache at {}').format(filename, onnx_models)
        raise DataNotFoundError(msg)
    return os.path.join(model_dir, filename)


class Model(core.Model):

    def __init__(self, frontend, params):
        self.frontend = frontend
        self.params = params
        self.onnx = frontend.onnx

    def setup(self):
        try:
            self.backend = importlib.import_module(self.frontend.backend_info.module_name)
        except ImportError:
            raise core.ExtrasNeeded(self.frontend.backend_info.requirements)
        try:
            data_path = download_onnx_data(self.params.network_name, 'test_data_0.npz',
                                           self.frontend.use_cached_data)
            self.x = np.load(data_path)['inputs'][0]
        except DataNotFoundError:
            # See if we can access it as a proto.
            data_path = download_onnx_data(self.params.network_name,
                                           os.path.join('test_data_set_0', 'input_0.pb'))
            tensor = self.onnx.TensorProto()
            with open(data_path, 'rb') as f:
                tensor.ParseFromString(f.read())
            self.x = self.onnx.numpy_helper.to_array(tensor)
        model_path = download_onnx_data(self.params.network_name, 'model.onnx')
        self.model = self.onnx.load(model_path)

    def compile(self):
        device = 'CPU' if self.frontend.cpu else self.frontend.backend_info.gpu_device
        kwargs = {}
        if device:
            kwargs['device'] = device
        self.rep = self.backend.prepare(self.model, **kwargs)

    def run(self, once=False, warmup=False):
        if once:
            examples = self.params.batch_size
        elif warmup:
            examples = self.params.warmups
        else:
            examples = self.params.examples
        for _ in range(examples // self.params.batch_size):
            partial_result = self.rep.run([self.x[:self.params.batch_size]])
        return (partial_result, {})

    def golden_output(self):
        try:
            full_path = download_onnx_data(self.params.network_name, 'test_data_0.npz')
            return (np.load(full_path)['outputs'][0], core.Precision.INFERENCE)
        except DataNotFoundError:
            # See if we can access it as a proto.
            data_path = download_onnx_data(self.params.network_name,
                                           os.path.join('test_data_set_0', 'output_0.pb'))
            tensor = self.onnx.TensorProto()
            with open(data_path, 'rb') as f:
                tensor.ParseFromString(f.read())
            return (self.onnx.numpy_helper.to_array(tensor), core.Precision.INFERENCE)


class Frontend(core.Frontend):
    NETWORK_NAMES = [
        'bvlc_alexnet',
        'densenet121',
        'inception_v1',
        'inception_v2',
        'resnet50',
        'shufflenet',
        'squeezenet',
        'vgg16',
        'vgg19',
    ]

    def __init__(self, backend, cpu, use_cached_data):
        super(Frontend, self).__init__(Frontend.NETWORK_NAMES)
        self.cpu = cpu
        self.use_cached_data = use_cached_data
        self.backend_info = backend

        try:
            importlib.import_module(backend.module_name)
        except ImportError:
            six.raise_from(core.ExtrasNeeded(backend.requirements), None)
        if backend.is_plaidml:
            self.configuration['plaid'] = plaidml.__version__

        self.onnx = importlib.import_module('onnx')
        importlib.import_module('onnx.numpy_helper')

    def model(self, params):
        return Model(self, params)

    def name(self):
        return 'onnx'


BackendInfo = namedtuple('BackendInfo',
                         ['name', 'module_name', 'gpu_device', 'is_plaidml', 'requirements'])


@click.command(cls=core.FrontendCommand, networks=Frontend.NETWORK_NAMES)
@click.option('--plaid',
              'backend',
              flag_value=BackendInfo('plaid', 'onnx_plaidml.backend', None, True,
                                     ['onnx-plaidml']),
              default=True,
              help='Use PlaidML as the backend')
@click.option('--caffe2',
              'backend',
              flag_value=BackendInfo('caffe2', 'caffe2.python.onnx.backend', 'CUDA', False,
                                     ['caffe2']),
              help='Use Caffe2 as the backend')
@click.option('--tensorflow',
              'backend',
              flag_value=BackendInfo('tensorflow', 'onnx_tf.backend', 'CUDA', False, ['onnx-tf']),
              help='Use TensorFlow as the backend')
@click.option('--cpu/--no-cpu', default=False, help='Use CPU instead of GPU')
@click.option('--use-cached-data/--no-use-cached-data',
              default=True,
              help='Use cached models when possible')
@click.argument('networks', nargs=-1, type=click.Choice(Frontend.NETWORK_NAMES))
@click.pass_context
def cli(ctx, backend, cpu, use_cached_data, networks):
    """Benchmarks ONNX models."""
    runner = ctx.ensure_object(core.Runner)

    frontend = Frontend(backend, cpu, use_cached_data)
    return runner.run(frontend, backend, networks)
