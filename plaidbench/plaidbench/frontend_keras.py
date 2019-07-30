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

import importlib
import os
import tempfile

import click
import numpy as np
import plaidml
from plaidbench import core


def setup_cifar(train, epoch_size):
    # Setup
    if train:
        # Training setup
        from keras.datasets import cifar10
        from keras.utils.np_utils import to_categorical
        click.echo('Loading CIFAR data')
        (x_train, y_train_cats), (_, _) = cifar10.load_data()
        x_train = x_train[:epoch_size]
        y_train_cats = y_train_cats[:epoch_size]
        y_train = to_categorical(y_train_cats, num_classes=1000)
    else:
        # Inference setup
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cifar_path = os.path.join(this_dir, 'cifar16.npy')
        x_train = np.load(cifar_path).repeat(1 + epoch_size // 16, axis=0)[:epoch_size]
        y_train = None
    return x_train, y_train


imdb_max_features = 20000
imdb_max_length = 80


def setup_imdb(train, epoch_size):
    # Setup
    if train:
        # Training setup
        from keras.datasets import imdb
        from keras.preprocessing import sequence
        click.echo('Loading IMDB data')
        (x_train, y_train), (_, _) = imdb.load_data(num_words=imdb_max_features)
        x_train = sequence.pad_sequences(x_train, maxlen=imdb_max_length)
        x_train = x_train[:epoch_size]
        y_train = y_train[:epoch_size]
    else:
        # Inference setup
        this_dir = os.path.dirname(os.path.abspath(__file__))
        imdb_path = os.path.join(this_dir, 'imdb16.npy')
        x_train = np.load(imdb_path).repeat(1 + epoch_size // 16, axis=0)[:epoch_size]
        y_train = None
    return x_train, y_train


class Model(core.Model):

    def __init__(self, frontend, params):
        learn_phase = params.learn_phase  # e.g. 0 for infer with fixed learning phase (but should come from param)
        if learn_phase is not None:
            from keras.backend import set_learning_phase
            set_learning_phase(learn_phase)
        self.frontend = frontend
        self.params = params

    def fold_batch_norm(self, model):
        import json
        import plaidml.keras.backend as K
        from keras.models import model_from_json

        def make_mults(weights):
            epsilon = 1e-3
            gamma = 1
            beta = 0
            mean = None
            var = None
            for w in weights:
                name = w.name.split('/')[1]
                if name == 'beta':
                    beta = K.get_value(w)
                elif name == 'gamma':
                    gamma = K.get_value(w)
                elif name == 'moving_mean':
                    mean = K.get_value(w)
                elif name == 'moving_variance':
                    var = K.get_value(w)
                else:
                    raise Exception("Unknown weight in batch norm")
            m = gamma / np.sqrt(var + epsilon)
            a = beta - mean * m
            return (m, a)

        allowed_layers = ['Conv2D', 'DepthwiseConv2D', 'SeparableConv2D']
        js = json.loads(model.to_json())
        old_layers = js['config']['layers']
        new_layers = []
        by_name = {}
        passthru = {}
        saved_kernels = {}
        saved_pointwise = {}
        saved_biases = {}

        # Compute new model
        for i in range(len(old_layers)):
            cur = old_layers[i]
            if cur['class_name'] != 'BatchNormalization':
                by_name[cur['name']] = cur
                if len(cur['inbound_nodes']):
                    new_inbound = []
                    for i in cur['inbound_nodes'][0]:
                        if i[0] in passthru:
                            new_inbound.append([passthru[i[0]], 0, 0, {}])
                        else:
                            new_inbound.append(i)
                    cur['inbound_nodes'] = [new_inbound]
                new_layers.append(cur)
                continue
            cur_name = cur['name']
            prev_name = cur['inbound_nodes'][0][0][0]
            prev = model.get_layer(prev_name)
            prev_js = by_name[prev_name]
            if prev_js['class_name'] == 'SeparableConv2D':
                # TODO: Fix the since it slows down xception
                new_layers.append(cur)
                continue
            passthru[cur_name] = prev_name
            if prev_js['class_name'] not in allowed_layers:
                raise Exception("Can only fold BN to Conv2d right now, got: " +
                                prev_js['class_name'] + " " + str(i))
            (m, a) = make_mults(model.layers[i].weights)
            bias_offset = 1
            if prev_js['class_name'] == 'DepthwiseConv2D':
                saved_kernels[prev_name] = np.reshape(m, (m.shape[0], 1)) * K.get_value(
                    prev.weights[0])
            elif prev_js['class_name'] == 'SeparableConv2D':
                saved_kernels[prev_name] = m * K.get_value(prev.weights[1])
                bias_offset = 2
            else:
                saved_kernels[prev_name] = m * K.get_value(prev.weights[0])
            if not prev_js['config']['use_bias']:
                saved_biases[prev_name] = a
                prev_js['config']['use_bias'] = True
            else:
                saved_biases[prev_name] = m * K.get_value(prev.weights[bias_offset]) + a

        # Make new model and copy weights across
        js['config']['layers'] = new_layers
        folded_model = model_from_json(json.dumps(js))
        for i in range(len(new_layers)):
            cur = new_layers[i]
            old_layer = model.get_layer(cur['name'])
            new_layer = folded_model.get_layer(cur['name'])
            if len(old_layer.get_weights()) == len(new_layer.get_weights()):
                if old_layer.input_shape == new_layer.input_shape and old_layer.output_shape == new_layer.output_shape:
                    new_layer.set_weights(old_layer.get_weights())

        # Do actual weight updates
        for layer in folded_model.layers:
            name = layer.get_config()['name']
            if name in saved_kernels:
                if len(layer.weights) == 2:
                    K.set_value(layer.weights[0], saved_kernels[name])
                    K.set_value(layer.weights[1], saved_biases[name])
                else:
                    K.set_value(layer.weights[1], saved_kernels[name])
                    K.set_value(layer.weights[2], saved_biases[name])

        return folded_model

    def setup(self):
        if self.params.network_name == 'imdb_lstm':
            x, y_train = setup_imdb(self.frontend.train, self.params.epoch_size)
        else:
            x, y_train = setup_cifar(self.frontend.train, self.params.epoch_size)
        self.x = x
        self.y_train = y_train

        this_dir = os.path.dirname(os.path.abspath(__file__))
        build_model_kwargs = dict()
        filename = '{}.py'.format(self.params.network_name)
        module_path = os.path.join(this_dir, 'networks', 'keras', filename)
        mod = {'__file__': __file__, '_backend_name': self.params.backend_name}
        with open(module_path) as f:
            code = compile(f.read(), module_path, 'exec')
            eval(code, mod)
        self.x = mod['scale_dataset'](self.x)
        self.model = mod['build_model'](**build_model_kwargs)
        if not self.frontend.train and os.getenv('USE_STRIPE', '0') == '1':
            click.echo('Model loaded, folding in batch_norm')
            self.model = self.fold_batch_norm(self.model)

    def compile(self):
        if self.params.network_name[:3] == 'vgg':
            from keras.optimizers import SGD
            optimizer = SGD(lr=0.0001)
        else:
            optimizer = 'sgd'

        if self.params.network_name == 'imdb_lstm':
            self.model.compile(optimizer=optimizer,
                               loss='binary_crossentropy',
                               metrics=['accuracy'])
        else:
            self.model.compile(optimizer=optimizer,
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

        self.model._make_predict_function()
        from keras.backend import backend as backend_name
        if backend_name() == 'plaidml.keras.backend':
            self.model.predict_function._invoker.set_const()

    def keras_golden_output(self, typename):
        filename = '{},bs-{}.npy'.format(typename, self.params.batch_size)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        golden_path = os.path.join(this_dir, 'golden', self.params.network_name, filename)
        if not os.path.exists(golden_path):
            raise core.GoldenOutputNotAvailableError()
        return np.load(golden_path)


class InferenceModel(Model):

    def run(self, once=False, warmup=False):
        if once:
            epoch_size = self.params.batch_size
        elif warmup:
            epoch_size = self.params.warmups
        else:
            epoch_size = self.params.epoch_size
        out = self.model.predict(x=self.x[:epoch_size], batch_size=self.params.batch_size)
        # Hack to ensure eventlog gets written
        if not once and not warmup:
            import keras.backend as b
            if b.backend() == 'plaidml':
                b._ctx.shutdown()
        return (out, {})

    def golden_output(self):
        return (self.keras_golden_output('infer'), core.Precision.INFERENCE)


class TrainingModel(Model):

    def validate(self):
        if self.params.examples % self.params.epochs != 0:
            raise ValueError('The number of examples must be divisible by the number of epochs.')
        if self.params.examples < self.params.epochs:
            raise ValueError(
                'The number of examples must be greater than or equal to the number of epochs (examples-per-epoch must be >= 1).'
            )
        if (self.params.examples // self.params.epochs) < self.params.batch_size:
            raise ValueError(
                'The number of examples per epoch must be greater than or equal to the batch size.'
            )
        if (self.params.examples // self.params.epochs) % self.params.batch_size != 0:
            raise ValueError(
                'The number of examples per epoch is not divisible by the batch size.')

    def run(self, once=False, warmup=False):
        if once:
            epoch_size = self.params.batch_size
            epochs = 1
        elif warmup:
            epoch_size = self.params.warmups
            epochs = 1
        else:
            epoch_size = self.params.epoch_size
            epochs = self.params.epochs
        history = self.model.fit(x=self.x[:epoch_size],
                                 y=self.y_train[:epoch_size],
                                 batch_size=self.params.batch_size,
                                 epochs=epochs,
                                 shuffle=False,
                                 initial_epoch=0)
        return (np.array(history.history['loss']), {})

    def golden_output(self):
        return (self.keras_golden_output('train'), core.Precision.TRAINING)


class Frontend(core.Frontend):
    NETWORK_NAMES = [
        'densenet121',
        'densenet169',
        'densenet201',
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
        'imdb_lstm',
    ]

    def __init__(self, backend, fp16, train):
        super(Frontend, self).__init__(Frontend.NETWORK_NAMES)
        self.backend = backend
        if backend == 'plaid':
            try:
                self.configuration['plaid'] = importlib.import_module('plaidml').__version__
                importlib.import_module('plaidml.keras').install_backend()
            except ImportError:
                raise core.ExtrasNeeded(['plaidml-keras'])
        elif backend == 'tensorflow':
            try:
                importlib.import_module('keras.backend')
            except ImportError:
                raise core.ExtrasNeeded(['keras', 'tensorflow'])
        if fp16:
            importlib.import_module('keras.backend').set_floatx('float16')
        if train:
            self.configuration['train'] = True
        self.fp16 = fp16
        self.train = train

    def model(self, params):
        if self.train:
            return TrainingModel(self, params)
        return InferenceModel(self, params)

    @property
    def name(self):
        return "keras"

    @property
    def init_args(self):
        return (self.backend, self.fp16, self.train)

    @property
    def blanket_batch_sizes(self):
        return [1, 4, 8, 16]


@click.command(cls=core.FrontendCommand, networks=Frontend.NETWORK_NAMES)
@click.option('--plaid',
              'backend',
              flag_value='plaid',
              default=True,
              help='Use PlaidML as the backend')
@click.option('--tensorflow',
              'backend',
              flag_value='tensorflow',
              help='Use TensorFlow as the backend')
@click.option('--fp16/--no-fp16',
              default=False,
              help='Use half-precision floats, settings floatx=\'float16\'')
@click.option('--train/--no-train',
              default=False,
              help='Measure training performance instead of inference')
@click.argument('networks', nargs=-1, type=click.Choice(Frontend.NETWORK_NAMES))
@click.option(
    '--tile',
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    default=None,
    help='Save network to *.tile file',
)
@click.option('--fix-learn-phase/--no-fix-learn-phase',
              default=False,
              help='Set the Keras learning_phase to an integer (rather than an input tensor)')
@click.pass_context
def cli(ctx, backend, fp16, train, networks, tile, fix_learn_phase):
    """Benchmarks Keras neural networks."""
    runner = ctx.ensure_object(core.Runner)
    if fix_learn_phase:
        if train:
            learn_phase = 1
        else:
            learn_phase = 0
    else:
        learn_phase = None
    runner.param_builder = core.ExplicitParamBuilder(
        runner.param_builder.params.batch_size,
        runner.param_builder.params.epochs,
        runner.param_builder.params.examples,
        learn_phase=learn_phase,
    )
    runner.tile = tile
    frontend = Frontend(backend, fp16, train)
    return runner.run(frontend, backend, networks)
