import argparse
import os
import pathlib
import tempfile

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from flatbuffers.python import flatbuffers
from plaidml.bridge.tensorflow.tests import archive_py_generated as schema
from plaidml.bridge.tensorflow.tests import util


def main(args):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        os.environ['XLA_FLAGS'] = '--xla_dump_to={}'.format(tmp_dir)
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        tf.compat.v1.enable_eager_execution()

        hub_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
        layer = hub.KerasLayer(hub_url, trainable=False)

        x = np.random.uniform(size=(1, 32, 224, 224, 3)).astype('float32')
        y = layer(x)
        module_path = tmp_path / 'module_0001.before_optimizations.txt'
        module_text = module_path.read_text()

    archive = schema.ArchiveT()
    archive.name = 'i3d'
    archive.model = module_text
    archive.weights = [util.convertBuffer(x.name, x.numpy()) for x in layer.weights]
    archive.inputs = [util.convertBuffer('input', x)]
    archive.outputs = [util.convertBuffer('output', y.numpy())]

    builder = flatbuffers.Builder(0)
    packed = archive.Pack(builder)
    builder.Finish(packed)
    args.output.write(builder.Output())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate archive for i3d')
    parser.add_argument('output',
                        type=argparse.FileType('wb'),
                        help='location to write the generated archive')
    args = parser.parse_args()
    main(args)
