import argparse
import os
import pathlib
import tempfile

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from flatbuffers.python import flatbuffers
from plaidml.bridge.tensorflow.tests import archive_py_generated as schema

dataTypeMap = {
    'int8': (schema.I8DataT, schema.Data.I8Data),
    'int16': (schema.I16DataT, schema.Data.I16Data),
    'int32': (schema.I32DataT, schema.Data.I32Data),
    'int64': (schema.I64DataT, schema.Data.I64Data),
    'uint8': (schema.U8DataT, schema.Data.U8Data),
    'uint16': (schema.U16DataT, schema.Data.U16Data),
    'uint32': (schema.U32DataT, schema.Data.U32Data),
    'uint64': (schema.U64DataT, schema.Data.U64Data),
    'float16': (schema.F16DataT, schema.Data.F16Data),
    'float32': (schema.F32DataT, schema.Data.F32Data),
    'float64': (schema.F64DataT, schema.Data.F64Data),
}


def convertBuffer(name, src):
    dtype = dataTypeMap.get(str(src.dtype))
    if not dtype:
        raise Exception('Unknown dtype: {}'.format(src.dtype))
    data = dtype[0]()
    data.data = np.ndarray.flatten(src)
    buffer = schema.BufferT()
    buffer.name = name
    buffer.dataType = dtype[1]
    buffer.data = data
    return buffer


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
    archive.weights = [convertBuffer(x.name, x.numpy()) for x in layer.weights]
    archive.inputs = [convertBuffer('input', x)]
    archive.outputs = [convertBuffer('output', y.numpy())]

    builder = flatbuffers.Builder(0)
    packed = archive.Pack(builder)
    print(packed)
    builder.Finish(packed)
    args.output.write(builder.Output())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate archive for i3d')
    parser.add_argument('output',
                        type=argparse.FileType('wb'),
                        help='location to write the generated archive')
    args = parser.parse_args()
    main(args)
