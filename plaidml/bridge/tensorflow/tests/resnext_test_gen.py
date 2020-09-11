import argparse
import glob
import os
import pathlib
import tarfile
import tempfile

import numpy as np

import tensorflow as tf
from flatbuffers.python import flatbuffers
from plaidml.bridge.tensorflow.tests import archive_py_generated as schema
from plaidml.bridge.tensorflow.tests import util


def main(args):
    saved_model = tarfile.open(args.saved_model_path)
    saved_model.extractall()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        os.environ['XLA_FLAGS'] = '--xla_dump_to={}'.format(tmp_dir)
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

        # add resnext model here
        x = np.ones((1, 224, 224, 3))
        weights = {}
        model_name = "resnext50_tf_saved_model"
        with tf.compat.v1.Session() as sess:
            model = tf.saved_model.load(sess, ["train"], model_name)
            input_name = model.signature_def['serving_default'].inputs['input'].name
            input_tensor = tf.get_default_graph().get_tensor_by_name(input_name)
            output_tensor = tf.get_default_graph().get_tensor_by_name('stage4_unit3_relu:0')
            weights_ref = tf.all_variables()
            for index in range(len(weights_ref)):
                weights[weights_ref[index].name.replace("/", "_").replace(":0", "")] = sess.run(
                    weights_ref[index].value())
            y = sess.run(output_tensor, feed_dict={input_tensor: x})

        module_path = tmp_path / 'module_0000.before_optimizations.txt'
        module_text = module_path.read_text()

    archive = schema.ArchiveT()
    archive.name = 'resnext'
    archive.model = module_text
    archive.weights = [util.convertBuffer(key, value) for key, value in weights.items()]
    archive.inputs = [util.convertBuffer('input', x)]
    archive.outputs = [util.convertBuffer('output', y)]

    builder = flatbuffers.Builder(0)
    packed = archive.Pack(builder)
    builder.Finish(packed)
    args.output.write(builder.Output())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate archive for resnext')
    parser.add_argument('output',
                        type=argparse.FileType('wb'),
                        help='location to write the generated archive')
    parser.add_argument('saved_model_path', help='location where the saved_model is located')
    args = parser.parse_args()
    main(args)
