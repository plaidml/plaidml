import argparse
import glob
import os
import pathlib
import tarfile
import tempfile
import time

import numpy as np

import tensorflow as tf
import flatbuffers
from plaidml.bridge.tensorflow.tests import archive_py_generated as schema
from plaidml.bridge.tensorflow.tests import util


def main(args):
    saved_model = tarfile.open(args.saved_model_path)
    saved_model.extractall()

    input_shape = [1, 224, 224, 3]
    model_name = "resnext50_tf_saved_model"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        os.environ['XLA_FLAGS'] = '--xla_dump_to={} --xla_dump_hlo_as_proto'.format(tmp_dir)
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        with tf.compat.v1.Session() as sess:
            model = tf.compat.v1.saved_model.load(sess, ["train"], model_name)
            x_name = model.signature_def['serving_default'].inputs['input'].name
            y_name = 'pool1:0'
            x = tf.compat.v1.get_default_graph().get_tensor_by_name(x_name)
            y = tf.compat.v1.get_default_graph().get_tensor_by_name(y_name)
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess,  # The session in which the weights were initialized
                sess.graph.as_graph_def(),  # The original (non-frozen) graph def
                [y.op.name]  # The output op name
            )
        with tf.compat.v1.Graph().as_default() as frozen_graph:
            frozen_graph_name = 'freeze'
            tf.compat.v1.import_graph_def(output_graph_def, name=frozen_graph_name)
            with tf.compat.v1.Session(graph=frozen_graph) as sess:
                input_tensor = frozen_graph.get_tensor_by_name(frozen_graph_name + '/' + x_name)
                output_tensor = frozen_graph.get_tensor_by_name(frozen_graph_name + '/' + y_name)
                x_input = np.random.default_rng().random(input_shape, np.float32)
                y_output = sess.run(output_tensor, feed_dict={input_tensor: x_input})

        module_path = tmp_path / 'module_0000.before_optimizations.hlo.pb'
        os.system('cp ' + str(module_path) + ' ' + args.hlo_pb)

    archive = schema.ArchiveT()
    archive.name = 'resnext'
    archive.inputs = [util.convertBuffer('input', x_input)]
    archive.outputs = [util.convertBuffer('output', y_output)]

    builder = flatbuffers.Builder(0)
    packed = archive.Pack(builder)
    builder.Finish(packed)
    args.archive.write(builder.Output())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate archive for resnext')
    parser.add_argument('saved_model_path', help='location where the saved_model is located')
    parser.add_argument('archive',
                        type=argparse.FileType('wb'),
                        help='location to write the generated archive')
    parser.add_argument('hlo_pb', type=str, help='location to write the frozen hlo pb')
    args = parser.parse_args()
    main(args)
