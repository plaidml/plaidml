import argparse
import os
import pathlib
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import flatbuffers

from plaidml.bridge.tensorflow.tests import archive_py_generated as schema
from plaidml.bridge.tensorflow.tests import util


def main(args):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        os.environ[
            'XLA_FLAGS'] = '--xla_dump_to={} --xla_dump_hlo_as_text --xla_dump_hlo_as_proto'.format(
                tmp_dir)
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        with tf.compat.v1.Session() as sess:
            handle = str(args.model.parent)
            layer = hub.Module(handle)
            # Initialize weights in this session
            sess.run(tf.compat.v1.global_variables_initializer())
            # Create placeholders for graphdef generation
            input_shape = [1, 224, 224, 3]
            x_name = 'Placeholder'
            y_name = 'module_apply_default/resnet_v2/use_global_pool'
            x = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
            y = layer(dict(inputs=x, decay_rate=0))
            print(y.op.name)
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess,  # The session in which the weights were initialized
                sess.graph.as_graph_def(),  # The original (non-frozen) graph def
                [y_name]  # The output op name
            )
        with tf.compat.v1.Graph().as_default() as frozen_graph:
            frozen_graph_name = 'freeze'
            tf.compat.v1.import_graph_def(output_graph_def, name=frozen_graph_name)
            with tf.compat.v1.Session(graph=frozen_graph) as sess:
                input_tensor = frozen_graph.get_tensor_by_name(frozen_graph_name + '/' + x_name +
                                                               ':0')
                output_tensor = frozen_graph.get_tensor_by_name(frozen_graph_name + '/' + y_name +
                                                                ':0')
                x_input = np.random.default_rng().random(input_shape, np.float32)
                y_output = sess.run(output_tensor, feed_dict={input_tensor: x_input})
                print(y_output.shape)

        os.system('ls ' + str(tmp_path))
        module_path = tmp_path / 'module_0001.before_optimizations.hlo.pb'
        os.system('cp ' + str(module_path) + ' ' + args.hlo_pb)

    archive = schema.ArchiveT()
    archive.name = 'resnet'
    archive.inputs = [util.convertBuffer('input', x_input)]
    archive.outputs = [util.convertBuffer('output', y_output)]

    builder = flatbuffers.Builder(0)
    packed = archive.Pack(builder)
    builder.Finish(packed)
    args.archive.write(builder.Output())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate archive for resnet')
    parser.add_argument('model', type=pathlib.Path, help='location to read the model')
    parser.add_argument('archive',
                        type=argparse.FileType('wb'),
                        help='location to write the generated archive')
    parser.add_argument('hlo_pb', type=str, help='location to write the frozen hlo pb')
    args = parser.parse_args()
    main(args)
