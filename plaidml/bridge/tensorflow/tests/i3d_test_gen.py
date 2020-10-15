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
    with tf.compat.v1.Session() as sess:
        handle = str(args.model.parent)
        layer = hub.Module(handle)
        # Initialize weights in this session
        sess.run(tf.compat.v1.global_variables_initializer())
        # Create placeholders for graphdef generation
        input_shape = [1, 32, 224, 224, 3]
        x = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
        y = layer(x)
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,  # The session in which the weights were initialized
            sess.graph.as_graph_def(),  # The original (non-frozen) graph def
            [y.op.name]  # The output op name
        )
        # Run an inference with random inputs to generate a correctness baseline for the archive
        x_input = np.random.uniform(size=input_shape)
        y_output = sess.run(y, feed_dict={x: x_input})
    # Write frozen graph def as binary
    with tf.io.gfile.GFile(args.graphdef, "wb+") as f:
        f.write(output_graph_def.SerializeToString())

    archive = schema.ArchiveT()
    archive.name = 'i3d'
    archive.inputs = [util.convertBuffer('input', x_input)]
    archive.outputs = [util.convertBuffer('output', y_output)]

    builder = flatbuffers.Builder(0)
    packed = archive.Pack(builder)
    builder.Finish(packed)
    args.archive.write(builder.Output())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate archive for i3d')
    parser.add_argument('model', type=pathlib.Path, help='location to read the model')
    parser.add_argument('archive',
                        type=argparse.FileType('wb'),
                        help='location to write the generated archive')
    parser.add_argument('graphdef', type=str, help='location to write the frozen graphdef')
    args = parser.parse_args()
    main(args)
