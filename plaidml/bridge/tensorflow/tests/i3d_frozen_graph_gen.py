import argparse
import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def main(args):
    with tf.compat.v1.Session() as sess:
        handle = str(args.model.parent)
        layer = hub.Module(handle)
        sess.run(tf.compat.v1.global_variables_initializer())
        x = tf.compat.v1.placeholder(tf.float32, shape=[1, 32, 224, 224, 3])
        y = layer(x)
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,  # The session in which the weights were initialized
            sess.graph.as_graph_def(),  # The original (non-frozen) graph def
            [y.op.name]  # The output op name
        )
    with tf.io.gfile.GFile(args.output, "wb+") as f:
        f.write(output_graph_def.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate archive for i3d')
    parser.add_argument('model', type=pathlib.Path, help='location to read the model')
    parser.add_argument('output', type=str, help='location to write the frozen graphdef')
    args = parser.parse_args()
    main(args)
