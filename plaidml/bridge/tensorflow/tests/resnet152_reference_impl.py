import argparse
import os
import pathlib
import tempfile
import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def main():
    with tf.compat.v1.Session() as sess:
        layer = hub.Module("https://tfhub.dev/deepmind/local-linearity/imagenet/1")
        # Initialize weights in this session
        sess.run(tf.compat.v1.global_variables_initializer())
        # Create placeholders for graphdef generation
        input_shape = [1, 224, 224, 3]
        x_name = 'Placeholder'
        y_name = 'module_apply_default/resnet_v2/use_global_pool'
        x = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
        y = layer(dict(inputs=x, decay_rate=0))
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,  # The session in which the weights were initialized
            sess.graph.as_graph_def(),  # The original (non-frozen) graph def
            [y_name]  # The output op name
        )
    with tf.compat.v1.Graph().as_default() as frozen_graph:
        frozen_graph_name = 'freeze'
        tf.compat.v1.import_graph_def(output_graph_def, name=frozen_graph_name)
        with tf.compat.v1.Session(graph=frozen_graph) as sess:
            input_tensor = frozen_graph.get_tensor_by_name(frozen_graph_name + '/' + x_name + ':0')
            output_tensor = frozen_graph.get_tensor_by_name(frozen_graph_name + '/' + y_name +
                                                            ':0')
            x_input = np.random.default_rng().random(input_shape, np.float32)
            start_time = time.time()
            y_output = sess.run(output_tensor, feed_dict={input_tensor: x_input})
            end_time = time.time()
            print("Elapsed:", end_time - start_time, "seconds")


if __name__ == "__main__":
    main()
