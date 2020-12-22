import tarfile
import time

import numpy as np
import tensorflow as tf


def main():
    saved_model = tarfile.open("../models_resnext50/file/downloaded")
    saved_model.extractall()

    input_shape = [1, 224, 224, 3]
    model_name = "resnext50_tf_saved_model"

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
            start_time = time.time()
            y_output = sess.run(output_tensor, feed_dict={input_tensor: x_input})
            end_time = time.time()
            print("Elapsed:", end_time - start_time, "seconds")


if __name__ == "__main__":
    main()
