import argparse
import glob
import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub


def write_weights(of):
    tf.compat.v1.enable_eager_execution()

    tmp_weights_path = "/tmp/weights"

    glob_header = "#include <vector>\n\n"
    weights_header = "namespace weights {\n"
    weights_footer = "} // namespace weights\n"
    inputs_header = "namespace inputs {\n"
    inputs_footer = "} // namespace inputs\n"
    outputs_header = "namespace outputs {\n"
    outputs_footer = "} // namespace outputs\n"

    hub_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
    i3d = hub.KerasLayer(hub_url, trainable=False)

    # start with an empty directory
    if os.path.exists(tmp_weights_path):
        shutil.rmtree(tmp_weights_path)

    os.mkdir(tmp_weights_path)

    # write each weight to a file
    for weight in i3d.weights:
        name = weight.name.split(":")[0].replace("/", "_")
        np.savetxt(tmp_weights_path + "/" + name + ".h",
                   np.ndarray.flatten(weight.numpy()),
                   newline=",\n",
                   header="std::vector<float> " + name + " = {",
                   footer="};",
                   comments="")

    # concat all files together
    i3d_weights_files = glob.glob(tmp_weights_path + "/*")
    with open(of, 'w+') as outfile:
        outfile.write(glob_header)
        outfile.write(weights_header)
        for f in i3d_weights_files:
            with open(f) as infile:
                f_str = infile.read()
                f_str = f_str.replace("{,", "{").replace(";,", ";").replace(",\n}", "\n}")
                outfile.write(f_str)
        outfile.write(weights_footer)

    # remove temporary weights
    shutil.rmtree(tmp_weights_path)
    os.mkdir(tmp_weights_path)

    #tf.compat.v1.disable_eager_execution()

    # create input, output pairs
    model = tf.keras.Sequential(layers=i3d)
    x = np.random.uniform(size=(1, 32, 224, 224, 3))
    y = model.predict(x)
    x_fname = tmp_weights_path + "/input.h"
    y_fname = tmp_weights_path + "/output.h"
    np.savetxt(x_fname,
               np.ndarray.flatten(x),
               newline=",\n",
               header="std::vector<float> input = {",
               footer="};",
               comments="")
    np.savetxt(y_fname,
               np.ndarray.flatten(y),
               newline=",\n",
               header="std::vector<std::vector<float>> output = {{",
               footer="}};",
               comments="")
    with open(of, 'a+') as outfile:
        outfile.write(inputs_header)
        with open(x_fname) as infile:
            f_str = infile.read()
            f_str = f_str.replace("{,", "{").replace(";,", ";").replace(",\n}", "\n}")
            outfile.write(f_str)
        outfile.write(inputs_footer)
        outfile.write(outputs_header)
        with open(y_fname) as infile:
            f_str = infile.read()
            f_str = f_str.replace("{,", "{").replace(";,", ";").replace(",\n}", "\n}")
            outfile.write(f_str)
        outfile.write(outputs_footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate headers for i3d')
    parser.add_argument('--output', dest='outfile', help='location to write the generated header')
    args = parser.parse_args()
    write_weights(args.outfile)
