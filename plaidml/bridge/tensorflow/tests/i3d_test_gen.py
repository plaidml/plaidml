import argparse
import glob
import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub


def write_weights(of):
    tmp_weights_path = "/tmp/weights"

    glob_header = "namespace weights {\n"
    glob_footer = "} // namespace weights"

    hub_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
    i3d = hub.KerasLayer(hub_url)

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
        for f in i3d_weights_files:
            with open(f) as infile:
                for line in infile:
                    outfile.write(line.replace("{,", "{").replace(";,", ";"))
        outfile.write(glob_footer)

    # remove temporary weights
    shutil.rmtree(tmp_weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate headers for i3d')
    parser.add_argument('--output', dest='outfile', help='location to write the generated header')
    args = parser.parse_args()
    write_weights(args.outfile)
