import argparse
import itertools
import numpy as np
import os
import shutil
import tensorflow as tf
import time

tf.compat.v1.disable_eager_execution()

i_sizes = [[1, 16, 16, 3]]
k_sizes = [[3, 3, 3, 8]]
strides = [(1, 1), (2, 2)]
paddings = [[[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [1, 1], [1, 1], [0, 0]], "VALID", "SAME"]
dilations = [(1, 1), (2, 2)]

desstr = '\nstd::vector<std::string> conv_descriptions = {'
istr = '\nstd::vector<std::vector<float>> conv_is = {'
kstr = '\nstd::vector<std::vector<float>> conv_ks = {'
ostr = '\nstd::vector<std::vector<float>> conv_os = {'
modstr = '\nstd::vector<std::string> conv_modules = {'


def ary2str(A):
    A = A.flatten()
    ret = '{'
    for i in range(len(A)):
        ret += str(A[i]) + ', '
    return ret[:-1] + '}'


#Calculate convoluion for each combination; store inputs, outputs & module
for (i, combination) in enumerate(itertools.product(i_sizes, k_sizes, strides, paddings,
                                                    dilations)):
    nstr = str(i).zfill(4)
    I = tf.compat.v1.placeholder(tf.float32, combination[0])
    K1 = tf.compat.v1.placeholder(tf.float32, combination[1])
    C1 = tf.nn.conv2d(I,
                      K1,
                      strides=combination[2],
                      padding=combination[3],
                      dilations=combination[4])

    with tf.compat.v1.Session() as sess:
        ia = np.random.uniform(size=combination[0])
        k1 = np.random.uniform(size=combination[1])
        result = sess.run(C1, feed_dict={I: ia, K1: k1})

    desstr += '\n\"'
    for ci in range(len(combination)):
        desstr += str(combination[ci]).replace(', ', 'x') + '__'
    desstr += "\","
    istr += '\n' + ary2str(ia) + ','
    kstr += '\n' + ary2str(k1) + ','
    ostr += '\n' + ary2str(result) + ','
    moddir = os.environ['XLA_DUMPDIR']
    modfile = open(moddir + "/module_" + nstr + ".before_optimizations.txt")
    module = modfile.read()
    modfile.close()
    modstr += '\nR\"#(' + module + ')#\",'

#Format & save header file
istr = istr[:-1] + '};'
kstr = kstr[:-1] + '};'
ostr = ostr[:-1] + '};'
modstr = modstr[:-1] + '};'
desstr = desstr[:-1].replace('[', '').replace(']', '') + '};'

fstr = '#include <vector>\n'
fstr += '#include <string>\n'
fstr += desstr + istr + kstr + ostr + modstr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate headers for conv opo')
    parser.add_argument('--output', dest='outfile', help='location to write the generated header')
    args = parser.parse_args()
    print(args.outfile)
    iofile = open(args.outfile, 'w+')
    iofile.write(fstr)
    iofile.close()
