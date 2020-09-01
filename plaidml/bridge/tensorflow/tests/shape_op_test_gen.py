import argparse
import itertools
import numpy as np
import os
import shutil
import tensorflow as tf
import time

tf.compat.v1.disable_eager_execution()

opsetname = 'shape'

ops = ['reshape', 'pad', 'slice', 'transpose', 'broadcast']


def getInputs(opname):
    if opname == 'broadcast':
        # Input shape, output shape
        tests = [[[1, 3], [3, 3]], [[5, 1, 2], [5, 3, 2]]]

        def opfunc(test):
            I = tf.compat.v1.placeholder(tf.float32, test[0])
            O = tf.broadcast_to(I, test[1])
            with tf.compat.v1.Session() as sess:
                i = np.random.uniform(size=test[0])
                o = sess.run(O, feed_dict={I: i})
            return [i], [o]

        return tests, opfunc
    elif opname == 'reshape':
        # Product of each isize must be divisible by product of each odim
        i_sizes = [[12, 12], [12, 12, 12], [24]]
        o_dims = [[1], [2], [6], [3, 4], [2, 3, 1], [12], [3, -1, 2], [-1, 12]]
        tests = itertools.product(i_sizes, o_dims)

        def opfunc(test):
            if np.any(np.less(test[1], 0)):
                osize = test[1]
            else:
                fdim = np.product(test[0]) // np.product(test[1])
                osize = test[1] + [fdim]
            if len(osize) == len(test[0]):
                osize = osize + [1]  # Ensure the reshape is not a no-op
            I = tf.compat.v1.placeholder(tf.float32, test[0])
            O = tf.reshape(I, osize)
            with tf.compat.v1.Session() as sess:
                i = np.random.uniform(size=test[0])
                o = sess.run(O, feed_dict={I: i})
            return [i], [o]

        return tests, opfunc
    elif opname == 'pad':
        # Product of each isize must be divisible by product of each odim
        i_sizes = [[2, 2], [3, 4, 1, 6]]
        max_pads = [3, 2, 1]
        modes = ['CONSTANT']  # REFLECT, SYMMETRIC
        cvals = [-1, 5, 0]
        tests = itertools.product(i_sizes, max_pads, modes, cvals)

        def opfunc(test):
            I = tf.compat.v1.placeholder(tf.float32, test[0])

            padding = np.random.randint(test[1], size=(len(test[0]), 2))
            if np.all(padding == 0):
                padding[0][0] = 1
            if test[2] == 'CONSTANT':
                O = tf.pad(I, padding, "CONSTANT", None, test[3])
            elif test[2] == 'REFLECT':
                padding = np.minimum(padding, np.expand_dims(np.array(test[0]), 1) - 1)
                O = tf.pad(I, padding, "REFLECT", None, test[3])
            elif test[2] == 'SYMMETRIC':
                padding = np.minimum(padding, np.expand_dims(np.array(test[0]), 1))
                O = tf.pad(I, padding, "SYMMETRIC", None, test[3])

            with tf.compat.v1.Session() as sess:
                i = np.random.uniform(size=test[0])
                o = sess.run(O, feed_dict={I: i})
            if test[2] == 'CONSTANT':
                return [np.array(test[3]), i], [o]
            return [i], [o]

        return tests, opfunc
    elif opname == 'slice':
        i_sizes = [[2, 2, 1]]
        begins = [[0, 0, 0], [1, 2, 3], [5, 4, 2]]
        s_sizes = [[1, 1, 1], [2, 1, 4]]

        tests = itertools.product(i_sizes, begins, s_sizes)

        def opfunc(test):
            i_size, begin, s_size = np.array(test[0]), np.array(test[1]), np.array(test[2])
            s_size[s_size > i_size] = i_size[s_size > i_size]
            begin = np.minimum(begin, i_size - s_size)

            I = tf.compat.v1.placeholder(tf.float32, test[0])
            O = tf.slice(I, begin, s_size)
            with tf.compat.v1.Session() as sess:
                i = np.random.uniform(size=test[0])
                o = sess.run(O, feed_dict={I: i})
            return [i], [o]

        return tests, opfunc
    elif opname == 'transpose':
        # Input shape, perm
        tests = [[[3, 1, 2], [1, 0, 2]], [[3, 1, 3], []]]

        def opfunc(test):
            I = tf.compat.v1.placeholder(tf.float32, test[0])
            if len(test[1]) > 0:
                O = tf.transpose(I, test[1])
            else:
                O = tf.transpose(I)
            with tf.compat.v1.Session() as sess:
                i = np.random.uniform(size=test[0])
                o = sess.run(O, feed_dict={I: i})
            return [i], [o]

        return tests, opfunc
    pass


def ary2str(A):
    A = A.flatten()
    ret = '{'
    for i in range(len(A)):
        ret += str(A[i]) + ', '
    return ret[:-2] + '}'


fstr = '#include <vector>\n'
fstr += '#include <string>\n'
n = 0

for opname in ops:
    tests, opfunc = getInputs(opname)

    desstr = '\nstd::vector<std::string> ' + opname + '_descriptions = {'
    istr = '\nstd::vector<std::vector<std::vector<float>>> ' + opname + '_is = {'
    ostr = '\nstd::vector<std::vector<std::vector<float>>> ' + opname + '_os = {'
    modstr = '\nstd::vector<std::string> ' + opname + '_modules = {'

    for test in tests:
        nstr = str(n).zfill(4)

        inputs, outputs = opfunc(test)

        desstr += '\n\"'
        for ti in test:
            desstr += str(ti).replace(', ', 'x') + '__'
        desstr = desstr[:-2] + "\","
        istr += '\n{'
        for inp in inputs:
            istr += ary2str(inp) + ','
        istr = istr[:-1] + '},'
        ostr += '\n{'
        for outp in outputs:
            ostr += ary2str(outp) + ','
        ostr = ostr[:-1] + '},'
        moddir = os.environ['XLA_DUMPDIR']
        modfile = open(moddir + "/module_" + nstr + ".before_optimizations.txt")
        module = modfile.read()
        modfile.close()
        modstr += '\nR\"#(' + module + ')#\",'
        n += 1

    #Format & save header file
    istr = istr[:-1] + '};\n'
    ostr = ostr[:-1] + '};\n'
    modstr = modstr[:-1] + '};\n'
    desstr = desstr[:-1].replace('[', '').replace(']', '') + '};\n'

    fstr += desstr + istr + ostr + modstr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate headers for shape op')
    parser.add_argument('--output', dest='outfile', help='location to write the generated header')
    args = parser.parse_args()
    print(args.outfile)
    iofile = open(args.outfile, 'w+')
    iofile.write(fstr)
    iofile.close()
