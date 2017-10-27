# Copyright Vertex.AI.

import functools
import numpy as np
import numpy.testing as npt
import operator
import os
import unittest
import sys
import plaidml
import testing.plaidml_config
#import plaidml  #only needed if adjusting vlog

from keras.backend.common import set_floatx, floatx
set_floatx('float32')

from keras.backend import theano_backend as th
from plaidml.keras import backend as pkb
from keras.backend import tensorflow_backend as tf

# Tensorflow needs some code called directly
import tensorflow

# Theano breaks on convolution if given a default optimizer
import theano
theano.config.optimizer = "None"


def m(*args, **kwargs):
    dtype = kwargs.get('dtype', 'float32')
    """Makes a test matrix whose dimensions are the supplied arguments."""
    total = functools.reduce(operator.mul, args, 1)
    arr = np.array(range(-2, total-2), dtype=dtype)
    arr = np.reshape(arr, args)
    return arr


def n(*args):
    """Makes a test matrix whose dimensions are the supplied arguments.

    Differs from m only in what values it has."""
    total = functools.reduce(operator.mul, args, 1)
    arr = np.array(range(-11, total-11), dtype='float32')
    arr = np.reshape(arr, args)
    for i in range(5):
        if len(args) > i + 1:
            np.swapaxes(arr, 0, i+1)
    arr = np.reshape(arr, args)
    return arr

def r(*args):
    """Makes a test matrix whose dimensions are the supplied arguments. Uniform random values"""

    return np.random.uniform(0, 1.0, args)


def _conv_inp(IN, IC, OC, IS, KS, data_format=None):
    kernel_mat_np = m(*(KS + [IC, OC]))
    if data_format == 'channels_first':
        input_mat_np = m(*([IN] + [IC] + IS))
    else:
        input_mat_np = m(*([IN] + IS + [IC]))
    inputMat = input_mat_np
    kernelMat = kernel_mat_np
    return [inputMat, kernelMat, data_format]


def _separable_conv_inp(IN, IC, OC, CM, IS, KS, data_format=None):
    depth_kernel_mat = m(*(KS + [IC, CM]))
    point_kernel_mat = m(*([1]*len(KS) + [CM * IC, OC]))
    if data_format == 'channels_first':
        input_mat = m(*([IN] + [IC] + IS))
    else:
        input_mat = m(*([IN] + IS + [IC]))
    return [input_mat, depth_kernel_mat, point_kernel_mat, data_format]


def compareForwardExact(skip_theano=False, skip_tensorflow=False):
    """Decorates test methods, checking equality under multiple backends."""
    def decorator(test_func):
        def compare(self, *args):
            if not skip_theano:
                theano_result = test_func(self, th, *args).eval()
            if not skip_tensorflow:
                tf_session = tensorflow.Session()
                tf.set_session(tf_session)
                tensorflow_result_intermediate = test_func(self, tf, *args)
                tf_session.run(tensorflow.global_variables_initializer())
                tensorflow_result = tensorflow_result_intermediate.eval(session=tf_session)
            plaidml_result = test_func(self, pkb, *args).eval()

            if not skip_theano:
                npt.assert_array_equal(plaidml_result, theano_result, err_msg='x=plaidml, y=theano')
            if not skip_tensorflow:
                npt.assert_array_equal(plaidml_result, tensorflow_result, err_msg='x=plaidml, y=tensorflow')
                tf_session.close()
        return compare
    return decorator


def compareForwardClose(epsilon=1e-03, atol=1e-8, skip_theano=False, skip_tensorflow=False):
    """Decorates test methods, checking near-equality under multiple backends."""
    def decorator(test_func):
        def compare(self, *args):
            if not skip_theano:
                theano_result = test_func(self, th, *args).eval()
            if not skip_tensorflow:
                tf_session = tensorflow.Session()
                tf.set_session(tf_session)
                tensorflow_result_intermediate = test_func(self, tf, *args)
                tf_session.run(tensorflow.global_variables_initializer())
                tensorflow_result = tensorflow_result_intermediate.eval(session=tf_session)
            plaidml_result = test_func(self, pkb, *args).eval()
            if not skip_theano:
                npt.assert_allclose(plaidml_result, theano_result, rtol=epsilon, atol=atol,
                                    err_msg='x=plaidml, y=theano')
            if not skip_tensorflow:
                npt.assert_allclose(plaidml_result, tensorflow_result, rtol=epsilon, atol=atol,
                                    err_msg='x=plaidml, y=tensorflow')
                tf_session.close()
        return compare
    return decorator


def opTest(in_data, tol=1e-3, atol=1e-8, skip_theano=False, skip_tensorflow=False, verbose=False):
    # If using with non-tensor parameters, all tensor params must appear before
    # all non-tensor params
    def run_one_backend(self, data, test_func, b, *args):
        tf_session = tensorflow.Session()
        tf.set_session(tf_session)
        results = []
        with tf_session.as_default():
            x = [b.placeholder(shape = t.shape) for t in data if hasattr(t, 'shape')]
            xv = [b.variable(t, dtype='float32') for t in data if hasattr(t, 'shape')]
            ps = [t for t in data if not hasattr(t, 'shape')]
            grad_funcs = test_func(self, b, *(x + ps +  list(args)))
            funcs = test_func(self, b, *(xv + ps + list(args)))
            tf_session.run(tensorflow.global_variables_initializer())
            for gf, f in zip(grad_funcs, funcs):
                df = b.gradients(b.mean(gf), x)
                gfn = b.function(x, df, updates=[])
                fr = f.eval()
                gr = gfn([t for t in data if hasattr(t, 'shape')])
                if verbose:
                    print(b, fr, gr)
                results.append((fr,gr))
        tf_session.close()
        return results

    def apply(test_func):
        def output(self, *args):
            for didx, data in enumerate(in_data):
                if not skip_theano:
                    theano_results = run_one_backend(self, data, test_func, th, *args)
                if not skip_tensorflow:
                    tensorflow_results = run_one_backend(self, data, test_func, tf, *args)
                plaidml_results = run_one_backend(self, data, test_func, pkb, *args)
                if not skip_theano:
                    for idx, (pmlr, thr) in enumerate(zip(plaidml_results, theano_results)):
                        idx = idx + 1
                        npt.assert_allclose(pmlr[0], thr[0], rtol=tol, atol=atol,
                                            err_msg='ERR: datum={}, test={}, x=plaidml, y=theano'.format(didx, idx))
                        for x in range(0, len(pmlr[1])):
                            npt.assert_allclose(pmlr[1][x], thr[1][x], rtol=tol, atol=atol,
                                                err_msg='ERR: datum={}, test={}, grad, x=plaidml, y=theano'.format(didx, idx))
                if not skip_tensorflow:
                    for idx, (pmlr, tfr) in enumerate(zip(plaidml_results, tensorflow_results)):
                        idx = idx + 1
                        npt.assert_allclose(pmlr[0], tfr[0], rtol=tol, atol=atol,
                                            err_msg='ERR: datum={}, test={}, x=plaidml, y=tensorflow'.format(didx, idx))
                        for x in range(0, len(pmlr[1])):
                            npt.assert_allclose(pmlr[1][x], tfr[1][x], rtol=tol, atol=atol,
                                                err_msg='ERR: datum={}, test={}, grad, x=plaidml, y=tensorflow'.format(didx, idx))
        return output
    return apply


class TestBackendOps(unittest.TestCase):
    """Tests PlaidML Keras operation definitions"""

    def a_testLearningPhase(self):
        # Test name prefixed with 'a_' because this needs to run before other tests
        npt.assert_equal(pkb.learning_phase().eval(), 0)
        pkb.set_learning_phase(1)
        npt.assert_equal(pkb.learning_phase().eval(), 1)
        pkb.set_learning_phase(0)
        npt.assert_equal(pkb.learning_phase().eval(), 0)

    @compareForwardExact()
    def testShape(self, b):
        return b.shape(b.variable(m(3,3,3,3,4)))

    @compareForwardExact()
    def testPassthrough(self, b):
        return b.variable(m(3,3))

    #@compareExact()
    #def testFp16(self, b):
    #  set_floatx('float16')
    #  return b.dot(b.variable(m(4, 4, dtype='float16')), b.variable(m(4, 4, dtype='float16')))

    @opTest([[m(3, 3), m(3, 3)], [m(2, 3, 4, 5), m(2, 3, 5, 2)]])
    def testDot(self, b, x, y):
        return [b.dot(x, y)]

    # TODO(T1046): Once Keras is updated beyond 2.0.8, re-enable TF on batch_dot tests
    @opTest([[m(10, 20), m(10, 30, 20), (1, 2)],
             [m(2, 3, 4, 5), m(2, 3, 5, 2), None],
             [m(2, 3, 4, 5), m(2, 16, 5, 3), (1, 3)],
             [m(2, 5), m(2, 5), 1],
             [m(2, 4, 5), m(2, 5, 2), None],],
            skip_tensorflow=True)
    def testBatchDot(self, b, x, y, ax):
        if ax is None:
            return [b.batch_dot(x, y)]
        else:
            return [b.batch_dot(x, y, axes=ax)]

    @opTest([[m(2, 3, 4, 5)]], skip_tensorflow=True)
    def testBatchDot2(self, b, x):
        return [b.batch_dot(x, b.variable(m(2, 3, 5, 2))),
                b.batch_dot(x, b.variable(m(2, 6, 5, 3)), axes=(1,3))]

    @opTest([[m(2, 5)]])
    def testBatchDot3(self, b, x):
        return [b.batch_dot(x, b.variable(m(2, 5)), axes=1)]

    @opTest([[m(2, 4, 5)]])
    def testBatchDot4(self, b, x):
        return [b.batch_dot(x, b.variable(m(2, 5, 2)))]

    #TODO: Does not need to exist longterm
    @unittest.skip("Helper test for debugging testAddElements, not standalone")
    def testMicroAddElementsFail(self):
        data = [m(3, 3), m(3, 3)]
        test_func = self.testAddElements
        args = list()
        ###############
        x = [pkb.placeholder(shape = t.shape) for t in data if isinstance(t, np.ndarray)]
        xv = [pkb.variable(t, dtype='float32') for t in data if isinstance(t, np.ndarray)]
        par = [t for t in data if not isinstance(t, np.ndarray)]
        grad_funcs = test_func(pkb, *(x + par + list(args)))
        funcs = test_func(pkb, *(xv + par + list(args)))
        #for gf, f in zip(grad_funcs, funcs):
        gf = grad_funcs[0]
        f = funcs[0]
        df = pkb.gradients(pkb.mean(gf), x)
        gfn = pkb.function(x, df, updates=[])
        fr = f.eval()
        gr = gfn([t for t in data if isinstance(t, np.ndarray)])
        if verbose:
            print(pkb, fr, gr)
        results.append((fr,gr))
        return results

    def testTileIdentity(self):
        x = pkb.variable(m(3))
        f = '''function (I[N]) -> (O) { O = I; }'''
        result = pkb._Op("TileIdent", x.dtype, (3,), f, {'I': x}, ['O'])
        output = result.eval()
        return 0

    def testTwoOutputs(self):
        x = pkb.variable(m(3))
        f = '''function (I[N]) -> (O1, O2) { O1 = I; O2 = I; }'''
        result = pkb._Op("TwoOut", x.dtype, (None,), f, {'I': x}, ['O1', 'O2'])
        output = result.eval()
        return 0

    @unittest.skip("TODO(T1028): This test is known to fail")
    @opTest([[m(3, 3), m(3, 3)]])
    def testAddElements(self, b, x, y):
        return [x + y]

    @opTest([[m(3, 3), 1.0]])
    def testAddConstant(self, b, x, c):
        return [x + c,
                c + x]

    @opTest([[m(3, 3), m(3, 3)]])
    def testSubElements(self, b, x, y):
        return [x - y]

    @opTest([[m(3, 3), 1.0]])
    def testSubConstant(self, b, x, c):
        return [x - c,
                c - x]

    @opTest([[m(3, 3), m(3, 3)],
             [m(2, 4), m(2, 4)],])
    def testMulElements(self, b, x, y):
        return [x * y]

    @opTest([[m(3, 3), 2.0]])
    def testMulConstant(self, b, x, c):
        return [x * c,
                c * x]

    @opTest([[m(3, 3), m(3, 3)],
             [m(2, 1, 1), m(1)],
             [m(2), m(1)]
    ], skip_theano=True)
    def testDivElements(self, b, x, y):
        return [x / y]

    @opTest([[m(3, 3), 2.0]])
    def testDivConstant(self, b, x, c):
        return [x / c,
                c / x]

    @opTest([[m(3, 3)],
             [m(3, 3), None, True],
             [m(2, 3, 4, 5), [1, 3]],
             [m(3, 4, 5), -1],])
    def testSum(self, b, x, ax=None, kd=False):
        return [b.sum(x, axis=ax, keepdims=kd)]

    # TODO(T1026): Switch to opTest once PROD AggregationOp supports derivatives
    @compareForwardExact()
    def testProd(self, b):
        return b.prod(b.variable(m(3,3)))

    # TODO(T1026): Switch to opTest once PROD AggregationOp supports derivatives
    @compareForwardExact()
    def testProdKeepdims(self, b):
        return b.prod(b.variable(m(3, 3)), keepdims=True)

    # TODO(T1026): Switch to opTest once PROD AggregationOp supports derivatives
    @compareForwardClose()
    def testProdAxis(self, b):
        return b.prod(b.variable(m(2,3,4,5)), axis=[1,3])

    # TODO(T1026): Switch to opTest once PROD AggregationOp supports derivs
    @compareForwardClose()
    def testProdNegAxis(self, b):
        return b.prod(b.variable(m(3,4,5)), axis=-1)

    @opTest([[m(3, 4)],
             [m(3, 3, 2), None, True],
             [m(4, 3, 2, 1), [1, 3]]])
    def testMax(self, b, x, ax=None, kd=False):
        return [b.max(x, axis=ax, keepdims=kd)]

    # T1031: This doesn't match TF/Theano on boundaries
    @opTest([[m(3, 4) - 3.3, m(3, 4) / 2.0]])
    def testMaximum(self, b, x, y):
        return [b.maximum(x, y)]

    @opTest([[m(2, 4, 3)]])
    def testMin(self, b, x):
        return [b.min(x)]

    # T1031: This doesn't match TF/Theano on boundaries
    @opTest([[m(4, 3) - 3.1, m(4, 3) / 2.0]])
    def testMinimum(self, b, x, y):
        return [b.minimum(x, y)]

    @opTest([[m(3, 3)],
             [m(3, 3), None, True],
             [m(2, 3, 4, 5), [1, 3]],
             [m(1, 2, 3, 4, 5), [-2, 2]], # Note: axis -2 means next to last axis
    ])
    def testMean(self, b, x, ax=None, kd=False):
        return [b.mean(x, axis=ax, keepdims=kd)]

    # T1031: This doesn't match TF/Theano on boundaries
    @opTest([[m(3, 3), 2.0001, 5.0001]])
    def testClip(self, b, x, lo, hi):
        return [b.clip(x, lo, hi)]

    # T1031: This doesn't match TF/Theano on corner
    @opTest([[m(3, 3) - 0.0001, 0.5, 3],
             [m(3, 4) + 0.0001, 0.1, 5],])
    def testRelu(self, b, x, a=0.0, m=None):
        return [b.relu(x),
                b.relu(x, alpha=a),
                b.relu(x, max_value=m),
                b.relu(x, alpha=a, max_value=m)]

    @compareForwardExact()
    def testEqual(self, b):
        return b.equal(b.variable(m(3, 3)), b.variable(m(3, 3)))

    @compareForwardExact()
    def testNotEqual(self, b):
        return b.not_equal(b.variable(m(3, 3)), b.variable(m(3, 3)))

    @compareForwardExact()
    def testLess(self, b):
        return b.less(b.variable(2 * m(3, 3)), b.variable(m(3, 3)))

    @compareForwardExact()
    def testLessEqual(self, b):
        return b.less_equal(b.variable(2 * m(3, 3)), b.variable(m(3, 3)))

    @compareForwardExact()
    def testGreater(self, b):
        return b.greater(b.variable(2 * m(3, 3)), b.variable(m(3, 3)))

    @compareForwardExact()
    def testGreaterEqual(self, b):
        return b.greater_equal(b.variable(2 * m(3, 3)), b.variable(m(3, 3)))

    @opTest([[m(3, 3)]])
    def testSquare(self, b, x):
        return [b.square(x)]

    @opTest([[m(2,3) + 3], [m(2,3,4) + 3]])
    def testSqrt(self, b, x):
        return [b.sqrt(x)]

    @opTest([[np.sqrt(m(5, 5, 10) + 2) - 3]], 1e-02, skip_theano=True)
    def testSoftmax(self, b, x):
        return [-b.log(b.softmax(x))]

    @opTest([[m(10, 10)]], skip_theano=True)
    def testSigmoid(self, b, x):
        return [b.sigmoid(x)]

    @opTest([[m(2, 2)]], skip_theano=True)
    def testBinaryCrossentropy(self, b, x):
        return [
            b.binary_crossentropy(b.variable(np.array([[0,1],[1,0]])), x, from_logits=True),
            b.binary_crossentropy(b.variable(np.array([[0.3,0.7],[0.1,0.9]])), x, from_logits=False),
            b.binary_crossentropy(b.variable(np.array([[0,0.7],[1,.3]])), b.sigmoid(x))
        ]

    @opTest([[np.array([[0,0,0], [0,1,0], [0,0,0]]), (m(3, 3) + 3) / 15.0],
             [np.array([0, 0, 1, 0, 0, 0]), (m(6) + 7) / 11.0]])
    def testCategoricalCrossentropy(self, b, x, y):
        return [b.categorical_crossentropy(x, y)]

    @opTest([[np.array([[0,0,0], [0,0,1], [1,1,0]]), (m(3, 3) + 3)]])
    def testSoftCat(self, b, x, y):
        return [b.categorical_crossentropy(x, b.softmax(y))]

    @unittest.skip("Doesn't need to agree b/c what we do with garbage input is implementation detail")
    @opTest([[(m(2, 2) + 3) / 10.0, np.array([[0., 0.], [1., 2.]])]])
    def testCategoricalCrossentropyGarbageIn(self, b, x, y):
        return [b.categorical_crossentropy(x, y)]

    #TODO: Merge with general cat xentropy if we can resolve the TF problems
    @opTest([[np.array([[0,0,0], [0,1,0], [0,0,0]]), (m(3, 3) + 3) / 15.0]],
            atol=1e-7, skip_tensorflow=True)
    def testCategoricalCrossentropyLogits(self, b, x, y):
        return [b.categorical_crossentropy(x, y, from_logits=True)]

    @opTest([[m(3, 3, 10)]], skip_theano=True, tol=0.01)
    def testSparseCategoricalCrossentropy(self, b, x):
        smax = b.softmax(x)
        sbest = b.variable(np.array([[7,8,5], [9,3,8], [0,7,6]]))
        return [b.sparse_categorical_crossentropy(sbest, smax),
                b.sparse_categorical_crossentropy(sbest, smax, from_logits=True)]

    @compareForwardExact(skip_theano=True)
    def testOneHot(self, b):
        A = b.variable(np.array([[0,1,2],[2,4,0],[0,2,7]]), dtype='int32')
        return b.one_hot(A, 20)

    @opTest([[m(20)], [m(7,3)]])
    def testExp(self, b, x):
        return [b.exp(x)]

    @opTest([[m(20)], [m(2,2,2)]])
    def testPow(self, b, x):
        return [b.pow(x, 5)]

    @opTest([[m(20) + 3], [m(10,3)]])
    def testLog(self, b, x):
        return [b.log(x)]

    @opTest([[m(10)],[m(2,2,2,3)]], 1e-2)
    def testTanh(self, b, x):
        return [b.tanh(x)]

    @compareForwardClose(.1)
    def testRandomNormalMean(self, b):
        rand = b.random_normal((1000, 1000), mean=42.0, stddev=0.1)
        return b.mean(rand)

    @compareForwardClose(.1)
    def testRandomNormalDev(self, b):
        rand = b.random_normal((1000, 1000), mean=42.0, stddev=0.1)
        mean = b.mean(rand)
        diffs = rand - mean
        return b.mean(b.square(diffs))

    @compareForwardClose(.1)
    def testTruncatedNormalMean(self, b):
        rand = b.truncated_normal((1000, 1000), mean=42.0, stddev=0.1)
        return b.mean(b.variable(rand))

    @compareForwardClose(.1, skip_theano=True)
    def testTruncatedNormalDev(self, b):
        rand = b.truncated_normal((1000, 1000), mean=42.0, stddev=0.1)
        X = b.variable(rand)
        mean = b.mean(X)
        diffs = X - mean
        return b.mean(b.square(diffs))

    def _separable_conv2d(self, b, IN, IH, IW, IC, OC, CM, KH, KW,
                          strides=(1,1), padding='same', data_format=None):
        depth_kernel_mat_np = m(KH, KW, IC, CM)
        point_kernel_mat_np = m(1, 1, CM*IC, OC)
        if data_format == 'channels_first':
            input_mat_np = m(IN, IC, IH, IW)
        else:
            input_mat_np = m(IN, IH, IW, IC)
        inputMat = b.variable(input_mat_np, dtype='float32')
        depthKernelMat = b.variable(depth_kernel_mat_np, dtype='float32')
        pointKernelMat = b.variable(point_kernel_mat_np, dtype='float32')
        return b.separable_conv2d(inputMat, depthKernelMat, pointKernelMat, padding=padding,
                                  strides=strides, data_format=data_format)

    @opTest([_conv_inp(IN=1, IC=16, OC=16, IS=[4, 5], KS=[3, 3]),],
            1e-04, skip_theano=True)
    def testWinograd(self, b, im, km, df):
        return [
            b.conv2d(im, km, padding='same', force_winograd=True) if b == pkb else b.conv2d(im, km, padding='same'),
        ]


    # Asymmetric stride examples not included for separable convolutions b/c they
    # aren't implemented in tensorflow (and theano doesn't do separable convs)
    @opTest([ _separable_conv_inp(IN=1, IC=2, OC=6, CM=3, IS=[8, 8], KS=[3, 3]),
              _separable_conv_inp(IN=4, IC=3, OC=6, CM=2, IS=[7, 9], KS=[3, 4]),
              _separable_conv_inp(IN=1, IC=2, OC=5, CM=1, IS=[10, 12], KS=[2, 5]),
              _separable_conv_inp(IN=2, IC=4, OC=8, CM=2, IS=[12, 12], KS=[3, 3],
                                  data_format='channels_first'),
    ],
            atol=1e-5,  # TF separable conv math is really loose, and ends up with
            # values like 2.59e-6 where a 0 should be.
            skip_theano=True)
    def testSeparableConv2d(self, b, im, dkm, pkm, df):
        return [
            b.separable_conv2d(im, dkm, pkm, padding='valid', strides=(2, 2), data_format=df),
            b.separable_conv2d(im, dkm, pkm, padding='valid', strides=(1, 1), data_format=df),
            b.separable_conv2d(im, dkm, pkm, padding='same', strides=(3, 3), data_format=df),
        ]

    @opTest(
        [_conv_inp(IN=1, IC=3, OC=1, IS=[5], KS=[2], data_format='channels_last'),
         _conv_inp(IN=2, IC=1, OC=4, IS=[5], KS=[3], data_format='channels_last')],
        # Tensorflow doesn't support 1d convos in this order yet
        #_conv_inp(IN=4, IC=1, OC=5, IS=[9], KS=[4], data_format='channels_first')],
            1e-04, skip_theano=True)
    def testConv1d(self, b, im, km, df):
        return [
            b.conv1d(im, km, padding='same', data_format=df),
            b.conv1d(im, km, padding='valid', data_format=df),
            b.conv1d(im, km, padding='valid', strides=(2), data_format=df),
        ]

    @opTest([
        _conv_inp(IN=2, IC=2, OC=4, IS=[4, 7], KS=[3, 3]),
        _conv_inp(IN=3, IC=3, OC=1, IS=[9, 8], KS=[2, 2], data_format='channels_last'),
        _conv_inp(IN=1, IC=1, OC=3, IS=[5, 4], KS=[3, 3], data_format='channels_first'),
        _conv_inp(IN=2, IC=4, OC=2, IS=[5, 5], KS=[2, 2], data_format='channels_first'),
    ],
            1e-04, skip_theano=True)
    def testConv2d(self, b, im, km, df):
        return [
            b.conv2d(im, km, padding='same', data_format=df),
            b.conv2d(im, km, padding='valid', data_format=df),
            b.conv2d(im, km, padding='same', strides=(2,2), data_format=df),
            b.conv2d(im, km, padding='valid', strides=(3,1), data_format=df),
        ]

    @unittest.skip("TODO(T1046): This case is bugged in Keras 2.0.8 TF")
    @opTest([_conv_inp(IN=1, IC=1, OC=1, IS=[1, 6], KS=[1, 1], data_format='channels_last')],
            1e-04, skip_theano=True)
    def testConv2dSpecial(self, b, im, km, df):
        '''A simplified example highlighting a bug in Keras 2.0.8 TF

        Probably doesn't need to be retained once the corresponding case in conv3d
        is fixed.'''
        return [b.conv2d(im, km, padding='same', strides=(2,3), data_format=df)]

    @opTest(
        [_conv_inp(IN=3, IC=1, OC=3, IS=[4, 7, 4], KS=[3, 3, 3]),
         _conv_inp(IN=3, IC=4, OC=2, IS=[3, 6, 3], KS=[2, 1, 2], data_format='channels_last'),
         _conv_inp(IN=2, IC=3, OC=1, IS=[5, 5, 3], KS=[3, 2, 2], data_format='channels_first'),
        ],
        1e-04, skip_theano=True)
    def testConv3d(self, b, im, km, df):
        return [
            b.conv3d(im, km, padding='same', data_format=df),
            # TODO(T1046): TF broken in Keras 2.0.8 on this; see testConv2dSpecial
            #b.conv3d(im, km, padding='same', strides=(2,3,3), data_format=df),
            b.conv3d(im, km, padding='valid', strides=(2,1,2), data_format=df),
        ]

    @opTest([[m(1, 4, 4, 1)], [m(1, 7, 5, 1)], [m(2, 11, 13, 3)]], skip_theano=True)
    def testAvgPool(self, b, x):
        return [b.pool2d(x, (2, 2), strides=(2, 2), pool_mode='avg'),
                b.pool2d(x, (3, 3), strides=(1, 1), pool_mode='avg', padding='same'),
                b.pool2d(x, (3, 4), strides=(2, 3), pool_mode='avg', padding='valid'),]

    @opTest([[m(1, 4, 4, 1)],
             [m(1, 9, 9, 1)],
             [m(1, 8, 10, 1)],
             [m(2, 9, 11, 3)],
    ],
            skip_theano=True)
    def testMaxPool(self, b, x):
        return [b.pool2d(x, (2, 2), strides=(2, 2), pool_mode='max'),
                b.pool2d(x, (3, 3), strides=(1, 1), pool_mode='max'),
                b.pool2d(x, (3, 3), strides=(2, 2), pool_mode='max'),
                b.pool2d(x, (2, 2), strides=(2, 2), pool_mode='max', padding='same'),
                b.pool2d(x, (3, 3), strides=(2, 2), pool_mode='max', padding='same')
        ]

    @opTest([[m(1, 1, 60), (60,)],
             [m(4, 3, 70, 2), (14, 10, 6, 2)],
             [m(7, 3, 2, 4), (-1,)],
             [m(4, 4), (-1,)]])
    def testReshape(self, b, x, s):
        return [b.reshape(x, s)]

    @opTest([[m(1, 1, 60), (60,)],
             [m(4, 3, 70, 2), (14, 10, 6, 2)],
             [m(7, 3, 2, 4), (-1,)],
             [m(4, 4), (-1,)],])
    def testTransposeReshape(self, b, x, s):
        return [b.reshape(b.transpose(x), s)]

    @opTest([[m(4, 2, 1, 3, 2), 2],
             [m(5, 3, 2, 1), -1]])
    def testSqueeze(self, b, x, ax):
        return [b.squeeze(x, ax)]

    @compareForwardExact()
    def testZeros(self, b):
        a = b.zeros(shape=(10,))
        return a

    @compareForwardExact()
    def testOnes(self, b):
        a = b.ones(shape=(10,))
        return a

    @compareForwardExact()
    def testConstant(self, b):
        a = b.constant(5, shape=(10,))
        return a

    # Note: we skip tensorflow since init_global must be called in the middle of this function
    # for correct semantics, and Theano is sufficient.
    @compareForwardExact(skip_tensorflow=True)
    def testUpdate(self, b):
        a = b.variable(m(10, 10))
        a2 = a * a
        up = b.update(a, a2)
        f = b.function([],[], updates=[up])
        f([])
        f([])
        return a

    @compareForwardExact()
    def testRandomChanges(self, b):
        a = b.random_uniform((10, 10))
        f = b.function([], [a])
        out1 = f([])[0]
        out2 = f([])[0]
        diff = np.abs(out1 - out2).max()
        if diff < .01:
            raise Exception("Random isn't random")
        return b.constant(0)

    # Note: This test assumes that our update code matches Theano's, and
    # that testing the second component of the returned update tuple is
    # sufficient. It may be worthwhile to make this test more resilient
    # to refactoring and make it test that the update portion is working
    # as expected.
    @compareForwardClose(skip_tensorflow=True)
    def testMovingAverageUpdate(self, b):
        return b.moving_average_update(b.variable(m(5,4,9,3,2)), b.variable(n(5,4,9,3,2)), 0.95)[1]

    @compareForwardClose(skip_tensorflow=True, atol=1e-6)
    def testBatchNormAndUpdate(self, b):
        b.set_learning_phase(1)
        x = b.variable(n(4, 7))
        moving_mean = b.variable(m(4, 1))
        moving_var = b.variable(m(4, 1))
        beta = b.zeros([4, 1])
        gamma = b.ones([4, 1])
        normed, mean, var = b.normalize_batch_in_training(x, gamma, beta, reduction_axes=[1])
        mean_update = b.moving_average_update(moving_mean, mean, 0.01)
        var_update = b.moving_average_update(moving_var, var, 0.01)
        f = b.function([], [], updates=[mean_update, var_update])
        f([])
        return moving_var

    @opTest([[m(2, 3, 5),
              m(2, 3, 1) + 3,
              m(2, 3, 1) + 4,
    ]],
            atol=1e-7)
    def testNormalizeBatchInTrainingSimple(self, b, x, mov_avg, mov_var):
        return [(b.normalize_batch_in_training(x, mov_avg, mov_var, [2]))[0]]

    @opTest([[n(2,3), np.array([3., 4.,.7]), np.array([1.44, .99, .98])]],
            skip_theano=True, skip_tensorflow=True)
    def testNormalizeBatchInTraining(self, b, x, beta, gamma):
        return [b.normalize_batch_in_training(x, gamma, beta, [1])[0]]

    @compareForwardClose(skip_tensorflow=True)
    def testNormalizeBatchInTrainingWeirdAxis(self, b):
        return b.normalize_batch_in_training(b.variable(n(5,4,7,3)), b.constant(0.8, shape=(5,1,7,3)), b.constant(-5, shape=(5,1,7,3)), [1])[1]

    @compareForwardClose(skip_tensorflow=True)
    def testNormalizeBatchInTrainingMultiAxis(self, b):
        return b.normalize_batch_in_training(b.variable(n(2,3,5,7,11)), b.constant(11, shape=(1,3,1,1,11)), b.constant(0, shape=(1,3,1,1,11)), [0,2,3])[2]

    @opTest([[n(4,3), np.array([0.0, 0.1, 0.1]), np.array([100., 101., 50.]),
              np.array([3., 4.,.7]), np.array([1.44, .99, .98])]])
    def testBatchNormalization(self, b, x, mean, var, beta, gamma):
        return [b.batch_normalization(x, mean, var, beta, gamma)]

    @opTest([[np.array([100])]], skip_theano=True)
    def testBatchNormalizationVar(self, b, var):
        return [b.batch_normalization(b.variable(n(1, 1, 2)), b.variable(np.array([15])), var, None, None),
                b.batch_normalization(b.variable(n(2, 1, 1)), b.variable(np.array([15])), var, None, None)]

    @opTest([[np.array([15])]], skip_theano=True)
    def testBatchNormalizationMean(self, b, mean):
        return [b.batch_normalization(b.variable(n(3,4,5)), mean, b.variable(np.array([100])), None, None)]

    @compareForwardClose()
    def testBatchNormalizationOneElement(self, b):
        x = b.variable(n(1,4,5))
        return b.batch_normalization(b.variable(n(1,4,5)), b.variable(np.array([15])), b.variable(np.array([100])), b.variable(np.array([3])), b.variable(np.array([1.44])))

    @compareForwardClose()
    def testBatchNormalizationNoBeta(self, b):
        return b.batch_normalization(b.variable(n(3,4,5)), b.variable(np.array([15])), b.variable(np.array([100])), None, b.variable(np.array([1.44])))

    @compareForwardClose()
    def testBatchNormalizationNoGamma(self, b):
        return b.batch_normalization(b.variable(n(3,4,5)), b.variable(np.array([15])), b.variable(np.array([100])), b.variable(np.array([3])), None)

    @opTest([[m(4, 6)],
             [m(4, 3, 5)],
             [m(3, 7), None, True],
             [m(2, 5, 4, 7, 3), 1],
    ], atol=1e-7)
    def testVarSimple(self, b, x, ax=None, kd=False):
        return [b.var(x, axis=ax, keepdims=kd)]

    @opTest([[m(3, 3)]])
    def testSelfMult(self, b, x):
        A = x
        return [b.dot(A, A)]

    @unittest.skip("TODO(T1037): This test is not yet working")
    @opTest([[np.array([[1.0, 2.0], [2.0, 7.0], [5.0, 6.0]])]])
    def testGather(self, b, v):
        I = b.variable(np.array([0, 2, 1, 0], dtype='int32'), dtype='int32')
        return [b.gather(v, I)]

    @compareForwardClose()
    def testGatherLong(self, b):
        V = b.variable(np.array([[1.0, 2.0], [2.0, 7.0], [5.0, 6.0]]))
        I = b.variable(np.array([[0, 1, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0]], dtype='int32'), dtype='int32')
        return b.gather(V, I)

    @compareForwardClose()
    def testGatherLong2(self, b):
        V = b.variable(np.array([[1.0, 2.0], [2.0, 7.0], [5.0, 6.0]]))
        I = b.variable(np.array([[[0, 1, 1, 0], [1, 0, 0, 1]], [[1, 0, 1, 0], [0, 0, 1, 1]]], dtype='int32'), dtype='int32')
        return b.gather(V, I)

    @opTest([[m(2, 3)]])
    def testRepeat(self, b, x):
        return [b.repeat(x, 4)]

    @opTest([[m(3, 2, 4, 5, 6)]])
    def testRepeatElements(self, b, x):
        return [b.repeat_elements(x, 3, 4)]

    @opTest([[m(4, 6, 9, 3), 3, 1, 'channels_last'],
             [m(2, 3, 12, 12), 2, 3, 'channels_first'],])
    def testResizeImages(self, b, x, h, w, df):
        return [b.resize_images(x, h, w, df)]

    @opTest([[m(3, 5)]])
    def testL2Normalize(self, b, x):
        return [b.l2_normalize(x, axis=1)]

    @opTest([[m(2, 3, 4), m(2, 3, 3), m(2, 3, 1)],
             [m(3, 2, 4), m(3, 1, 4), m(3, 3, 4), 1],])
    def testConcatenate(self, b, x, y, z, ax=-1):
        return [b.concatenate([x, y, z], axis=ax)]

    @compareForwardExact()
    def testZerosLike(self, b):
        A = b.variable(m(3,2,4))
        return b.zeros_like(A)

    @compareForwardExact()
    def testOnesLike(self, b):
        A = b.variable(m(3,2,4))
        return b.ones_like(A)

    @opTest([[m(3, 2, 4), 1],
             [m(2, 5, 3)],])
    def testExpandDims(self, b, x, ax=-1):
        return [b.expand_dims(x, ax)]

    @opTest([[m(3, 2, 4), [1, 7, 3]],
             [m(2, 3, 1), [2, 1, 4]]])
    def testTile(self, b, x, n):
        return [b.tile(x, n)]

    @opTest([[m(34)]])
    def testSliceBasic(self, b, x):
        return [b.exp(x[2:30]), b.log(x[:5]), b.tanh(x[-4:]), b.sqrt(x[-1])]

    @opTest([[m(4, 3, 3, 2, 5)]])
    def testSliceMessy(self, b, x):
        return [x[-1::-3,:2:2,-3:-2,::-1,-1:-5:-2]]

    @opTest([[m(2, 3, 2)]])
    def testSliceShort(self, b, x):
        return [x[1]]

    def testConvParameterRankExceptions(self):
        A = pkb.variable(m(2,3,1))
        B = pkb.variable(m(1,2,1))
        C = pkb.variable(m(2,2,2,1))
        with self.assertRaises(ValueError):
            pkb.conv(A, C)
        with self.assertRaises(ValueError):
            pkb.conv(A, B, strides=(2,3))
        with self.assertRaises(ValueError):
            pkb.conv(A, B, dilation_rate=(1,1))

    @compareForwardExact()
    def testCastToInt(self, b):
        A = b.variable(m(3,2,4))
        return b.cast(A, dtype='int16')

    @compareForwardExact()
    def testCastToFloat(self, b):
        A = b.variable(m(3,2,4))
        A2 = b.cast(A, dtype='int32')
        return b.cast(A, dtype='float32')

    @compareForwardExact()
    def testCastToUInt(self, b):
        A = b.variable(m(3,2,4))
        return b.cast(A + 2, dtype='uint8')

    # Th/TF disagree w/ us about negative zeros and I think the even/odd rounding
    # direction for numbers ending in *.5, so we'll settle for a close match.
    @compareForwardClose()
    def testRound(self, b):
        vals = np.array([[1.7, 0.8, 1.5], [0.9, -0.3, -0.8], [0, 1.7, 0.6]])
        return b.round(b.variable(vals))

    @opTest([[m(3, 2, 4), n(3, 2, 4), 0],
             [m(2, 3), n(2, 3), 1]])
    def testSwitch(self, b, e, t, c):
        c_tensor = b.variable(c)
        return [b.switch(c_tensor, e, t),]


if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    #plaidml._internal_set_vlog(4)
    testing.plaidml_config.default_config()
    unittest.main()
