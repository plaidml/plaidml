# Copyright 2018 Intel Corporation.

import argparse
import os
import sys
import unittest

os.environ['KERAS_BACKEND'] = 'plaidml.bridge.keras'

import keras
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential

# from plaidml.bridge.keras import backend as pkb


class RegressionTests(unittest.TestCase):
    """PlaidML Keras regression tests"""

    @unittest.skipIf(os.environ.get("PLAIDML_USE_STRIPE", "0") == "1", "Stripe fails this test")
    def testBatchNormalizationWithFlatten(self):
        # This regression test is thanks to Hans Pinckaers (HansPinckaers on
        # GitHub), who reported https://github.com/plaidml/plaidml/issues/57,
        # showing a case where PlaidML was producing a kernel that was writing
        # to an output tensor that it already had that output as an input.
        inputs = Input(shape=(2, 2))
        x = Flatten()(inputs)
        x = BatchNormalization()(x)
        predictions = Dense(2, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        x_train = np.array([[[1, 1], [1, 1]]])
        y = model.predict(x=x_train, batch_size=1)
        model.fit(x_train, np.array([[0., 1.]]), batch_size=1, epochs=1, verbose=1)

    def testConv2ThenMaxPooling2DWithChannelsFirst(self):
        # This regression test is thanks to Paul Herz (phrz on GitHub),
        # who reported https://github.com/plaidml/plaidml/issues/61, a
        # straightforward Python typing issue in the backend when pooling
        # with channels_first using an existing op as input.
        model = Sequential()
        model.add(Reshape((1, 64, 64), input_shape=(1, 64 * 64)))
        model.add(Conv2D(filters=3, kernel_size=(5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

    def testRecompileWithChangingProgramCacheSize(self):
        # This test is thanks to iperov,
        # who reported https://github.com/plaidml/plaidml/issues/274,
        # demonstrating a case where exceeding certain number of ops
        # causes recompiling of kernels (the encoder is slightly modified from
        # his example for reproduciblilty)

        shape = (64, 64, 3)
        LeakyReLU = keras.layers.LeakyReLU

        def encflow(x):
            x = LeakyReLU()(keras.layers.Conv2D(128, 5, strides=2, padding="same")(x))
            x = keras.layers.Conv2D(128, 5, strides=2, padding="same")(x)
            x = keras.layers.Conv2D(256, 5, strides=2, padding="same")(x)
            x = keras.layers.Conv2D(256, 5, strides=2, padding="same")(x)
            x = keras.layers.Conv2D(256, 5, strides=2, padding="same")(x)
            x = keras.layers.Conv2D(512, 5, strides=2, padding="same")(x)
            x = keras.layers.Conv2D(512, 5, strides=2, padding="same")(x)
            x = keras.layers.Conv2D(1024, 5, strides=2, padding="same")(x)
            x = keras.layers.Conv2D(1024, 5, strides=2, padding="same")(x)
            x = keras.layers.Conv2D(1024, 5, strides=2, padding="same")(x)
            x = keras.layers.Dense(64)(keras.layers.Flatten()(x))
            x = keras.layers.Dense(4 * 4 * 1024)(x)
            x = keras.layers.Reshape((4, 4, 1024))(x)
            x = keras.layers.Conv2DTranspose(512, 3, strides=2, padding="same")(x)
            return x

        def decflow(x):
            x = x[0]
            x = LeakyReLU()(keras.layers.Conv2DTranspose(512, 3, strides=2, padding="same")(x))
            x = keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
            x = keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
            x = keras.layers.Conv2D(3, 5, strides=1, padding="same")(x)
            return x

        def modelify(model_functor):

            def func(tensor):
                return keras.models.Model(tensor, model_functor(tensor))

            return func

        encoder = modelify(encflow)(keras.Input(shape))
        decoder1 = modelify(decflow)([keras.Input(pkb.int_shape(x)[1:]) for x in encoder.outputs])
        decoder2 = modelify(decflow)([keras.Input(pkb.int_shape(x)[1:]) for x in encoder.outputs])

        inp = x = keras.Input(shape)
        code = encoder(x)
        x1 = decoder1(code)
        x2 = decoder2(code)

        loss = pkb.mean(pkb.square(inp - x1)) + pkb.mean(pkb.square(inp - x2))
        train_func = pkb.function([inp], [loss],
                                  keras.optimizers.Adam().get_updates(
                                      loss, encoder.trainable_weights +
                                      decoder1.trainable_weights + decoder2.trainable_weights))
        view_func1 = pkb.function([inp], [x1])
        view_func2 = pkb.function([inp], [x2])

        for i in range(5):
            print("Loop %i" % i, flush=True)
            data = np.zeros((1, 64, 64, 3))
            train_func([data])
            view_func1([data])
            view_func2([data])
            print("Saving weights", flush=True)
            encoder.save_weights(r"testweights.h5")
            decoder1.save_weights(r"testweights1.h5")
            decoder2.save_weights(r"testweights2.h5")

    def testWrongOutputDims(self):
        model_in = Input((10,), name='input')
        model_out = model_in
        model_out = Reshape((-1, 1), name='reshape')(model_out)
        model_out = Dot(axes=2, name='dot')([model_out, model_out])
        model_out = Flatten(name='flatten')(model_out)
        model = Model(model_in, model_out)
        model.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args, remainder = parser.parse_known_args()

    # plaidml._internal_set_vlog(args.verbose)
    # if args.fp16:
    #     pkb.set_floatx('float16')
    #     DEFAULT_TOL = 1e-2
    #     DEFAULT_ATOL = 1e-5
    # else:
    #     pkb.set_floatx('float32')

    unittest.main(argv=sys.argv[:1] + remainder, verbosity=args.verbose + 1)
