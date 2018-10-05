# Copyright 2018 Intel Corporation.

import argparse
import sys
import unittest

# Install the plaidml backend
import plaidml.keras
plaidml.keras.install_backend()

from plaidml.keras import backend as pkb

import numpy as np

import keras
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, Flatten, MaxPooling2D, Reshape
from keras.models import Model, Sequential


class RegressionTests(unittest.TestCase):
    """PlaidML Keras regression tests"""

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args, remainder = parser.parse_known_args()

    plaidml._internal_set_vlog(args.verbose)
    if args.fp16:
        pkb.set_floatx('float16')
        DEFAULT_TOL = 1e-2
        DEFAULT_ATOL = 1e-5
    else:
        pkb.set_floatx('float32')

    unittest.main(argv=sys.argv[:1] + remainder, verbosity=args.verbose + 1)
