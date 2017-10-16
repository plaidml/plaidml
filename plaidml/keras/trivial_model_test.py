import numpy as np
import sys

import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend

import testing.plaidml_config

from keras.layers import Input
from keras.models import Model


def run():
    X_test = np.array([[1., 3., 4.]])
    Y_test = np.array([[0., 1., 4.]])

    img_input = Input(shape=(3,))
    model = Model(img_input, img_input, name='trivial')
    model.compile(loss='mean_squared_error', optimizer='sgd')

    score = model.evaluate(X_test, Y_test)
    return score


if __name__ == '__main__':
    plaidml.keras.backend.set_config(testing.plaidml_config.config())
    score = run()
    expected = 1.66666666666666666666
    if (-0.0001 < score - expected and score - expected < 0.0001):
        sys.exit(0)
    sys.exit(1)
