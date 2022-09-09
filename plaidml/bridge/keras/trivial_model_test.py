import os
import sys

os.environ['KERAS_BACKEND'] = 'plaidml.bridge.keras'

import numpy as np
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
    score = run()
    print('score:', score)
    expected = 1.66666666666666666666
    if (-0.0001 < score - expected and score - expected < 0.0001):
        sys.exit(0)
    sys.exit(1)
