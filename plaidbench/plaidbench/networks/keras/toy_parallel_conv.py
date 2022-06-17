# Copyright 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os


def scale_dataset(x_train):
    import numpy as np
    return np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)


def build_model():
    from keras.models import Model
    from keras.layers import Activation, Conv2D, Input
    from keras.layers import add as k_add

    in_shape = (224, 224, 3)
    inp = Input(in_shape)
    l1 = Activation('relu')(Conv2D(64, 7, strides=2, padding='same')(inp))
    l2a = Activation('relu')(Conv2D(256, 1, strides=2, padding='same')(l1))
    l2b = Activation('relu')(Conv2D(256, 3, strides=2, padding='same')(l1))
    outp = Activation('relu')(k_add([l2a, l2b]))
    model = Model(inp, outp)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(this_dir, 'networks', 'keras', 'toy_parallel_conv.h5')
    model.load_weights(weights_path)

    return model
