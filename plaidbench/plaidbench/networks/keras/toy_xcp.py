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


def scale_dataset(x_train):
    import numpy as np
    # Upscale image size by a factor of 10
    x_train = np.repeat(np.repeat(x_train, 10, axis=1), 10, axis=2)
    # Crop the images to 199 x 199 and normalize
    return (x_train[:, 10:10 + 299, 10:10 + 299]) / 255.


def build_model(use_batchnorm=False):
    from keras.models import Model
    from keras.layers import Activation, BatchNormalization, Conv2D, Input, MaxPooling2D, SeparableConv2D
    import keras.backend as K

    # To use SeparableConv2D, we have to pretend to be TF
    old_backend = K.backend
    K.backend = lambda: "tensorflow"

    in_shape = (299, 299, 3)
    inp = Input(in_shape)
    b1c1 = Conv2D(64, (3, 3), strides=(2, 2), use_bias=False)(inp)
    b1c1relu = Activation('relu')(b1c1)

    b2sc1 = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(b1c1relu)
    if use_batchnorm:
        b2sc1 = BatchNormalization()(b2sc1)

    b2pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(b2sc1)

    outp = b2pool

    model = Model(inp, outp)

    K.backend = old_backend

    return model
