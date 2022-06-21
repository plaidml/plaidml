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
    x_train = x_train[:, :, :8, 0]
    return np.reshape(x_train, (-1, 256))


def build_model():
    from keras.models import Model
    from keras.layers import Activation, Dense, Dropout, Input
    from keras.layers import add as k_add

    in_shape = (256,)
    inp = Input(in_shape)
    l1 = Activation('relu')(Dense(512)(inp))
    l2 = Activation('relu')(Dense(1024)(l1))
    if False:
        # Dropout currently disabled
        l2 = Dropout(0.5)(l2)
    l3 = Activation('relu')(Dense(2048)(l2))
    outp = Activation('relu')(Dense(1000)(l3))
    model = Model(inp, outp)

    return model
