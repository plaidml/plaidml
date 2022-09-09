# Copyright 2018 Intel Corporation
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
    return np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)


def build_model():
    import keras.applications as kapp
    import keras.backend as K
    # MobileNet throws on initialization if the backend claims to be anything
    # other than "tensorflow", because tensorflow is the only backend built into
    # Keras which supports depthwise convolutions. It'll work fine with PlaidML,
    # but we have to fool the name check by monkeypatching backend().
    old_backend = K.backend
    K.backend = lambda: "tensorflow"
    model = kapp.mobilenet_v2.MobileNetV2()
    K.backend = old_backend
    return model
