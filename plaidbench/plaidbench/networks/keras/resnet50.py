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
    # cifar10 data is only 32x32 pixels, so we must upsample by a factor of 7
    # to produce the 224x224 images required by resnet.
    return np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)


def build_model():
    import keras.applications as kapp
    from keras.backend import floatx
    from keras.layers import Input
    inputLayer = Input(shape=(224, 224, 3), dtype=floatx())
    return kapp.ResNet50(input_tensor=inputLayer)
