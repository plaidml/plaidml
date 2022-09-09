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
    # Upscale image size by a factor of 10
    x_train = np.repeat(np.repeat(x_train, 10, axis=1), 10, axis=2)
    # Crop the images to 199 x 199 and normalize
    return (x_train[:, 10:10 + 299, 10:10 + 299]) / 255.


def build_model():
    import keras.applications as kapp
    import keras.backend as K
    # Xception throws on initialization if the backend claims to be anything
    # other than "tensorflow", because it doesn't work on Theano or CNTK due to
    # its dependence on separable convolutions. Xception will work just fine
    # with PlaidML, but it doesn't know that, so we'll circumvent its name check
    # by monkeypatching the backend() function.
    old_backend = K.backend
    K.backend = lambda: "tensorflow"
    model = kapp.xception.Xception()
    K.backend = old_backend
    return model
