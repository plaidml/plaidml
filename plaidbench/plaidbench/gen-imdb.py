#!/usr/bin/env python
#
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

import numpy as np

# Load the dataset and scrap everything but the first 16 entries
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 20000
max_length = 80

if __name__ == '__main__':
    print('Fetching the imdb dataset')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = x_train[:16]
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    np.save('imdb16.npy', x_train)
