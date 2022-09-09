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
    return x_train


def build_model(weights_path=None):
    import os
    from keras.models import Sequential
    from keras.layers import Dense, Embedding
    from keras.layers import LSTM

    model = Sequential()
    model.add(Embedding(20000, 128, input_length=80))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    if weights_path is not None:
        pass
    else:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(this_dir, 'networks', 'keras', 'imdb_lstm.h5')
    model.load_weights(weights_path)

    return model
