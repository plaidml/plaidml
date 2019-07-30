# Copyright 2018 Vertex.AI
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

import math
import numpy as np
import os

from plaidbench.networks.ops import op


class Dense(op.Op):

    def __init__(self, params, i, j, k, activation=None):
        super(Dense, self).__init__(params)
        self.i = i
        self.j = j
        self.k = k
        self.activation = activation
        self.bs = params.batch_size

    def flops(self):
        flops = self.i * self.j * self.k * 2
        flops += self.j if self.activation else 0
        return flops

    def create_dataset_plaid(self):
        import numpy as np
        return np.zeros((self.params.epoch_size, self.i, self.k), dtype='float')

    def build_model_plaid(self):
        # Plaid doesn't actually need the batch size to be bound yet
        from keras.layers import Dense, Input
        from keras.models import Model
        inp = Input(shape=(self.i, self.k), name='data')
        op = Dense(self.j, name='dense', use_bias=True, activation=self.activation)(inp)
        return Model(inp, op, name=self.get_key())

    def get_key(self):
        return "dense{}".format("_".join(
            str(i) for i in [self.bs, self.i, self.j, self.k, self.activation]))

    def get_tc_cache(self):
        path = os.path.join(self.params.backend_opts['tc_cachedir'], self.get_key())
        try:
            os.makedirs(path)
        except IOError:
            pass
        return path

    def create_dataset_tc(self):
        # inp, weights, biases
        import torch
        return torch.randn(self.bs, self.i,
                           self.k).cuda(), torch.randn(self.k,
                                                       self.j).cuda(), torch.randn(self.j).cuda()

    def build_model_tc(self):
        actmap = {"relu": "fmax(OUT(n, i, j), 0)"}
        import tensor_comprehensions as tc
        lang = """
        def matmul(float(N, I, K) IN, float(K, J) W, float(J) B) -> (OUT) {
            OUT(n, i, j) +=! IN(n, i, k) * W(k, j)
            OUT(n, i, j) = OUT(n, i, j) + B(j)
         """
        if self.activation:
            lang += "OUT(n, i, j) = {}\n".format(actmap[self.activation])
        lang += "}"
        inp, wgt, bias = self.get_dataset()
        matmul = tc.define(lang, name="matmul")
        matmul.autotune(inp,
                        wgt,
                        bias,
                        cache=self.get_tc_cache(),
                        options=tc.Options("mlp"),
                        generations=self.params.backend_opts['tc_at_generations'],
                        pop_size=self.params.backend_opts['tc_at_population'],
                        elites=1,
                        threads=8)
        return matmul

    def create_dataset_tf(self):
        with tf.device('/gpu:0'):
            import tensorflow as tf
            # tf Graph input
            I = tf.constant([self.bs * self.i, self.k])
            J = tf.constant([self.k, self.j])
        return I, J

    def build_model_tf(self):
        import tensorflow as tf
        I, J = self.get_dataset()
        # Initialize the variables (i.e. assign their default value)
        return tf.matmul(I, J)

    def create_dataset_tvm(self):
        import tvm
        from topi.util import get_const_tuple
        # TVM does not support batched matmuls, so we fake it
        I = tvm.placeholder((self.bs * self.i, self.k), name='I')
        W = tvm.placeholder((self.j, self.k), name='W')
        B = tvm.placeholder((self.j,), name='B')

        i_shape = get_const_tuple(I.shape)
        w_shape = get_const_tuple(W.shape)
        b_shape = get_const_tuple(B.shape)
        dtype = I.dtype
        i_np = np.random.uniform(size=i_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=b_shape).astype(dtype)
        return I, W, B, i_np, w_np, b_np

    def run_model_tvm(self, loops):
        # Derived from here, taking some shortcuts: https://github.com/dmlc/tvm/blob/master/topi/tests/python/test_topi_conv2d_nchw.py
        import topi
        from topi.util import get_const_tuple
        import tvm
        from plaidbench import core
        device = self.params.backend_opts['tvm_driver']
        ctx = tvm.context(device, 0)
        with tvm.target.create(device):
            I, W, B, i_np, w_np, b_np = self.create_dataset_tvm()
            O = topi.nn.dense_default(I, W, B)
            t_sched = topi.cuda.dense.schedule_dense([O])
            i = tvm.nd.array(i_np, ctx)
            w = tvm.nd.array(w_np, ctx)
            b = tvm.nd.array(b_np, ctx)
            with tvm.build_config(auto_unroll_max_step=1400, unroll_explicit=device != 'cuda'):
                op = tvm.build(t_sched, [I, W, B], device, name=self.get_key())
                sw = core.StopWatch(False)
                sw.start()
                for _ in range(loops):
                    op(i, w, b)
                ctx.sync()
                sw.stop()
                return sw.elapsed()
