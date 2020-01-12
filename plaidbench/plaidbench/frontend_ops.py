# Copyright 2018 Intel Corporation
"""
# Tensor Comprehensions
## Methodology
TC GFLOP/s numbers were validated against nvprof 
first, obtain plaidbench results
`nvprof plaidbench -n 30 ops conv2d_odd_med --tc --no-tc-autotune --cuda-profile`
`plaidbench -n 30 ops conv2d_odd_med --tc --no-tc-autotune --cuda-profile`
"""

import importlib
import os

import numpy as np

import click
from plaidbench import core
from plaidbench.networks.ops import conv2d, dense

MS_OPS = {
    # need to add strides and possibly dialation if TC is coo "resnetup" : lambda p: conv2d.Conv2d(p, 64, 56, 56, 643)
    "conv2d_odd_sml": lambda p: conv2d.Conv2d(p, 16, 57, 57, 34, 3, 2),
    "conv2d_odd_med": lambda p: conv2d.Conv2d(p, 133, 16, 16, 266, 4, 4),
    "conv2d_resnet50_med": lambda p: conv2d.Conv2d(p, 256, 14, 14, 1024, 1, 1),
    "dense_odd_sml": lambda p: dense.Dense(p, 122, 98, 179),
    "dense_odd_med": lambda p: dense.Dense(p, 110, 512, 313),
    # need to add fusion support into TVM
    #"dense_med_relu" : lambda p: dense.Dense(p, 110, 512, 313, 'relu'),
}

LARGE_OPS = {
    "dense_odd_lrg": lambda p: dense.Dense(p, 333, 455, 633),
    "conv2d_vgg_lrg": lambda p: conv2d.Conv2d(p, 128, 122, 122, 128, 3, 3),
}

OPS = dict(MS_OPS)
OPS.update(LARGE_OPS)


class ProgramTimeFilter(object):

    def __init__(self):
        self.tot_time_ns = 0
        self.runs = 0

    def filter(self, record):
        msg = record.getMessage()
        if "Total program execution duration:" in msg:
            self.runs += 1
            self.tot_time_ns += int(msg.split(' ')[-1])
        return True


class Model(core.Model):

    def __init__(self, params):
        self.params = params

    def setup(self):
        self.op = OPS[self.params.network_name](self.params)
        self.flops = self.op.flops()
        click.secho("GFLOPS in this op: {:.6f}".format(self.flops / 10.0**9), fg='yellow')

    def compile(self):
        self.model = self.op.build_model()

    def run(self, once=False, warmup=False):
        loops = self.params.epoch_size // self.params.batch_size
        if once or warmup:
            loops = 1
        res = {}
        runfunc = getattr(self, 'run_{}'.format(self.params.backend_name))
        tm = None
        # TODO: Consider promoting to top level
        if self.params.backend_opts['cuda_profile']:
            import torch
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    tm = runfunc(loops)
        else:
            tm = runfunc(loops)

        if tm:
            res['time'] = tm
        res['flops'] = self.flops
        return (None, res)

    def run_plaid(self, loops):
        self.model.predict(x=self.op.get_dataset()[:loops * self.params.batch_size],
                           batch_size=self.params.batch_size)
        return (None, {})

    def run_tc(self, loops):
        import torch
        stop_watch = core.StopWatch(False)
        stop_watch.start()
        for _ in range(loops):
            self.model(*self.op.get_dataset(), cache=self.op.get_tc_cache())
        torch.cuda.synchronize()
        stop_watch.stop()
        return stop_watch.elapsed()

    def run_tf(self, loops):
        import tensorflow as tf
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(loops):
                sess.run(self.model)

    def run_tvm(self, loops):
        # This whole model needs to be refactored.
        # Ops need the ability to hold more state
        # Contexts
        return self.op.run_model_tvm(loops)

    def golden_output(self):
        raise core.NoCorrectnessDesired()


class Frontend(core.Frontend):
    NETWORK_NAMES = sorted(OPS.keys())

    def __init__(self, backend, networks, backend_opts):
        super(Frontend, self).__init__(networks)
        self.backend_opts = backend_opts
        self.backend = backend
        if backend == 'plaid':
            try:
                self.configuration['plaid'] = importlib.import_module('plaidml').__version__
                importlib.import_module('plaidml.bridge.keras')
            except ImportError:
                raise core.ExtrasNeeded(['plaidml-keras'])
        elif backend == 'tc':
            try:
                importlib.import_module('tensor_comprehensions')
            except ImportError:
                raise core.ExtrasNeeded(['torch', 'tensor_comprehensions'])
            if backend_opts['tc_cachedir']:
                try:
                    os.makedirs(backend_opts['tc_cachedir'])
                except OSError:
                    pass
        elif backend == 'tvm':
            try:
                importlib.import_module('tvm')
            except ImportError:
                raise core.ExtrasNeeded(['tvm', 'topi'])

    def model(self, params):
        return Model(params._replace(backend_opts=self.backend_opts))

    @property
    def name(self):
        return "ops"

    @property
    def init_args(self):
        return (self.backend, self.network_names, self.backend_opts)

    @property
    def blanket_batch_sizes(self):
        return [1, 2, 8, 128]


@click.command(cls=core.FrontendCommand, networks=Frontend.NETWORK_NAMES)
@click.option('--plaid',
              'backend',
              flag_value='plaid',
              default=True,
              help='Use PlaidML as the backend')
# It would be great to find another framework to run
# things against where we can reliably get granular timings
#@click.option(
#    '--tensorflow', 'backend', flag_value='tensorflow', help='Use TensorFlow as the backend')
@click.option('--tvm', 'backend', flag_value='tvm', help='Use TVM as the backend')
@click.option('--tc', 'backend', flag_value='tc', help='Use TensorComprehensions as the backend')
@click.option('--tf', 'backend', flag_value='tf', help='Use tensorflow as the backend')
@click.argument('networks', nargs=-1, type=click.Choice(Frontend.NETWORK_NAMES))
@click.option('--large-ops/--no-large-ops', default=True)
@click.option('--tc-autotune/--no-tc-autotune', default=True)
@click.option('--tc-cachedir', default="~/.plaidbench-tccache")
@click.option('--tc-at-generations', default=13)
@click.option('--tc-at-population', default=13)
@click.option('--tvm-driver', default='cuda')
@click.option('--cuda-profile/--no-cuda-profile', default=False)
@click.pass_context
def cli(ctx, backend, networks, large_ops, tc_autotune, tc_cachedir, tc_at_generations,
        tc_at_population, tvm_driver, cuda_profile):
    """Benchmarks compiled tensor kernels."""
    tc_cachedir = os.path.expanduser(tc_cachedir)
    runner = ctx.ensure_object(core.Runner)
    frontend = Frontend(
        backend, OPS if large_ops else MS_OPS, {
            'tc_autotune': tc_autotune,
            'tc_cachedir': tc_cachedir,
            'tc_at_generations': tc_at_generations,
            'tc_at_population': tc_at_population,
            'cuda_profile': cuda_profile,
            'tvm_driver': str(tvm_driver)
        })

    return runner.run(frontend, backend, networks)
