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

import os
import six
import tempfile

import click
import plaidml

from . import core


def _find_frontends():
    result = {}
    prefix = 'frontend_'
    suffix = '.py'
    dirname = os.path.dirname(__file__)
    for fname in os.listdir(dirname):
        if fname.startswith(prefix) and fname.endswith(suffix):
            result[fname[len(prefix):-len(suffix)]] = os.path.join(dirname, fname)
    return result


_FRONTENDS = _find_frontends()


def _get_frontend_mod(name):
    try:
        fname = _FRONTENDS[name]
    except KeyError:
        return None
    mod = {'__file__': fname}
    with open(fname) as f:
        code = compile(f.read(), fname, 'exec')
        eval(code, mod)
    return mod


class _PlaidbenchCommand(click.MultiCommand):

    def list_commands(self, ctx):
        return _FRONTENDS.keys()

    def get_command(self, ctx, name):
        return _get_frontend_mod(name)['cli']


@click.command(cls=_PlaidbenchCommand)
@click.option('-v', '--verbose', count=True)
@click.option('-n',
              '--examples',
              type=int,
              default=None,
              help='Number of examples to use (over all epochs)')
@click.option(
    '--blanket-run',
    is_flag=True,
    help='Run all networks at a range of batch sizes, ignoring the --batch-size and --examples '
    'options and the choice of network.')
@click.option('--results',
              type=click.Path(exists=False, file_okay=False, dir_okay=True),
              default=os.path.join(tempfile.gettempdir(), 'plaidbench_results'),
              help='Destination directory for results output')
@click.option('--callgrind/--no-callgrind',
              default=False,
              help='Invoke callgrind during timing runs')
@click.option('--epochs', type=int, default=1, help="Number of epochs per test")
@click.option('--batch-size', type=int, default=1)
@click.option('--timeout-secs', type=int, default=None)
@click.option('--warmup/--no-warmup', default=True, help='Do warmup runs before main timing')
@click.option('--kernel-timing/--no-kernel-timing', default=True, help='Emit kernel timing info')
@click.option('--print-stacktraces/--no-print-stacktraces',
              default=False,
              help='Print a stack trace if an exception occurs')
@click.pass_context
def plaidbench(ctx, verbose, examples, blanket_run, results, callgrind, epochs, batch_size,
               timeout_secs, warmup, print_stacktraces, kernel_timing):
    """PlaidML Machine Learning Benchmarks
    
    plaidbench runs benchmarks for a variety of ML framework, framework backend,
    and neural network combinations.

    For more information, see http://www.github.com/plaidml/plaidbench
    """
    runner = ctx.ensure_object(core.Runner)
    if blanket_run:
        runner.param_builder = core.BlanketParamBuilder(epochs)
        runner.reporter = core.BlanketReporter(os.path.expanduser(results))
        runner.reporter.configuration['train'] = False
    else:
        runner.param_builder = core.ExplicitParamBuilder(batch_size, epochs, examples)
        runner.reporter = core.ExplicitReporter(results)
    if verbose:
        plaidml._internal_set_vlog(verbose)
    runner.verbose = verbose
    runner.callgrind = callgrind
    runner.warmup = warmup
    runner.kernel_timing = kernel_timing
    runner.print_stacktraces = print_stacktraces
    runner.timeout_secs = timeout_secs
