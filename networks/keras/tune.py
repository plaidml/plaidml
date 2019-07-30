#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import multiprocessing
import os
import sys
import time
import pprint

import hyperopt
from hyperopt import hp


def run_vgg19(batch_size=64):
    import numpy as np
    import keras
    import plaidml

    # cifar10 data is 1/7th the size vgg19 needs in the spatial dimensions,
    # but if we upscale we can use it
    dataset = keras.datasets.cifar10

    # Load the dataset
    print("Loading the data")
    (x_train, y_train_cats), (x_test, y_test_cats) = dataset.load_data()

    # Get rid of all the data except the training images (for now
    y_train_cats = None
    x_test = None
    y_test_cats = None

    # truncate number of images
    x_train = x_train[:batch_size]

    # Upscale image size by a factor of 7
    print("Upscaling the data")
    x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)

    # Load the model
    print("Loading the model")
    model = keras.applications.VGG19()

    # Prep the model and run an initial un-timed batch
    print("Compiling")
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Running initial batch")
    y = model.predict(x=x_train, batch_size=batch_size)

    return plaidml.get_perf_counter("post_scan_time")


def run_inner(args, config, elapsed):
    import plaidml.keras
    if args.verbose:
        plaidml._internal_set_vlog(args.verbose)
    plaidml.keras.install_backend()
    import plaidml.keras.backend

    plaidml.keras.backend.set_config(config)
    elapsed.value = run_vgg19(args.batch_size)


def run_outer(args, config):
    try:
        elapsed = multiprocessing.Value('f', 0.0)
        proc = multiprocessing.Process(target=run_inner, args=(args, config, elapsed))
        proc.start()
        proc.join(args.timeout)
        if proc.is_alive():
            proc.terminate()
            print('Timeout')
            proc.join(3)
            return 0.0
        print('Elapsed: %s' % elapsed.value)
        return elapsed.value
    except Exception as ex:
        print('Exception: %s' % ex)
        return 0.0


def make_settings(params):
    return {
        'threads': {
            'value': params[0]
        },
        'vec_size': {
            'value': 1
        },
        'mem_width': {
            'value': params[1]
        },
        'max_mem': {
            'value': params[2]
        },
        'max_regs': {
            'value': params[3]
        },
        'goal_groups': {
            'value': params[4]
        },
        'goal_flops_per_byte': {
            'value': params[5]
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', const=1)
    parser.add_argument('--result', default='/tmp/result.json')
    parser.add_argument('--max_evals', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--timeout', type=int, default=500)
    args = parser.parse_args()
    print(args)
    name = args.name

    space = [
        hp.choice('threads', [1 << i for i in range(6, 10)]),
        hp.choice('mem_width', [1 << i for i in range(2, 10)]),
        hp.choice('max_mem', [i * 1024 for i in range(1, 48)]),
        hp.choice('max_regs', [i * 1024 for i in range(1, 48)]),
        hp.quniform('goal_groups', 1, 32, 1),
        hp.quniform('goal_fpb', 1, 50, 1),
    ]
    context = {'count': 0}

    def objective(params):
        context['count'] += 1
        print('-' * 20)
        print('Iteration: %d' % context['count'])
        print('-' * 20)
        settings = make_settings(params)
        config = {
            'platform': {
                '@type': 'type.vertex.ai/vertexai.tile.local_machine.proto.Platform',
                'hals': [{
                    '@type': 'type.vertex.ai/vertexai.tile.hal.opencl.proto.Driver',
                }],
                'settings_overrides': [{
                    'sel': {
                        'name_regex': name
                    },
                    'settings': settings
                }]
            }
        }

        #if settings['mem_width']['value'] < settings['vec_size']['value']:
        #    return {'status': hyperopt.STATUS_FAIL}

        elapsed = run_outer(args, json.dumps(config))
        if elapsed == 0.0:
            status = hyperopt.STATUS_FAIL
        else:
            status = hyperopt.STATUS_OK
        return {'status': status, 'loss': elapsed}

    trials = hyperopt.Trials()
    best = hyperopt.fmin(objective, space, hyperopt.tpe.suggest, args.max_evals, trials=trials)

    print('=' * 20)
    print('Best elapsed: %s' % trials.best_trial['result']['loss'])
    print('Best settings:')
    result = hyperopt.space_eval(space, best)
    pprint.pprint(make_settings(result))


if __name__ == '__main__':
    main()
