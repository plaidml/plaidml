#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import os
import sys
import time

import numpy as np


class StopWatch(object):

    def __init__(self, use_callgrind):
        self.__start = None
        self.__stop = None
        self.__use_callgrind = use_callgrind
        self.__callgrind_active = False
        self.__total = 0.0

    def start_outer(self):
        # Like start(), but does not turn on callgrind.
        self.__start = time.time()

    def start(self):
        self.__start = time.time()
        if self.__use_callgrind:
            os.system('callgrind_control --instr=on %d' % (os.getpid(),))
            self.__callgrind_active = True

    def stop(self):
        if self.__start is not None:
            stop = time.time()
            self.__total += stop - self.__start
            self.__start = None
        if self.__callgrind_active:
            self.__callgrind_active = False
            os.system('callgrind_control --instr=off %d' % (os.getpid(),))

    def is_running(self):
        return self.__start is not None

    def elapsed(self):
        return self.__total


class Output(object):

    def __init__(self):
        self.contents = None
        self.precision = 'untested'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorflow', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', const=3)
    parser.add_argument('--results', default='/tmp')
    parser.add_argument('--backtrace', action='store_true')
    parser.add_argument('--callgrind', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('module')
    args, remain = parser.parse_known_args()
    print(args, remain)

    if not args.tensorflow:
        import plaidml.keras
        if args.verbose:
            plaidml._internal_set_vlog(args.verbose)
        plaidml.keras.install_backend()
        if args.fp16:
            from keras.backend import set_floatx
            set_floatx('float16')
        if args.backtrace:
            plaidml.set_backtrace(True)

    stop_watch = StopWatch(args.callgrind)
    compile_stop_watch = StopWatch(args.callgrind)
    output = Output()
    globals = {
        '__name__': '__main__',
        'stop_watch': stop_watch,
        'compile_stop_watch': compile_stop_watch,
        'output': output,
        'batch_size': args.batch_size,
    }
    data = {
        'example': args.module,
        'args': list(sys.argv),
    }
    stop_watch.start_outer()
    compile_stop_watch.start_outer()
    try:
        sys.argc = len(remain) + 1
        sys.argv[1:] = remain
        print(sys.argv)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        module = os.path.join(this_dir, 'examples', '%s.py' % args.module)
        globals['__file__'] = module
        with open(module) as module_file:
            exec(module_file.read(), globals)
        stop_watch.stop()
        compile_stop_watch.stop()
        execution_duration = stop_watch.elapsed()
        compile_duration = compile_stop_watch.elapsed()
        data['execution_duration'] = execution_duration
        data['compile_duration'] = compile_duration
        print('Example finished, elapsed: {} (compile), {} (execution)'.format(
            compile_duration, execution_duration))
        data['precision'] = output.precision
    except Exception as ex:
        print(ex)
        data['exception'] = str(ex)
        raise
    finally:
        with open(os.path.join(args.results, 'result.json'), 'w') as out:
            json.dump(data, out)
        if isinstance(output.contents, np.ndarray):
            if args.verbose and args.verbose > 0:
                print("Result in output.contents:\n{}".format(output.contents))
            np.save(os.path.join(args.results, 'result.npy'), output.contents)


if __name__ == '__main__':
    main()
