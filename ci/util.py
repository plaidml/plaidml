import argparse
import os
import platform
import shutil
import subprocess
import sys

if platform.system() == 'Windows':
    import win32api

verbose = False  # pylint: disable=invalid-name


def printf(*args, **kwargs):
    excludes_env = {key: kwargs[key] for key in kwargs if key not in ['env']}
    if excludes_env:
        print(*args, excludes_env)
    else:
        print(*args)
    sys.stdout.flush()


def call(cmd, **kwargs):
    if verbose:
        printf(cmd, **kwargs)
    return subprocess.call(cmd, **kwargs)


def check_call(cmd, **kwargs):
    if verbose:
        printf(cmd, **kwargs)
    subprocess.check_call(cmd, **kwargs)


def check_output(cmd, **kwargs):
    if verbose:
        printf(cmd, **kwargs)
    return subprocess.check_output(cmd, **kwargs)


class CondaEnv(object):

    def __init__(self, path):
        self.path = path.absolute().resolve()
        if platform.system() == 'Windows':
            self.bin = self.path / 'Scripts'
            self.python = self.path / 'python.exe'
            self.paths = [
                str(self.path),
                str(self.path / 'Library' / 'mingw-64' / 'bin'),
                str(self.path / 'Library' / 'usr' / 'bin'),
                str(self.path / 'Library' / 'bin'),
                str(self.path / 'Scripts'),
                str(self.path / 'bin'),
            ]
        else:
            self.bin = self.path / 'bin'
            self.python = self.bin / 'python'
            self.paths = [str(self.bin)]

    def env(self):
        env = {
            'CONDA_DEFAULT_ENV': str(self.path),
            'PATH': os.pathsep.join(self.paths + os.getenv('PATH').split(os.pathsep)),
        }
        if platform.system() != 'Windows':
            env['JAVA_HOME'] = str(self.path)
        else:
            env['JAVA_HOME'] = str(self.path / 'Library')
        return env

    def create(self, spec):
        try:
            if not self.path.exists():
                check_call(['conda', 'env', 'create', '-f', spec, '-p', str(self.path)])
            else:
                check_call(['conda', 'env', 'update', '--prune', '-f', spec, '-p', str(self.path)])
        except:
            if self.path.exists():
                shutil.rmtree(self.path)
            raise

    def clone(self, path):
        if path.exists():
            shutil.rmtree(path)
        check_call(['conda', 'create', '--clone', str(self.path), '-p', str(path)])
        return CondaEnv(path)

    def install(self, package):
        check_call([self.python, '-m', 'pip', 'install', package])


class DictAction(argparse.Action):

    def __init__(self, **kwargs):

        def key_value(string):
            return string.split('=', 1)

        super(DictAction, self).__init__(default={}, type=key_value, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        k, v = values
        var = getattr(namespace, self.dest)
        var[k] = v
        setattr(namespace, self.dest, var)


def first(choices):
    for choice in choices:
        if choice is not None:
            return choice
    return None


class PlanOption(object):

    def __init__(self, suite, workload, platform):
        self._suite = suite
        self._workload = workload
        self._platform = platform

    def get(self, name, default=None):
        """
        precedence order for options:
        - platform_overrides
        - workload
        - suite
        - default
        - None
        """
        override = self._workload.get('platform_overrides', {}).get(self._platform, {}).get(name)
        return first([
            override,
            self._workload.get(name),
            self._suite.get(name),
            default,
        ])


class Platform(object):

    def __init__(self, full, gpu_flops):
        parts = full.split('-')
        self.full = full
        self.framework = parts[0]
        self.engine = '_'.join(parts[1:3])
        self.gpu = parts[3]
        self.gpu_flops = gpu_flops.get(self.gpu)

    def __repr__(self):
        return '<Platform({})>'.format(self.full)


class TestInfo(object):

    def __init__(self, suite, workload, platform, batch_size):
        self.suite_name, self.suite = suite
        self.workload_name, self.workload = workload
        self.platform_name, self.platform = platform
        self.batch_size = batch_size

    def __repr__(self):
        return '{}/{}/{}/bs{}'.format(self.suite_name, self.workload_name, self.platform_name,
                                      self.batch_size)

    def label(self):
        label_parts = [self.platform.gpu, self.workload_name]
        if self.batch_size:
            label_parts += [str(self.batch_size)]
        label_parts += [self.platform.engine]
        return '-'.join(label_parts)

    def path(self, root):
        batch_size = 'BATCH_SIZE={}'.format(self.batch_size)
        return root / self.suite_name / self.workload_name / self.platform_name / batch_size
