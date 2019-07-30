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
    subprocess.call(cmd, **kwargs)


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
            env['CONDA_EXE'] = win32api.SearchPath(os.getenv('PATH'), 'conda', ".exe")[0]

        return env

    def create(self, spec):
        try:
            if not self.path.exists():
                check_call(['conda', 'env', 'create', '-f', spec, '-p', str(self.path)])
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
