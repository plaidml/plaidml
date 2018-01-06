# Copyright Vertex.AI.

from __future__ import print_function

import hashlib
import os
import platform
import shutil
import sys
from subprocess import call, check_call

MAIN = "__BZL_MAIN__"
REQUIREMENTS = ["__BZL_REQUIREMENTS__"]
VENV_ARGS = __BZL_VENV_ARGS__
WORKSPACE = "__BZL_WORKSPACE__"


def _find_in_runfiles(logical_name):
    key = logical_name
    if key.startswith('external/'):
        key = key[len('external/'):]
    key = WORKSPACE + '/' + key
    try:
        return _find_in_runfiles.manifest.get(key, logical_name)
    except AttributeError:
        _find_in_runfiles.manifest = {}
        manifest_filename = None
        if 'RUNFILES_MANIFEST_FILE' in os.environ:
            manifest_filename = os.environ['RUNFILES_MANIFEST_FILE']
        elif 'RUNFILES_DIR' in os.environ:
            manifest_filename = os.path.join(os.environ['RUNFILES_DIR'], 'MANIFEST')
        if manifest_filename and os.path.exists(manifest_filename):
            with open(manifest_filename) as manifest:
                for line in manifest:
                    (logical, physical) = line.split(' ', 2)
                    _find_in_runfiles.manifest[logical] = physical.strip()
        return _find_in_runfiles.manifest.get(key, logical_name)


class VirtualEnv(object):

    def __init__(self, requirements):
        self._requirements = requirements
        hasher = hashlib.md5()
        for arg in VENV_ARGS:
            hasher.update(arg)
        for requirement in requirements:
            with open(_find_in_runfiles(requirement)) as file_:
                hasher.update(file_.read())
        self._path = os.path.join(os.path.expanduser('~'), '.t2', 'venv', hasher.hexdigest())

        if platform.system() == 'Windows':
            self._venv_bin = os.path.join(self._path, 'Scripts')
        else:
            self._venv_bin = os.path.join(self._path, 'bin')
        self._pip = os.path.join(self._venv_bin, 'pip')
        self.python = os.path.join(self._venv_bin, 'python')

    def make(self):
        try:
            if not os.path.exists(self._path):
                if platform.system() == 'Windows':
                    vpython = []
                else:
                    vpython = ['-p', 'python2']
                check_call(['virtualenv'] + vpython + VENV_ARGS + [self._path])
                if platform.system() == 'Darwin':
                    check_call([
                        self.python, self._pip, 'install',
                        'git+https://github.com/gldnspud/virtualenv-pythonw-osx.git'
                    ])
                    check_call([
                        self.python,
                        os.path.join(self._venv_bin, 'fix-osx-virtualenv'), self._path
                    ])
                for requirement in self._requirements:
                    check_call(
                        [self._pip, 'install', '-r',
                         _find_in_runfiles(requirement)])
        except:
            if os.path.exists(self._path):
                shutil.rmtree(self._path)
            raise
        env = dict(os.environ)
        env['VIRTUAL_ENV'] = self._path
        env['PATH'] = os.pathsep.join([self._venv_bin, os.getenv('PATH', "")])
        if platform.system() == 'Windows':
            env['PATHEXT'] = '.EXE'
        return env


def main():
    venv = VirtualEnv(REQUIREMENTS)
    env = venv.make()
    args = [venv.python, MAIN] + sys.argv[1:]
    args[1:] = [_find_in_runfiles(arg) for arg in args[1:]]
    print('Running in venv: {}'.format(venv._path))
    sys.exit(call(args, env=env))


if __name__ == '__main__':
    main()
