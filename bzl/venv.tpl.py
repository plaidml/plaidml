import hashlib
import os
import shutil
import sys
from subprocess import check_call

MAIN = "__BZL_MAIN__"
REQUIREMENTS = ["__BZL_REQUIREMENTS__"]
VENV_ARGS = __BZL_VENV_ARGS__


class VirtualEnv(object):

    def __init__(self, requirements):
        self._requirements = requirements
        hasher = hashlib.md5()
        for arg in VENV_ARGS:
            hasher.update(arg)
        for requirement in requirements:
            with open(requirement) as file_:
                hasher.update(file_.read())
        self._path = os.path.expanduser('~/.t2/venv/{}'.format(hasher.hexdigest()))

        self._venv_bin = os.path.join(self._path, 'bin')
        self._pip = os.path.join(self._venv_bin, 'pip')
        self.python = os.path.join(self._venv_bin, 'python')

    def make(self):
        try:
            if not os.path.exists(self._path):
                check_call(['virtualenv', '-p', 'python2'] + VENV_ARGS + [self._path])
                for requirement in self._requirements:
                    check_call([self.python, self._pip, 'install', '-r', requirement])
        except:
            if os.path.exists(self._path):
                shutil.rmtree(self._path)
            raise
        env = dict(os.environ)
        env['VIRTUAL_ENV'] = self._path
        env['PATH'] = os.pathsep.join([self._venv_bin, os.getenv('PATH')])
        return env


def main():
    venv = VirtualEnv(REQUIREMENTS)
    env = venv.make()
    args = [venv.python, MAIN] + sys.argv[1:]
    print('Running in venv: {}'.format(venv._path))
    os.execve(args[0], args, env)


if __name__ == '__main__':
    main()
