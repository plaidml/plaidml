import argparse
import atexit
import os
import subprocess
import sys
import tarfile
import urllib.request
import venv
from pathlib import Path

OPEN_ZOO_RELEASE = os.getenv('OPEN_ZOO_RELEASE', '2021.3')
OPEN_ZOO_URL = f'https://github.com/openvinotoolkit/open_model_zoo/archive/refs/tags/{OPEN_ZOO_RELEASE}.tar.gz'
OPENCV_PYTHON_VER = '4.5.1.48'

AC_DIR = Path(f'open_model_zoo-{OPEN_ZOO_RELEASE}') / 'tools' / 'accuracy_checker'
DATASET_SRC = os.getenv('DATASET_SRC')
DATASET_DIR = Path('tmp/ac').resolve()


def benchmark_app(args):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = args.pkgdir.resolve()
    cmd = [args.pkgdir / 'benchmark_app']
    cmd += ['-d', 'PLAIDML']
    cmd += ['-m', args.model]
    cmd += ['-nireq', '1']
    cmd += ['-niter', '10']
    cmd += ['-report_type', 'no_counters']
    cmd += ['-report_folder', args.outdir]
    print(cmd)
    subprocess.check_call(cmd, env=env)


def accuracy_check(args):
    model_dir = args.model.parent
    CONDA_PREFIX = Path(os.getenv('CONDA_PREFIX'))

    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([
        str(args.pkgdir.resolve() / 'python_api' / 'python3.7'),
        str(AC_DIR.resolve()),
    ])
    env['LD_LIBRARY_PATH'] = os.pathsep.join([
        str(args.pkgdir.resolve()),
    ])
    print('PYTHONPATH:', env['PYTHONPATH'])
    print('LD_LIBRARY_PATH:', env['LD_LIBRARY_PATH'])

    ac_config = locate_ac_config(model_dir)
    if not ac_config:
        sys.exit(f'accuracy_checker.yml could not be found for: {model_dir}')

    venv_path = prepare_virtual_env()

    unmount_sshfs()
    atexit.register(unmount_sshfs)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(['sshfs', DATASET_SRC, DATASET_DIR])

    cmd = [venv_path / 'bin' / 'python']
    cmd += [venv_path / 'bin' / 'accuracy_check']
    cmd += ['-c', ac_config]
    cmd += ['-m', model_dir]
    cmd += ['-s', DATASET_DIR]
    cmd += ['-td', 'PLAIDML']
    cmd += ['-ss', '10']
    print(cmd)
    subprocess.check_call(cmd, env=env)


def unmount_sshfs():
    subprocess.call(['fusermount', '-u', DATASET_DIR])


def prepare_virtual_env(venv_path=Path('.venv')):
    print(f'Creating virtualenv: {venv_path}')
    venv_bin = venv_path / 'bin'
    venv.create(venv_path, with_pip=True)

    opencv = f'opencv-python-headless=={OPENCV_PYTHON_VER}'
    cmd = [venv_bin / 'pip', 'install', opencv]
    print(cmd)
    subprocess.check_call(cmd)

    # Download and extract accuracy checker
    print(f'Downloading: {OPEN_ZOO_URL}')
    zoo_tarball, _ = urllib.request.urlretrieve(OPEN_ZOO_URL)
    with tarfile.open(zoo_tarball) as zoo:
        zoo.extractall()

    # install AC into venv
    cmd = [venv_bin / 'python', AC_DIR / 'setup.py', 'install']
    print(cmd)
    subprocess.check_call(cmd)

    return venv_path


def locate_ac_config(model_dir):
    """Search through the model_dir branch for a file named
       accuracy_checker.yml"""

    while True:
        path = model_dir / 'accuracy_checker.yml'
        if path.is_file():
            return path
        if model_dir.parent == model_dir:
            return None
        model_dir = model_dir.parent


def main():
    dispatch = {
        'benchmark_app': benchmark_app,
        'accuracy_check': accuracy_check,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('test', choices=dispatch.keys())
    parser.add_argument('model', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--pkgdir', type=Path, default='testkit')
    args = parser.parse_args()

    if not args.pkgdir.exists():
        sys.exit(f'pkgdir could not be found: {args.pkgdir}')

    if not args.model.exists():
        sys.exit(f'model could not be found: {args.model}')

    if not args.outdir.exists():
        args.outdir.mkdir(parents=True, exist_ok=True)

    dispatch[args.test](args)


if __name__ == '__main__':
    main()
