"""A generic setup.py template for use with Bazel.

To build a Python wheel, Bazel transforms this template into a per-package
setup.py, and then evaluates it."""

from __future__ import print_function

import os
import os.path
import shutil
import subprocess

from setuptools import setup


def main():
    """Builds the Python wheel."""
    os.chdir(os.path.dirname(__file__) or '.')

    if 'bzl_target_cpu' == 'x64_windows':
        subprocess.call('attrib -R /S')

    shutil.rmtree('build', ignore_errors=True)
    shutil.rmtree(os.path.join('pkg', 'bzl_package_name.egg-info'), ignore_errors=True)

    setup(
        name='bzl_package_name',
        version='bzl_version',
        package_dir={'': 'pkg'},
    )

    if 'bzl_target_cpu' == 'x64_windows':
        subprocess.call('attrib -R /S pkg\\bzl_package_name.egg-info')

    shutil.rmtree('build', ignore_errors=True)
    shutil.rmtree(os.path.join('pkg', 'bzl_package_name.egg-info'), ignore_errors=True)


if __name__ == '__main__':
    main()
