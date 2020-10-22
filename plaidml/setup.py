# Copyright 2019 Intel Corporation.

import os
import sys

from setuptools import setup

from tools.py_setup import bazel_stage

CONSOLE_SCRIPTS = [
    'plaidml-setup = plaidml.plaidml_setup:main',
]

REQUIRED_PACKAGES = [
    'cffi',
    'enum34 >= 1.1.6',
    'numpy',
    'six',
]


def main():
    if os.getenv('BZL_SRC'):
        bazel_stage()

    if sys.platform == 'win32':
        binary_name = 'plaidml.dll'
    else:
        binary_name = 'libplaidml.so'

    setup(
        name='plaidml',
        version=os.getenv('BZL_VERSION', '0.0.0'),
        author='Intel Corporation',
        author_email='plaidml-dev@googlegroups.com',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows :: Windows 10',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: C++',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        data_files=[
            ('include/plaidml/core', [
                'com_intel_plaidml/plaidml/core/core.h',
                'com_intel_plaidml/plaidml/core/ffi.h',
            ]),
            ('include/plaidml/edsl', [
                'com_intel_plaidml/plaidml/edsl/edsl.h',
                'com_intel_plaidml/plaidml/edsl/ffi.h',
            ]),
            ('include/plaidml/exec', [
                'com_intel_plaidml/plaidml/exec/exec.h',
                'com_intel_plaidml/plaidml/exec/ffi.h',
            ]),
            ('include/plaidml/op', [
                'com_intel_plaidml/plaidml/op/op.h',
                'com_intel_plaidml/plaidml/op/ffi.h',
            ]),
            ('lib', [os.path.join('com_intel_plaidml', 'plaidml', binary_name)]),
            ('share/plaidml', [
                'com_intel_plaidml/LICENSE',
                'com_intel_plaidml/plaidml/plaidml-config.cmake',
            ]),
            ('share/plaidml/boost', ['boost/LICENSE_1_0.txt']),
            ('share/plaidml/easylogging', ['easylogging/LICENSE']),
            ('share/plaidml/gmock', ['com_intel_plaidml/bzl/googlemock.LICENSE']),
            ('share/plaidml/half', ['half/LICENSE.txt']),
            ('share/plaidml/llvm', ['llvm-project/llvm/LICENSE.TXT']),
            ('share/plaidml/mlir', ['llvm-project/mlir/LICENSE.TXT']),
            ('share/plaidml/xsmm', ['xsmm/LICENSE.md']),
        ],
        description='PlaidML machine learning accelerator',
        entry_points={
            'console_scripts': CONSOLE_SCRIPTS,
        },
        install_requires=REQUIRED_PACKAGES,
        keywords='plaidml ml machine learning tensor compiler',
        license='https://www.apache.org/licenses/LICENSE-2.0',
        long_description='PlaidML is a framework for making machine learning work everywhere.',
        package_data={'plaidml': [binary_name]},
        package_dir={'': 'com_intel_plaidml'},
        packages=[
            'plaidml',
            'plaidml.core',
            'plaidml.edsl',
            'plaidml.exec',
            'plaidml.op',
        ],
        url='https://www.intel.ai/plaidml',
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
