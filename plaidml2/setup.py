# Copyright 2019 Intel Corporation.

import os
import sys

from setuptools import setup

from tools.py_setup import bazel_stage

CONSOLE_SCRIPTS = [
    'plaidml2-setup = plaidml2.plaidml_setup:main',
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
        binary_name = 'plaidml2.dll'
    elif sys.platform == 'darwin':
        binary_name = 'libplaidml2.dylib'
    else:
        binary_name = 'libplaidml2.so'

    setup(
        name='plaidml2',
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
            ('include/plaidml2/core', [
                'com_intel_plaidml/plaidml2/core/core.h',
                'com_intel_plaidml/plaidml2/core/ffi.h',
            ]),
            ('include/plaidml2/edsl', [
                'com_intel_plaidml/plaidml2/edsl/edsl.h',
                'com_intel_plaidml/plaidml2/edsl/ffi.h',
            ]),
            ('include/plaidml2/exec', [
                'com_intel_plaidml/plaidml2/exec/exec.h',
                'com_intel_plaidml/plaidml2/exec/ffi.h',
            ]),
            ('include/plaidml2/op', [
                'com_intel_plaidml/plaidml2/op/op.h',
                'com_intel_plaidml/plaidml2/op/ffi.h',
            ]),
            ('lib', [os.path.join('com_intel_plaidml', 'plaidml2', binary_name)]),
            ('share/plaidml2', [
                'com_intel_plaidml/LICENSE',
                'com_intel_plaidml/plaidml2/plaidml2-config.cmake',
            ]),
            ('share/plaidml2/boost', ['boost/LICENSE_1_0.txt']),
            ('share/plaidml2/easylogging', ['easylogging/LICENCE.txt']),
            ('share/plaidml2/gmock', ['gmock/googlemock/LICENSE']),
            ('share/plaidml2/half', ['half/LICENSE.txt']),
            ('share/plaidml2/llvm', ['llvm-project/llvm/LICENSE.TXT']),
            ('share/plaidml2/mlir', ['llvm-project/mlir/LICENSE.TXT']),
            ('share/plaidml2/tbb', ['tbb/LICENSE']),
            ('share/plaidml2/xsmm', ['xsmm/LICENSE.md']),
        ],
        description='PlaidML machine learning accelerator',
        entry_points={
            'console_scripts': CONSOLE_SCRIPTS,
        },
        install_requires=REQUIRED_PACKAGES,
        keywords='plaidml ml machine learning tensor compiler',
        license='https://www.apache.org/licenses/LICENSE-2.0',
        long_description='PlaidML is a framework for making machine learning work everywhere.',
        package_data={'plaidml2': [binary_name]},
        package_dir={'': 'com_intel_plaidml'},
        packages=[
            'plaidml2',
            'plaidml2.core',
            'plaidml2.edsl',
            'plaidml2.exec',
            'plaidml2.op',
        ],
        url='https://www.intel.ai/plaidml',
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
