# Copyright 2019 Intel Corporation.

import os
import sys

from setuptools import setup

from tools.py_setup import bazel_stage

CONSOLE_SCRIPTS = [
    'plaidml-setup = plaidml.plaidml_setup:main',
]

REQUIRED_PACKAGES = [
    'enum34 >= 1.1.6',
    'numpy',
    'six',
]


def main():
    bazel_stage()

    if sys.platform == 'win32':
        binary_name = 'plaidml2.dll'
    elif sys.platform == 'darwin':
        binary_name = 'libplaidml2.dylib'
    else:
        binary_name = 'libplaidml2.so'

    setup(
        name='plaidml2',
        version=os.getenv('BZL_VERSION'),
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
                'plaidml2/core/core.h',
                'plaidml2/core/ffi.h',
            ]),
            ('include/plaidml2/edsl', [
                'plaidml2/edsl/edsl.h',
                'plaidml2/edsl/ffi.h',
            ]),
            ('include/plaidml2/exec', [
                'plaidml2/exec/exec.h',
                'plaidml2/exec/ffi.h',
            ]),
            ('include/plaidml2/op', [
                'plaidml2/op/op.h',
                'plaidml2/op/ffi.h',
            ]),
            ('lib', [os.path.join('plaidml2', binary_name)]),
        ],
        description='PlaidML machine learning accelerator',
        entry_points={
            'console_scripts': CONSOLE_SCRIPTS,
        },
        include_package_data=True,
        install_requires=REQUIRED_PACKAGES,
        keywords='plaidml ml machine learning tensor compiler',
        license='https://www.apache.org/licenses/LICENSE-2.0',
        long_description='PlaidML is a framework for making machine learning work everywhere.',
        package_data={
            'plaidml2': [binary_name],
        },
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
