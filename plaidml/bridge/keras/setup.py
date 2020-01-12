# Copyright 2019 Intel Corporation.

import os

from setuptools import setup

from tools.py_setup import bazel_stage

REQUIRED_PACKAGES = [
    'keras == 2.2.4',
    'numpy',
    'plaidml',
    'scipy',
    'six',
]


def main():
    if os.getenv('BZL_SRC'):
        bazel_stage()

    setup(
        name='plaidml_keras',
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
        description='PlaidML Keras backend implementation',
        install_requires=REQUIRED_PACKAGES,
        keywords='plaidml ml machine learning tensor compiler keras',
        license='https://www.apache.org/licenses/LICENSE-2.0',
        long_description='PlaidML-Keras implements a Keras backend using PlaidML.',
        package_dir={'': 'com_intel_plaidml'},
        packages=[
            'plaidml.bridge.keras',
        ],
        url='https://www.intel.ai/plaidml',
    )


if __name__ == "__main__":
    main()
