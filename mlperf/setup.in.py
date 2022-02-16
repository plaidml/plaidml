# Copyright 2021 Intel Corporation

from setuptools import Distribution, setup


class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return True


def main():
    setup(
        name='mlperf',
        version='1.1',
        description="MLPerf Inference benchmark",
        url="https://mlperf.org",
        packages=['mlperf'],
        package_data={'mlperf': ['@_MLPERF_LOADGEN_MODULE@']},
        package_dir={'': '@PROJECT_BINARY_DIR@'},
        distclass=BinaryDistribution,
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
