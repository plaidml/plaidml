# Copyright 2021 Intel Corporation

from setuptools import Distribution, setup
from pathlib import Path
from sys import stderr


class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return True


def main():
    setup_package_name = 'mlperf'
    setup_package_dir = '@PROJECT_BINARY_DIR@'
    setup_wheel_name = '@_WHEEL_FILE@'
    result = setup(
        name=setup_package_name,
        version='1.1',
        description="MLPerf Inference benchmark",
        url="https://mlperf.org",
        packages=[setup_package_name],
        package_data={setup_package_name: ['@_MLPERF_LOADGEN_MODULE@']},
        package_dir={'': setup_package_dir},
        distclass=BinaryDistribution,
        zip_safe=False,
    )
    if 'bdist_wheel' in result.command_obj:
        bdist_wheel = result.command_obj['bdist_wheel']
        wheel_tags = '-'.join(bdist_wheel.get_tag())
        wheel_name = f'{bdist_wheel.wheel_dist_name}-{wheel_tags}.whl'
        if setup_wheel_name != wheel_name:
            wheel_path = Path(setup_package_dir) / setup_package_name
            wheel_orig = wheel_path / setup_wheel_name
            wheel_link = wheel_path / wheel_name
            wheel_orig.symlink_to(wheel_link)
            print(f'Symlinking {wheel_link} to {wheel_orig}', file=stderr)


if __name__ == "__main__":
    main()
