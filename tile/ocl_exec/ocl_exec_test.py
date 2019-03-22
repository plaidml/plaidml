#!/usr/bin/env python3

import argparse
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Output root directory', default='out')
    parser.add_argument('--tile', help='Tile file to load', default='$matmul')
    args = parser.parse_args()

    root_path = pathlib.Path(args.root).resolve()
    src_path = pathlib.Path('../com_intel_plaidml/tile/ocl_exec')
    bin_path = src_path / 'bin'
    json_path = src_path / 'gpu.json'

    root_path.mkdir(exist_ok=True)

    cmd = [
        str(bin_path),
        str(json_path),
        args.tile,
        str(root_path),
    ]
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
