#!/usr/bin/env python3

import argparse
import jinja2
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Output root directory', default='out')
    parser.add_argument('--tile', help='Tile file to load', default='$layer_test4_float')
    args = parser.parse_args()

    src_path = pathlib.Path('../com_intel_plaidml/tile/ocl_exec')
    root_path = pathlib.Path(args.root).resolve()

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(src_path)),
        lstrip_blocks=True,
        trim_blocks=True,
        line_comment_prefix='#',
    )

    tmpl = env.get_template('gpu.json.j2')
    config = {
        'LOCAL_MEM_KIB': 32,
        'REGS_MEM_KIB': 16,
        'NUM_THREADS': 256,
        'SUBGROUP_SIZE': 32,
        'CACHE_WIDTH': 128,
    }
    root_path.mkdir(parents=True, exist_ok=True)
    json_path = root_path / 'config.json'
    with json_path.open('w') as fp:
        tmpl.stream(config).dump(fp)

    subprocess.run([src_path / 'ocl_exec', str(json_path), args.tile, str(root_path)])


if __name__ == '__main__':
    main()
