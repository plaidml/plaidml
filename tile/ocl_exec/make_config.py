#!/usr/bin/env python

import jinja2
import pathlib
import os
import sys


def main():
    src_path = pathlib.Path(sys.argv[1])
    dir_path = src_path.parent
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(dir_path)),
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
    tmpl.stream(config).dump(sys.stdout)


if __name__ == '__main__':
    main()
