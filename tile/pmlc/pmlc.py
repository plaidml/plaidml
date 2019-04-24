#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import shutil

import yaml

import jinja2


class ParamsAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        params = getattr(namespace, 'params')
        for kvp in values:
            parts = kvp.split('=')
            params[parts[0]] = int(parts[1])


def main():
    parser = argparse.ArgumentParser(prog='pmlc')
    parser.add_argument('-v', '--verbose', help='Logging verbosity level', type=int, default=0)
    parser.add_argument('--pmlc', help='Path to pmlc binary', default='pmlc_bin')
    parser.add_argument(
        '-c',
        '--config',
        type=pathlib.Path,
        help='The configuration file to use for compilation.',
        required=True,
    )
    parser.add_argument(
        '-y',
        '--yaml',
        type=pathlib.Path,
        help='The parameters file to use for compilation.',
    )
    parser.add_argument(
        '-t',
        '--target',
        help='The target name to use within the specified --yaml file.',
    )
    parser.add_argument(
        '-D',
        '--outdir',
        help='The output directory.',
        type=pathlib.Path,
    )
    parser.add_argument(
        '--dump-passes',
        action='store_true',
        help='Enable dump passes.',
    )
    parser.add_argument(
        '-p',
        '--params',
        nargs='+',
        action=ParamsAction,
        metavar="KEY=VALUE",
        default={},
        help=
        'Parameters used for compilation. These values will override any parameters specified in a --yaml file.',
    )
    parser.add_argument('-r', '--run_under', help='Run pmlsim in a debugger')
    parser.add_argument(
        'tile',
        type=pathlib.Path,
        help='.tile source file to compile.',
    )
    args, remainder = parser.parse_known_args()

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(args.config.parent)),
        lstrip_blocks=True,
        trim_blocks=True,
        line_comment_prefix='##',
    )
    tmpl = env.get_template(args.config.name)
    if args.yaml:
        if not args.target:
            sys.exit('--target is required when using --yaml')
        with open(args.yaml) as fp:
            params = yaml.safe_load(fp).get(args.target)
    else:
        params = {}
    params.update(args.params)

    bin = shutil.which(args.pmlc)
    if not bin:
        bin = '../com_intel_plaidml/tile/pmlc/pmlc_bin'
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = pathlib.Path(tmp_dir) / 'config.json'
        with cfg_path.open('w') as fp:
            tmpl.stream(params).dump(fp)
        cmd = []
        if args.run_under:
            cmd += [args.run_under, '--']
        cmd += [bin, args.tile]
        cmd += ['--config', cfg_path]
        if args.outdir:
            shutil.copy(cfg_path, args.outdir / 'config.json')
            cmd += ['--outdir', args.outdir]
        if args.dump_passes:
            cmd += ['--dump-passes']
        if args.verbose > 0:
            cmd += ['-v', str(args.verbose)]
        cmd += remainder
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
