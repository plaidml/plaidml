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
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Logging verbosity level', type=int, default=0)
    parser.add_argument(
        '-c',
        '--config',
        help='The configuration file to use for compilation.',
        required=True,
    )
    parser.add_argument(
        '-y',
        '--yaml',
        help='The parameters file to use for compilation.',
    )
    parser.add_argument(
        '-t',
        '--target',
        help='The target name to use within the specified --yaml file.',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='The output path. If not specified, outputs to stdout.',
    )
    parser.add_argument(
        '-D',
        '--outdir',
        help='The output directory.',
    )
    parser.add_argument(
        '--dump-passes',
        action='store_true',
        help='Enable dumping of passes',
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
    parser.add_argument(
        'tile',
        help='.tile source file to compile.',
    )
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(cfg_path.parent)),
        lstrip_blocks=True,
        trim_blocks=True,
        line_comment_prefix='##',
    )
    tmpl = env.get_template(cfg_path.name)
    if args.yaml:
        if not args.target:
            sys.exit('--target is required when using --yaml')
        with open(args.yaml) as fp:
            params = yaml.safe_load(fp).get(args.target)
    else:
        params = {}
    params.update(args.params)

    bin = shutil.which('pmlc_bin')
    if not bin:
        bin = '../com_intel_plaidml/tile/pmlc/pmlc_bin'
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = pathlib.Path(tmp_dir) / 'config.json'
        with cfg_path.open('w') as fp:
            tmpl.stream(params).dump(fp)
        cmd = [bin, args.tile, '-config', cfg_path]
        if args.output:
            cmd += ['-out', args.output]
        if args.outdir:
            outdir_path = pathlib.Path(args.outdir)
            shutil.rmtree(outdir_path, ignore_errors=True)
            outdir_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(cfg_path, outdir_path / 'config.json')
            cmd += ['-outdir', args.outdir]
        if args.dump_passes:
            cmd += ['-dump-passes']
        if args.verbose > 0:
            cmd += ['-v', str(args.verbose)]
        subprocess.check_call(cmd)


if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        print(ex, file=sys.stderr)
