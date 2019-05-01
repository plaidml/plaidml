#!/usr/bin/env python3

import argparse
import pathlib
import sys

import yaml

import jinja2
from google.protobuf import json_format, text_format
from tile.codegen import codegen_pb2


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
        'config',
        type=pathlib.Path,
        help='The configuration file to use for compilation.',
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
        '-o',
        '--output',
        type=pathlib.Path,
        help='The output path. If not specified, outputs to stdout.',
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
        '-f',
        '--format',
        choices=[
            'json',
            'protobuf',
            'prototxt',
        ],
        default='json',
    )
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit('--config not found: {}'.format(args.config))
    if not args.yaml.exists():
        sys.exit('--yaml not found: {}'.format(args.yaml))

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(args.config.parent)),
        lstrip_blocks=True,
        trim_blocks=True,
        line_comment_prefix='##',
    )
    if args.yaml:
        if not args.target:
            sys.exit('--target is required when using --yaml')
        with args.yaml.open() as fp:
            params = yaml.safe_load(fp)[args.target]
    else:
        params = {}
    params.update(args.params)

    tmpl = env.get_template(args.config.name)
    json = tmpl.render(params)

    proto = codegen_pb2.Config()
    json_format.Parse(json, proto)

    if args.output:
        out = args.output.open('wb')
    else:
        out = sys.stdout.buffer

    if args.format == 'json':
        out.write(json.encode())
    elif args.format == 'prototxt':
        out.write(text_format.MessageToString(proto).encode())
    elif args.format == 'protobuf':
        out.write(proto.SerializeToString())


if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        print('{}: {}'.format(type(ex).__name__, ex), file=sys.stderr)
