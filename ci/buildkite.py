#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import pystache

import ci.plan

DEFAULT_PIPELINE = 'plaidml'
PIPELINE = os.getenv('PIPELINE', os.getenv('BUILDKITE_PIPELINE_NAME', DEFAULT_PIPELINE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', default=PIPELINE)
    parser.add_argument('--count', action='store_true')
    args = parser.parse_args()

    plan = ci.plan.load('ci/plan.yml')
    selector = dict(pipeline=args.pipeline)
    actions = list(plan.get_actions(selector))
    if args.count:
        print('actions: {}'.format(len(actions)))
    else:
        ctx = dict(actions=actions)
        renderer = pystache.Renderer(escape=lambda x: x)
        path = Path('ci/pipeline.yml')
        yml = renderer.render(path.read_text(), ctx)
        print(yml)


if __name__ == '__main__':
    main()
