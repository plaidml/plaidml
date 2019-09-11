#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import sys

import requests

GET_BUILD_URL = 'https://api.buildkite.com/v2/organizations/plaidml/pipelines/{pipeline}/builds/{number}'
BUILDKITE_TOKEN = os.getenv('BUILDKITE_TOKEN')
BUILDKITE_PIPELINE_SLUG = os.getenv('BUILDKITE_PIPELINE_SLUG', 'plaidml-plaidml')
BUILDKITE_BUILD_NUMBER = os.getenv('BUILDKITE_BUILD_NUMBER')


def parse_dt(dt):
    return datetime.datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S.%fZ')


def dur(start, end):
    delta = parse_dt(end) - parse_dt(start)
    return int(delta / datetime.timedelta(microseconds=1))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', default=BUILDKITE_PIPELINE_SLUG)
    parser.add_argument('--number', default=BUILDKITE_BUILD_NUMBER)
    args = parser.parse_args()

    headers = {'Authorization': 'Bearer {}'.format(BUILDKITE_TOKEN)}
    params = {'pipeline': args.pipeline, 'number': args.number}
    resp = requests.get(GET_BUILD_URL.format(**params), headers=headers)
    build = resp.json()
    epoch = build.get('started_at')

    events = []
    agents = {}
    for tid, job in enumerate(build['jobs'], start=1):
        created_at = job.get('created_at')
        if not created_at:
            continue
        started_at = job.get('started_at')
        finished_at = job.get('finished_at')
        agent_id = job['agent']['id']
        agents[agent_id] = job['agent']
        name = job.get('name').replace(':hammer_and_wrench:', 'build')
        parts = name.split(':')
        if len(parts) > 6:
            name = parts[-1]
        events.append(
            dict(
                name='thread_name',
                ph='M',
                pid=agent_id,
                tid=tid,
                args=dict(name=name),
            ))
        events.append(
            dict(
                cat=name,
                name='job',
                ph='X',
                ts=dur(epoch, started_at),
                dur=dur(started_at, finished_at),
                pid=agent_id,
                tid=tid,
            ))
    for agent in agents.values():
        events.append(
            dict(
                name='process_name',
                ph='M',
                pid=agent['id'],
                args=dict(name=agent['name']),
            ))

    doc = dict(traceEvents=events)
    json.dump(doc, sys.stdout)


if __name__ == '__main__':
    main()
