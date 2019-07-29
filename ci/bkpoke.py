#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
import sys
import time

import requests

GET_BUILD_URL = 'https://api.buildkite.com/v2/organizations/vertex-dot-ai/pipelines/{pipeline}/builds/{number}'
CREATE_BUILD_URL = 'https://api.buildkite.com/v2/organizations/vertex-dot-ai/pipelines/{pipeline}/builds'
VIEW_BUILD_URL = 'https://buildkite.com/vertex-dot-ai/{pipeline}/builds/{number}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline')
    args = parser.parse_args()

    token = os.getenv('BUILDKITE_TOKEN')
    cwd = os.getenv('CI_PROJECT_DIR')
    commit = os.getenv('CI_COMMIT_SHA')
    branch = os.getenv('CI_COMMIT_REF_NAME')
    message = os.getenv('CI_COMMIT_TITLE')

    name = subprocess.check_output(['git', 'show', '-s', '--format=%an', commit], cwd=cwd)
    email = subprocess.check_output(['git', 'show', '-s', '--format=%ae', commit], cwd=cwd)
    headers = {'Authorization': 'Bearer {}'.format(token)}
    payload = {
        'commit': commit,
        'branch': branch,
        'message': message,
        'author': {
            'name': name.decode().rstrip(),
            'email': email.decode().rstrip(),
        },
    }
    params = {
        'pipeline': args.pipeline,
    }
    resp = requests.post(CREATE_BUILD_URL.format(**params), headers=headers, json=payload)
    print(resp)
    json = resp.json()
    print(json)
    params['number'] = json['number']
    print('{}: {}'.format(json['state'], VIEW_BUILD_URL.format(**params)), flush=True)

    while json['finished_at'] is None:
        time.sleep(30)
        resp = requests.get(GET_BUILD_URL.format(**params), headers=headers)
        json = resp.json()
        print('.', end='', flush=True)

    print('', flush=True)
    print('{}: {}'.format(json['state'], VIEW_BUILD_URL.format(**params)), flush=True)

    if json['state'] != 'passed':
        sys.exit(1)


if __name__ == '__main__':
    main()
