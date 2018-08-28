#!/usr/bin/env python

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('srcdir')
    parser.add_argument('outdir')
    parser.add_argument('--plantuml')
    args = parser.parse_args()
    if args.plantuml:
        os.environ['PLANTUMLSH'] = 'java -jar {}'.format(os.path.abspath(args.plantuml))
    subprocess.check_call(['python', '-m' 'sphinx', '-b', 'html', args.srcdir, args.outdir])


if __name__ == '__main__':
    main()
