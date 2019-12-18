# Copyright 2019 Intel Corporation.

import argparse
import csv
import gzip

import pystache

def load_template(path):
    with open(path, 'r') as fp:
        return fp.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('template')
    parser.add_argument('out')
    args = parser.parse_args()

    with gzip.open(args.csv) as fp:
        lines = fp.read().decode().splitlines()
    reader = csv.DictReader(lines)
    data = list(reader)

    tpl = load_template(args.template)
    ctx = {'size': {'SIZE': len(data)}, 'key': data, 'value': data}
    out = pystache.render(tpl, ctx)

    with open(args.out, 'w') as fp:
        fp.write(out)
