#!/usr/bin/env python

import argparse

import plaidml
import plaidml.settings


def print_devices(heading, flag):
    print(heading)
    ctx = plaidml.Context()
    plaidml.settings.experimental = flag
    matched, unmatched = plaidml.devices(ctx, limit=100, return_all=True)
    for dev in matched:
        print('{0}   {1: >40} : {2}'.format('*', dev.id.decode(), dev.description.decode()))
    for dev in unmatched:
        print('{0}   {1: >40} : {2}'.format(' ', dev.id.decode(), dev.description.decode()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()

    plaidml._internal_set_vlog(args.verbose)

    print_devices('Stable configurations', False)
    print_devices('Experimental configurations', True)


if __name__ == "__main__":
    main()
