#! /usr/bin/env python

import argparse
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=pathlib.Path)
    parser.add_argument('dst', type=pathlib.Path)
    args = parser.parse_args()
    for child in args.src.iterdir():
        link = args.dst / child.name
        link.symlink_to(child)


if __name__ == '__main__':
    main()
