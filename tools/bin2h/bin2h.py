#!/usr/bin/env python3

import argparse
import pathlib

TEMPLATE = '''#include <string>

{entries}
'''

ENTRY = '''
std::string {symbol}("{elts}", {size});
'''


class DictAction(argparse.Action):

    def __init__(self, **kwargs):

        def key_value(string):
            return string.split('=', 1)

        super(DictAction, self).__init__(default={}, type=key_value, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        key, val = values
        var = getattr(namespace, self.dest)
        var[key] = pathlib.Path(val)
        setattr(namespace, self.dest, var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action=DictAction)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()

    entries = []
    for symbol, input in args.input.items():
        bytes = input.read_bytes()
        elts = ''.join(['\\x{:02X}'.format(x) for x in bytes])
        entries.append(ENTRY.format(symbol=symbol, elts=elts, size=len(bytes)))
    out_str = TEMPLATE.format(entries=''.join(entries))
    args.output.write_text(out_str)


if __name__ == '__main__':
    main()
