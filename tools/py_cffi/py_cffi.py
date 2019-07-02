# Copyright 2019 Intel Corporation.

import argparse
import pathlib
import io

import cffi
import pcpp


class Preprocessor(pcpp.Preprocessor):

    def on_include_not_found(self, is_system_include, curdir, includepath):
        pass


def preprocess(src):
    out = io.StringIO()
    cpp = Preprocessor()
    cpp.parse(src)
    cpp.write(out)
    return out.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--source', required=True, type=pathlib.Path, action='append')
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()

    ffibuilder = cffi.FFI()
    ffibuilder.set_source(args.module, None)
    for src in args.source:
        print(src)
        code = preprocess(src.read_text())
        # print(code)
        ffibuilder.cdef(code)
    ffibuilder.emit_python_code(args.output)
