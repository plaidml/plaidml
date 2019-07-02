# Copyright 2019 Intel Corporation.

import os

from plaidml2.ffi import decode_str, ffi, ffi_call, lib


def all():
    nitems = ffi_call(lib.plaidml_settings_list_count)
    raw_keys = ffi.new('plaidml_string*[]', nitems)
    raw_values = ffi.new('plaidml_string*[]', nitems)
    ffi_call(lib.plaidml_settings_list, nitems, raw_keys, raw_values)
    return {decode_str(key): decode_str(value) for key, value in zip(raw_keys, raw_values)}


def get(key):
    return os.getenv(key, all().get(key))


def set(key, value):
    pass
