# Copyright 2019 Intel Corporation.

from plaidml2.ffi import decode_str, ffi, ffi_call, lib


def all():
    nitems = ffi_call(lib.plaidml_settings_list_count)
    raw_keys = ffi.new('plaidml_string*[]', nitems)
    raw_values = ffi.new('plaidml_string*[]', nitems)
    ffi_call(lib.plaidml_settings_list, nitems, raw_keys, raw_values)
    return {decode_str(key): decode_str(value) for key, value in zip(raw_keys, raw_values)}


def get(key):
    ret = decode_str(ffi_call(lib.plaidml_settings_get, key.encode()))
    if ret is None:
        raise EnvironmentError('Could not find setting: {}'.format(key))
    return ret


def set(key, value):
    ffi_call(lib.plaidml_settings_set, key.encode(), value.encode())


def save():
    ffi_call(lib.plaidml_settings_save)


def load():
    ffi_call(lib.plaidml_settings_load)
    return all()
