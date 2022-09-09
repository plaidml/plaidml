# Copyright 2019 Intel Corporation.

import plaidml
from plaidml.ffi import decode_str, ffi_call, lib


def all():
    return plaidml.kvps_to_dict(ffi_call(lib.plaidml_settings_list))


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
