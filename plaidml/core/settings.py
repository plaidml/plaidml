# Copyright 2019 Intel Corporation.

from plaidml.ffi import decode_str, ffi, ffi_call, lib


def all():
    settings = ffi_call(lib.plaidml_settings_list)
    try:
        x = settings.kvps
        return {decode_str(x[i].key): decode_str(x[i].value) for i in range(settings.nkvps)}
    finally:
        ffi_call(lib.plaidml_settings_free, settings)


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
