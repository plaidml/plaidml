# Copyright 2019 Intel Corporation.

import logging
import os
import platform
import sys
import threading

import pkg_resources

from plaidml._ffi import ffi

logger = logging.getLogger(__name__)

_TLS = threading.local()
_TLS.err = ffi.new('plaidml_error*')

_LIBNAME = 'plaidml'

if platform.system() == 'Windows':
    lib_name = '{}.dll'.format(_LIBNAME)
else:
    lib_name = 'lib{}.so'.format(_LIBNAME)

lib_path = os.getenv('PLAIDML_LIB_PATH')
if not lib_path:
    lib_path = pkg_resources.resource_filename(__name__, lib_name)


def __load_library():
    logger.debug('Loading {} from {}'.format(lib_name, lib_path))
    return ffi.dlopen(lib_path)


lib = ffi.init_once(__load_library, 'plaidml_load_library')


def decode_str(ptr):
    if ptr:
        try:
            return ffi.string(lib.plaidml_string_ptr(ptr)).decode()
        finally:
            lib.plaidml_string_free(ptr)
    return None


def decode_list(ffi_list, ffi_free, fn, *args):
    list = ffi_call(ffi_list, *args)
    if fn is None:
        fn = lambda x: x
    try:
        return [fn(list.elts[i]) for i in range(list.size)]
    finally:
        ffi_call(ffi_free, list)


class Error(Exception):

    def __init__(self, err):
        Exception.__init__(self)
        self.code = err.code
        self.msg = decode_str(err.msg)

    def __str__(self):
        return self.msg


def ffi_call(func, *args):
    """Calls ffi function and propagates foreign errors."""
    ret = func(_TLS.err, *args)
    if _TLS.err.code:
        raise Error(_TLS.err)
    return ret


class ForeignObject(object):
    __ffi_obj__ = None
    __ffi_del__ = None
    __ffi_repr__ = None

    def __init__(self, ffi_obj):
        self.__ffi_obj__ = ffi_obj

    def __del__(self):
        if self.__ffi_obj__ and self.__ffi_del__:
            self._methodcall(self.__ffi_del__)

    def __repr__(self):
        if self.__ffi_obj__ is None:
            return 'None'
        if self.__ffi_obj__ and self.__ffi_repr__:
            return decode_str(self._methodcall(self.__ffi_repr__))
        return super(ForeignObject, self).__repr__()

    def _methodcall(self, func, *args):
        return ffi_call(func, self.as_ptr(), *args)

    def as_ptr(self, release=False):
        if self.__ffi_obj__ is None:
            return ffi.NULL
        ret = self.__ffi_obj__
        if release:
            self.__ffi_obj__ = None
        return ret
