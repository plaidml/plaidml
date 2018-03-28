# Copyright Vertex.AI

import ctypes
import logging
import os
import plaidml.exceptions

_LOGGER_FUNCTYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)

_LOG_SEVERITY_VERBOSE = 64
_LOG_SEVERITY_TRACE = 2
_LOG_SEVERITY_DEBUG = 4
_LOG_SEVERITY_INFO = 128
_LOG_SEVERITY_WARNING = 32
_LOG_SEVERITY_ERROR = 16
_LOG_SEVERITY_FATAL = 8

_LOG_SEVERITY_MAP = {
    _LOG_SEVERITY_VERBOSE: logging.DEBUG,
    _LOG_SEVERITY_TRACE: logging.DEBUG,
    _LOG_SEVERITY_DEBUG: logging.DEBUG,
    _LOG_SEVERITY_INFO: logging.INFO,
    _LOG_SEVERITY_WARNING: logging.WARNING,
    _LOG_SEVERITY_ERROR: logging.ERROR,
    _LOG_SEVERITY_FATAL: logging.CRITICAL
}

_PLAIDML_STATUS_CANCELLED = 1
_PLAIDML_STATUS_UNKNOWN = 2
_PLAIDML_STATUS_INVALID_ARGUMENT = 3
_PLAIDML_STATUS_DEADLINE_EXCEEDED = 4
_PLAIDML_STATUS_NOT_FOUND = 5
_PLAIDML_STATUS_ALREADY_EXISTS = 6
_PLAIDML_STATUS_PERMISSION_DENIED = 7
_PLAIDML_STATUS_RESOURCE_EXHAUSTED = 8
_PLAIDML_STATUS_FAILED_PRECONDITION = 9
_PLAIDML_STATUS_ABORTED = 10
_PLAIDML_STATUS_OUT_OF_RANGE = 11
_PLAIDML_STATUS_UNIMPLEMENTED = 12
_PLAIDML_STATUS_INTERNAL = 13
_PLAIDML_STATUS_UNAVAILABLE = 14
_PLAIDML_STATUS_DATA_LOSS = 15
_PLAIDML_STATUS_UNAUTHENTICATED = 16

_PLAIDML_ERRMAP = {
    _PLAIDML_STATUS_CANCELLED: plaidml.exceptions.Cancelled,
    _PLAIDML_STATUS_UNKNOWN: plaidml.exceptions.Unknown,
    _PLAIDML_STATUS_INVALID_ARGUMENT: plaidml.exceptions.InvalidArgument,
    _PLAIDML_STATUS_DEADLINE_EXCEEDED: plaidml.exceptions.DeadlineExceeded,
    _PLAIDML_STATUS_NOT_FOUND: plaidml.exceptions.NotFound,
    _PLAIDML_STATUS_ALREADY_EXISTS: plaidml.exceptions.AlreadyExists,
    _PLAIDML_STATUS_PERMISSION_DENIED: plaidml.exceptions.PermissionDenied,
    _PLAIDML_STATUS_RESOURCE_EXHAUSTED: plaidml.exceptions.ResourceExhausted,
    _PLAIDML_STATUS_FAILED_PRECONDITION: plaidml.exceptions.FailedPrecondition,
    _PLAIDML_STATUS_ABORTED: plaidml.exceptions.Aborted,
    _PLAIDML_STATUS_OUT_OF_RANGE: plaidml.exceptions.OutOfRange,
    _PLAIDML_STATUS_UNIMPLEMENTED: plaidml.exceptions.Unimplemented,
    _PLAIDML_STATUS_INTERNAL: plaidml.exceptions.Internal,
    _PLAIDML_STATUS_UNAVAILABLE: plaidml.exceptions.Unavailable,
    _PLAIDML_STATUS_DATA_LOSS: plaidml.exceptions.DataLoss,
    _PLAIDML_STATUS_UNAUTHENTICATED: plaidml.exceptions.Unauthenticated
}


class _C_Context(ctypes.Structure):
    pass


class Library(object):
    """A loaded PlaidML implementation library."""

    def __init__(self, lib, logger=logging.log):
        self._lib = lib
        self._logger = logger

        self.vai_last_status = lib.vai_last_status
        self.vai_last_status.argtypes = []

        self.vai_clear_status = lib.vai_clear_status
        self.vai_clear_status.argtypes = []

        self.vai_last_status_str = lib.vai_last_status_str
        self.vai_last_status_str.argtypes = []
        self.vai_last_status_str.restype = ctypes.c_char_p

        self.vai_set_logger = lib.vai_set_logger
        self.vai_set_logger.argtypes = [_LOGGER_FUNCTYPE, ctypes.c_void_p]

        self.vai_internal_set_vlog = lib.vai_internal_set_vlog
        self.vai_internal_set_vlog.argtypes = [ctypes.c_size_t]

        self.vai_get_perf_counter = lib.vai_get_perf_counter
        self.vai_get_perf_counter.argtypes = [ctypes.c_char_p]
        self.vai_get_perf_counter.restype = ctypes.c_longlong

        self.vai_set_perf_counter = lib.vai_set_perf_counter
        self.vai_set_perf_counter.argtypes = [ctypes.c_char_p, ctypes.c_longlong]

        self.vai_alloc_ctx = lib.vai_alloc_ctx
        self.vai_alloc_ctx.argtypes = []
        self.vai_alloc_ctx.restype = ctypes.POINTER(_C_Context)
        self.vai_alloc_ctx.errcheck = self._check_err

        self.vai_free_ctx = lib.vai_free_ctx
        self.vai_free_ctx.argtypes = [ctypes.POINTER(_C_Context)]

        self.vai_cancel_ctx = lib.vai_cancel_ctx
        self.vai_cancel_ctx.argtypes = [ctypes.POINTER(_C_Context)]

        self.vai_set_eventlog = lib.vai_set_eventlog
        self.vai_set_eventlog.argtypes = [ctypes.POINTER(_C_Context), ctypes.c_char_p]
        self.vai_set_eventlog.restype = ctypes.c_bool
        self.vai_set_eventlog.errcheck = self._check_err

        self._logger_wrapper = _LOGGER_FUNCTYPE(self._logger_callback)
        lib.vai_set_logger(self._logger_wrapper, None)

    def _check_err(self, result, func, args):
        if result:
            return result
        self.raise_last_status()

    def last_status(self):
        try:
            exclass = _PLAIDML_ERRMAP[self._lib.vai_last_status()]
        except KeyError:
            return Exception(self._lib.vai_last_status_str().decode())
        return exclass(self._lib.vai_last_status_str().decode())

    def raise_last_status(self):
        raise self.last_status()

    def _logger_callback(self, unused_arg, level, msg):
        severity = _LOG_SEVERITY_MAP.get(level, logging.ERROR)
        self._logger(severity, msg.decode())

    def get_perf_counter(self, name):
        return self.vai_get_perf_counter(name)

    def set_perf_counter(self, name, value):
        return self.vai_set_perf_counter(name, value)

    def _internal_set_vlog(self, l):
        self._lib.vai_internal_set_vlog(l)
