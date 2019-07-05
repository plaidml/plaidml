# Copyright 2019 Intel Corporation.

import contextlib

import numpy as np
import plaidml2 as plaidml
from plaidml2.ffi import ForeignObject, decode_str, ffi, ffi_call, lib


def __init():
    ffi_call(lib.plaidml_exec_init)


ffi.init_once(__init, 'plaidml_exec_init')


def list_devices():
    ndevices = ffi_call(lib.plaidml_device_list_count)
    raw_devices = ffi.new('plaidml_string*[]', ndevices)
    ffi_call(lib.plaidml_device_list, ndevices, raw_devices)
    return [decode_str(x) for x in raw_devices]


def list_targets():
    ntargets = ffi_call(lib.plaidml_target_list_count)
    raw_targets = ffi.new('plaidml_string*[]', ntargets)
    ffi_call(lib.plaidml_target_list, ntargets, raw_targets)
    return [decode_str(x) for x in raw_targets]


class Executable(ForeignObject):
    __ffi_del__ = lib.plaidml_executable_free

    def __init__(self, program, device_id, target, input_bindings, output_bindings):
        self._inputs = [x[1] for x in input_bindings]
        self._outputs = [x[1] for x in output_bindings]

        def wrap(x, y):
            return ffi.new('plaidml_binding*', [x.as_ptr(), y.as_ptr()])

        inputs = [wrap(x, y) for x, y in input_bindings]
        outputs = [wrap(x, y) for x, y in output_bindings]
        ffi_obj = ffi_call(
            lib.plaidml_compile,
            program.as_ptr(),
            device_id.encode(),
            target.encode(),
            len(inputs),
            inputs,
            len(outputs),
            outputs,
        )
        super(Executable, self).__init__(ffi_obj)

    def __call__(self, inputs):
        for buffer, ndarray in zip(self._inputs, inputs):
            buffer.copy_from_ndarray(ndarray)
        ffi_call(lib.plaidml_executable_run, self.as_ptr())
        return self._outputs


class _View(ForeignObject):
    __ffi_del__ = lib.plaidml_view_free

    def __init__(self, ffi_obj, shape):
        self.shape = shape
        super(_View, self).__init__(ffi_obj)

    @property
    def data(self):
        return ffi.buffer(ffi_call(lib.plaidml_view_data, self.as_ptr()), self.size)

    @property
    def size(self):
        return ffi_call(lib.plaidml_view_size, self.as_ptr())

    def writeback(self):
        ffi_call(lib.plaidml_view_writeback, self.as_ptr())

    def copy_from_ndarray(self, src):
        dst = np.frombuffer(self.data, dtype=self.shape.dtype.into_numpy())
        dst = dst.reshape(self.shape.sizes)
        np.copyto(dst, src)

    def copy_to_ndarray(self, dst):
        src = np.frombuffer(self.data, dtype=self.shape.dtype.into_numpy())
        src = src.reshape(self.shape.sizes)
        np.copyto(dst, src)


class Buffer(ForeignObject):
    __ffi_del__ = lib.plaidml_buffer_free

    def __init__(self, device_id, shape):
        self._shape = shape
        self._ndarray = None
        ffi_obj = ffi_call(lib.plaidml_buffer_alloc, device_id.encode(), shape.nbytes)
        super(Buffer, self).__init__(ffi_obj)

    @property
    def shape(self):
        return self._shape

    @contextlib.contextmanager
    def mmap_current(self):
        yield _View(ffi_call(lib.plaidml_buffer_mmap_current, self.as_ptr()), self.shape)

    @contextlib.contextmanager
    def mmap_discard(self):
        yield _View(ffi_call(lib.plaidml_buffer_mmap_discard, self.as_ptr()), self.shape)

    def as_ndarray(self):
        if self._ndarray is None:
            self._ndarray = np.ndarray(tuple(x for x in self.shape.sizes),
                                       dtype=self.shape.dtype.into_numpy())
        with self.mmap_current() as view:
            view.copy_to_ndarray(self._ndarray)
        return self._ndarray

    def copy_from_ndarray(self, ndarray):
        with self.mmap_discard() as view:
            view.copy_from_ndarray(ndarray)
            view.writeback()
