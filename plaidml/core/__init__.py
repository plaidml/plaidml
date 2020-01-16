# Copyright 2019 Intel Corporation.

import atexit
import contextlib
import enum
from collections import namedtuple

import numpy as np
from plaidml.core._version import PLAIDML_VERSION
from plaidml.ffi import Error, ForeignObject, ffi, ffi_call, lib


def __init():
    """
    Initializes PlaidML's Core API.
    """
    ffi_call(lib.plaidml_init)
    lib_version = ffi.string(ffi_call(lib.plaidml_version)).decode()
    if lib_version != PLAIDML_VERSION:
        raise EnvironmentError('Version mismatch. plaidml (python): {}, {} (C++): {}'.format(
            PLAIDML_VERSION, lib.lib_name, lib_version))
    return PLAIDML_VERSION


__version__ = ffi.init_once(__init, 'plaidml_init')


@atexit.register
def __shutdown():
    ffi_call(lib.plaidml_shutdown)


class DType(enum.IntEnum):
    """Describes the type of a tensor element."""
    INVALID = 0
    BOOLEAN = 2
    INT8 = 0x10
    INT16 = 0x11
    INT32 = 0x12
    INT64 = 0x13
    INT128 = 0x14
    UINT8 = 0x20
    UINT16 = 0x21
    UINT32 = 0x22
    UINT64 = 0x23
    FLOAT16 = 0x31
    FLOAT32 = 0x32
    FLOAT64 = 0x33
    BFLOAT16 = 0x38
    PRNG = 0x40

    def into_numpy(self):
        try:
            return PLAIDML_DTYPE_TO_NUMPY[self]
        except KeyError:
            raise ValueError("Unrecognized PlaidML dtype: {}".format(self))

    @property
    def info(self):
        try:
            return DTYPE_INFOS[self]
        except KeyError:
            raise ValueError("Unrecognized PlaidML dtype: {}".format(self))

    @staticmethod
    def from_numpy(dtype):
        if isinstance(dtype, np.dtype):
            dtype = str(dtype)
        if isinstance(dtype, DType):
            return dtype
        try:
            return NUMPY_DTYPE_TO_PLAIDML[dtype]
        except KeyError:
            raise ValueError("Unrecognized Numpy dtype {}".format(dtype))


NUMPY_DTYPE_TO_PLAIDML = {
    'bool': DType.BOOLEAN,
    'float16': DType.FLOAT16,
    'float32': DType.FLOAT32,
    'float64': DType.FLOAT64,
    'int8': DType.INT8,
    'int16': DType.INT16,
    'int32': DType.INT32,
    'int64': DType.INT64,
    'uint8': DType.UINT8,
    'uint16': DType.UINT16,
    'uint32': DType.UINT32,
    'uint64': DType.UINT64,
    'bfloat16': DType.BFLOAT16,
}

PLAIDML_DTYPE_TO_NUMPY = {v: k for k, v in NUMPY_DTYPE_TO_PLAIDML.items()}


class DTypeInfo(namedtuple('DTypeInfo', ['base', 'width'])):
    """Describes a PlaidML datatype."""
    __slots__ = ()

    @property
    def bitwidth(self):
        """The number of bits occupied by an instance of the type."""
        if self.base == 'bool':
            return 1
        return self.width * 8


DTYPE_INFOS = {
    DType.BOOLEAN: DTypeInfo(base='bool', width=1),
    DType.INT8: DTypeInfo(base='int', width=1),
    DType.INT16: DTypeInfo(base='int', width=2),
    DType.INT32: DTypeInfo(base='int', width=4),
    DType.INT64: DTypeInfo(base='int', width=8),
    DType.UINT8: DTypeInfo(base='uint', width=1),
    DType.UINT16: DTypeInfo(base='uint', width=2),
    DType.UINT32: DTypeInfo(base='uint', width=4),
    DType.UINT64: DTypeInfo(base='uint', width=8),
    DType.FLOAT16: DTypeInfo(base='float', width=2),
    DType.FLOAT32: DTypeInfo(base='float', width=4),
    DType.FLOAT64: DTypeInfo(base='float', width=8),
    DType.BFLOAT16: DTypeInfo(base='float', width=2),
}


class TensorShape(ForeignObject):
    """Docstring for class TensorShape"""
    __ffi_del__ = lib.plaidml_shape_free
    __ffi_repr__ = lib.plaidml_shape_repr

    def __init__(self, dtype=None, sizes=[], strides=None, ptr=None):
        if ptr:
            ffi_obj = ptr
        elif dtype is not None:
            if strides is None:
                strides = []
            if len(strides) != len(sizes):
                stride = 1
                for i in range(len(sizes) - 1, -1, -1):
                    strides.insert(0, stride)
                    stride *= sizes[i]
            raw_sizes = ffi.new('int64_t[]', sizes)
            raw_strides = ffi.new('int64_t[]', strides)
            ffi_obj = ffi_call(lib.plaidml_shape_alloc, dtype, len(sizes), raw_sizes, raw_strides)
        else:
            raise ValueError('One of dtype= or ptr= must be specified.')
        super(TensorShape, self).__init__(ffi_obj)

    @property
    def dtype(self):
        return DType(ffi_call(lib.plaidml_shape_get_dtype, self.as_ptr()))

    @property
    def ndims(self):
        return ffi_call(lib.plaidml_shape_get_ndims, self.as_ptr())

    @property
    def nbytes(self):
        return ffi_call(lib.plaidml_shape_get_nbytes, self.as_ptr())

    @property
    def sizes(self):
        return [
            ffi_call(lib.plaidml_shape_get_dim_size, self.as_ptr(), i) for i in range(self.ndims)
        ]

    @property
    def strides(self):
        return [
            ffi_call(lib.plaidml_shape_get_dim_stride, self.as_ptr(), i) for i in range(self.ndims)
        ]


class _View(ForeignObject):
    """Docstring for class _View"""
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
    """Docstring for class Buffer"""
    __ffi_del__ = lib.plaidml_buffer_free

    def __init__(self, shape, device=None, ptr=None):
        self._shape = shape
        self._ndarray = None
        if ptr:
            ffi_obj = ptr
        elif device:
            ffi_obj = ffi_call(lib.plaidml_buffer_alloc, device.encode(), shape.nbytes)
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
