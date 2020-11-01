# Copyright 2020 Intel Corporation

import atexit
import enum
import logging
from collections import namedtuple

import numpy as np

from plaidml.core._version import PLAIDML_VERSION
from plaidml.ffi import (Error, ForeignObject, decode_list, decode_str, ffi, ffi_call, lib)

logger = logging.getLogger(__name__)


def __init():
    """
    Initializes the PlaidML Core API.
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


def get_strings(ffi_list, *args):
    return decode_list(ffi_list, lib.plaidml_strings_free, decode_str, *args)


def get_integers(ffi_list, *args):
    return decode_list(ffi_list, lib.plaidml_integers_free, lambda x: x, *args)


def get_shapes(ffi_list, *args):

    def decode_shape(ptr):
        return TensorShape(ptr=ptr)

    return decode_list(ffi_list, lib.plaidml_shapes_free, decode_shape, *args)


def kvps_to_dict(kvps):
    try:
        x = kvps.elts
        return {decode_str(x[i].key): decode_str(x[i].value) for i in range(kvps.size)}
    finally:
        ffi_call(lib.plaidml_kvps_free, kvps)


def kvps_to_list(kvps):
    try:
        x = kvps.elts
        return [(decode_str(x[i].key), decode_str(x[i].value)) for i in range(kvps.size)]
    finally:
        ffi_call(lib.plaidml_kvps_free, kvps)


def list_targets():
    return get_strings(lib.plaidml_targets_get)


class DType(enum.IntEnum):
    """Defines the set of supported element types in a Tensor."""

    INVALID = 0
    """An invalid data type"""

    INTX = 1
    """An integer data type with arbitrary precision"""

    UINTX = 2
    """An unsigned integer data type with arbitrary precision"""

    FLOATX = 3
    """A floating point data type with arbitrary precision"""

    BOOLEAN = 4
    """A boolean data type"""

    INT8 = 5
    """An 8-bit signed integer data type"""

    UINT8 = 6
    """An 8-bit unsigned integer data type"""

    INT16 = 7
    """A 16-bit signed integer data type"""

    UINT16 = 8
    """A 16-bit unsigned integer data type"""

    INT32 = 9
    """A 32-bit signed integer data type"""

    UINT32 = 10
    """A 32-bit unsigned integer data type"""

    INT64 = 11
    """A 64-bit signed integer data type"""

    UINT64 = 12
    """A 64-bit unsigned integer data type"""

    BFLOAT16 = 13
    """A 16-bit blocked floating point data type"""

    FLOAT16 = 14
    """A 16-bit floating point data type"""

    FLOAT32 = 15
    """A 32-bit floating point data type"""

    FLOAT64 = 16
    """A 64-bit floating point data type"""

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
    'floatx': DType.FLOATX,
    'float16': DType.FLOAT16,
    'float32': DType.FLOAT32,
    'float64': DType.FLOAT64,
    'intx': DType.INTX,
    'int8': DType.INT8,
    'int16': DType.INT16,
    'int32': DType.INT32,
    'int64': DType.INT64,
    'uintx': DType.UINTX,
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
    DType.INTX: DTypeInfo(base='int', width=0),
    DType.INT8: DTypeInfo(base='int', width=1),
    DType.INT16: DTypeInfo(base='int', width=2),
    DType.INT32: DTypeInfo(base='int', width=4),
    DType.INT64: DTypeInfo(base='int', width=8),
    DType.UINTX: DTypeInfo(base='uint', width=0),
    DType.UINT8: DTypeInfo(base='uint', width=1),
    DType.UINT16: DTypeInfo(base='uint', width=2),
    DType.UINT32: DTypeInfo(base='uint', width=4),
    DType.UINT64: DTypeInfo(base='uint', width=8),
    DType.FLOATX: DTypeInfo(base='float', width=0),
    DType.FLOAT16: DTypeInfo(base='float', width=2),
    DType.FLOAT32: DTypeInfo(base='float', width=4),
    DType.FLOAT64: DTypeInfo(base='float', width=8),
    DType.BFLOAT16: DTypeInfo(base='float', width=2),
}


class TensorShape(ForeignObject):
    __ffi_del__ = lib.plaidml_shape_free
    __ffi_repr__ = lib.plaidml_shape_repr

    def __init__(self, dtype=None, sizes=[], strides=[], ptr=None):
        if ptr:
            ffi_obj = ptr
        elif dtype is not None:
            raw_sizes = ffi.new('int64_t[]', [0 if x is None else x for x in sizes])
            if strides:
                raw_strides = ffi.new('int64_t[]', strides)
            else:
                raw_strides = ffi.NULL
            ffi_obj = ffi_call(lib.plaidml_shape_alloc, dtype, len(sizes), raw_sizes, raw_strides)
        else:
            raise ValueError('One of dtype= or ptr= must be specified.')
        super(TensorShape, self).__init__(ffi_obj)

    @property
    def dtype(self):
        return DType(self._methodcall(lib.plaidml_shape_get_dtype))

    @property
    def rank(self):
        return self._methodcall(lib.plaidml_shape_get_rank)

    @property
    def byte_size(self):
        return self._methodcall(lib.plaidml_shape_get_nbytes)

    @property
    def sizes(self):
        return get_integers(lib.plaidml_shape_get_sizes, self.as_ptr())

    @property
    def strides(self):
        return get_integers(lib.plaidml_shape_get_strides, self.as_ptr())


class Buffer(ForeignObject):
    __ffi_del__ = lib.plaidml_buffer_free

    def __init__(self, shape=None, ptr=None, data=None):
        self._ndarray = None
        if data is not None:
            self.__data = data
            cdata = ffi.from_buffer(data)
            ffi_obj = ffi_call(lib.plaidml_buffer_adopt, shape.as_ptr(), cdata, len(cdata))
        elif ptr:
            ffi_obj = ptr
        else:
            ffi_obj = ffi_call(lib.plaidml_buffer_alloc, shape.as_ptr())
        super(Buffer, self).__init__(ffi_obj)

    def clone(self):
        return Buffer(ptr=self._methodcall(lib.plaidml_buffer_clone))

    @property
    def shape(self):
        return TensorShape(ptr=self._methodcall(lib.plaidml_buffer_shape))

    @property
    def data(self):
        return ffi.buffer(self._methodcall(lib.plaidml_buffer_data), self.size)

    @property
    def size(self):
        return self._methodcall(lib.plaidml_buffer_size)

    def as_ndarray(self):
        if self._ndarray is None:
            shape = self.shape
            self._ndarray = np.ndarray(tuple(x for x in shape.sizes),
                                       dtype=shape.dtype.into_numpy())
        self.copy_into_ndarray(self._ndarray)
        return self._ndarray

    def copy_into_ndarray(self, dst):
        src = np.frombuffer(self.data, dtype=self.shape.dtype.into_numpy())
        src = src.reshape(self.shape.sizes)
        np.copyto(dst, src)

    def copy_from_ndarray(self, src):
        dst = np.frombuffer(self.data, dtype=self.shape.dtype.into_numpy())
        dst = dst.reshape(self.shape.sizes)
        np.copyto(dst, src, casting='unsafe')


class Program(ForeignObject):
    __ffi_del__ = lib.plaidml_program_free
    __ffi_repr__ = lib.plaidml_program_repr

    def __init__(self, name, inputs, outputs, shapes=None):
        # logger.debug('Program({}, {}, {}, {})'.format(name, inputs, outputs, shapes))
        raw_inputs = [x.as_ptr() for x in inputs]
        raw_outputs = [x.as_ptr() for x in outputs]
        if shapes:
            raw_shapes = [x.as_ptr() for x in shapes]
        else:
            raw_shapes = ffi.NULL
        ffi_obj = ffi_call(
            lib.plaidml_build,
            name.encode(),
            len(raw_inputs),
            raw_inputs,
            raw_shapes,
            len(raw_outputs),
            raw_outputs,
        )
        super(Program, self).__init__(ffi_obj)

    def compile(self, target='', debug=False):
        self._methodcall(lib.plaidml_program_compile, debug, target.encode())

    def save(self):
        return Buffer(ptr=self._methodcall(lib.plaidml_program_save))

    @property
    def inputs(self):
        return get_shapes(lib.plaidml_program_get_inputs, self.as_ptr())

    @property
    def outputs(self):
        return get_shapes(lib.plaidml_program_get_outputs, self.as_ptr())

    @property
    def passes(self):
        """Returns a list of passes.

        Each pass in the list is a tuple of ``(name, ir)``, where ``ir`` means
        `intermediate representation`.

        Note that ``debug`` must be enabled when compiling the program.

        Returns:
            :obj:`list` of :obj:`tuple` of :obj:`str`: The passes.

        """
        return kvps_to_list(self._methodcall(lib.plaidml_program_get_passes))
