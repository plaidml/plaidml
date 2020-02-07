# Copyright 2019 Intel Corporation.

import logging

import numpy as np

import plaidml
import plaidml.edsl as edsl
import plaidml.settings
from plaidml.ffi import ForeignObject, decode_str, ffi, ffi_call, lib

logger = logging.getLogger(__name__)


def __init():
    """
    Initializes PlaidML's Execution API.
    """
    ffi_call(lib.plaidml_exec_init)


ffi.init_once(__init, 'plaidml_exec_init')


def list_devices():
    return plaidml.get_strs(lib.plaidml_devices_get)


class Executable(ForeignObject):
    """Docstring for class Executable"""
    __ffi_del__ = lib.plaidml_executable_free

    def __init__(self, program, inputs=[], outputs=[], device=None):
        if device is None:
            device = plaidml.settings.get('PLAIDML_DEVICE')

        def wrap(x, y):
            return ffi.new('plaidml_binding*', [x.as_ptr(), y.as_ptr()])

        inputs = [wrap(x, y) for x, y in inputs]
        outputs = [wrap(x, y) for x, y in outputs]
        ffi_obj = ffi_call(
            lib.plaidml_jit,
            program.as_ptr(),
            device.encode(),
            len(inputs),
            inputs,
            len(outputs),
            outputs,
        )
        super(Executable, self).__init__(ffi_obj)

    def run(self):
        self._methodcall(lib.plaidml_executable_run)


class Binder:
    """Docstring for class Binder"""

    def __init__(self, program, device=None):
        self.program = program
        if device is None:
            device = plaidml.settings.get('PLAIDML_DEVICE')
        self.device = device
        self.inputs = {arg.ref: arg.buffer for arg in program.inputs if arg.buffer}
        self.outputs = {arg.ref: arg.buffer for arg in program.outputs if arg.buffer}

    def input(self, tensor):
        if isinstance(tensor, edsl.Tensor):
            tensor = edsl.TensorRef(tensor)
        return self.inputs.get(tensor)

    def output(self, tensor):
        if isinstance(tensor, edsl.Tensor):
            tensor = edsl.TensorRef(tensor)
        return self.outputs.get(tensor)

    def set_input(self, tensor, buffer):
        if isinstance(tensor, edsl.Tensor):
            tensor = edsl.TensorRef(tensor)
        self.inputs[tensor] = buffer
        return self

    def set_output(self, tensor, buffer):
        if isinstance(tensor, edsl.Tensor):
            tensor = edsl.TensorRef(tensor)
        self.outputs[tensor] = buffer
        return self

    def compile(self):
        inputs = [(x.ref.tensor, self._get_buffer(self.inputs, x)) for x in self.program.inputs]
        outputs = [(x.ref.tensor, self._get_buffer(self.outputs, x)) for x in self.program.outputs]
        return Executable(self.program, inputs, outputs, device=self.device)

    def _get_buffer(self, map, arg):
        buffer = map.get(arg.ref)
        if buffer:
            return buffer
        buffer = plaidml.Buffer(arg.shape.into_TensorShape(), device=self.device)
        map[arg.ref] = buffer
        return buffer


def run(program, inputs, device=None):
    binder = Binder(program, device=device)
    executable = binder.compile()
    for tensor, data in inputs:
        buffer = binder.input(tensor)
        data = np.array(data, dtype=buffer.shape.dtype.into_numpy())
        buffer.copy_from_ndarray(data)
    executable.run()
    return [binder.output(x.ref).as_ndarray() for x in program.outputs]
