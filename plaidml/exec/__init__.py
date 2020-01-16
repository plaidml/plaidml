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
    Initializes PlaidML's execution engine
    """
    ffi_call(lib.plaidml_exec_init)


ffi.init_once(__init, 'plaidml_exec_init')


def list_devices():
    strs = ffi_call(lib.plaidml_devices_get)
    try:
        return [decode_str(strs[0].strs[i]) for i in range(strs.nstrs)]
    finally:
        ffi_call(lib.plaidml_strings_free, strs)


def list_targets():
    strs = ffi_call(lib.plaidml_targets_get)
    try:
        return [decode_str(strs[0].strs[i]) for i in range(strs.nstrs)]
    finally:
        ffi_call(lib.plaidml_strings_free, strs)


class Executable(ForeignObject):
    """Docstring for class Executable"""
    __ffi_del__ = lib.plaidml_executable_free

    def __init__(self, program, inputs=[], outputs=[], device=None, target=None):
        if device is None:
            device = plaidml.settings.get('PLAIDML_DEVICE')
        if target is None:
            target = plaidml.settings.get('PLAIDML_TARGET')

        def wrap(x, y):
            return ffi.new('plaidml_binding*', [x.as_ptr(), y.as_ptr()])

        inputs = [wrap(x, y) for x, y in inputs]
        outputs = [wrap(x, y) for x, y in outputs]
        ffi_obj = ffi_call(
            lib.plaidml_compile,
            program.as_ptr(),
            device.encode(),
            target.encode(),
            len(inputs),
            inputs,
            len(outputs),
            outputs,
        )
        super(Executable, self).__init__(ffi_obj)

    def run(self):
        ffi_call(lib.plaidml_executable_run, self.as_ptr())


class Binder:
    """Docstring for class Binder"""

    def __init__(self, program, device=None, target=None):
        self.program = program
        if device is None:
            device = plaidml.settings.get('PLAIDML_DEVICE')
        self.device = device
        if target is None:
            target = plaidml.settings.get('PLAIDML_TARGET')
        self.target = target
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
        return Executable(self.program, inputs, outputs, device=self.device, target=self.target)

    def _get_buffer(self, map, arg):
        buffer = map.get(arg.ref)
        if buffer:
            return buffer
        buffer = plaidml.Buffer(arg.shape.into_TensorShape(), device=self.device)
        map[arg.ref] = buffer
        return buffer


def run(program, inputs, device=None, target=None):
    binder = Binder(program, device=device, target=target)
    executable = binder.compile()
    for tensor, data in inputs:
        buffer = binder.input(tensor)
        data = np.array(data, dtype=buffer.shape.dtype.into_numpy())
        buffer.copy_from_ndarray(data)
    executable.run()
    return [binder.output(x.ref).as_ndarray() for x in program.outputs]
