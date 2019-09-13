# Copyright 2019 Intel Corporation.

import numpy as np

import plaidml2 as plaidml
import plaidml2.settings as plaidml_settings
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

    def __init__(self, program, inputs, device=None, target=None):
        if device is None:
            device = plaidml_settings.get('PLAIDML_DEVICE')
        if target is None:
            target = plaidml_settings.get('PLAIDML_TARGET')

        def make_buffer(tensor):
            # convert LogicalShape into TensorShape
            return plaidml.Buffer(device, tensor.shape.into_TensorShape())

        self._input_bindings = [(x, make_buffer(x)) for x in inputs]
        self._output_bindings = [(x, make_buffer(x)) for x in program.outputs]

        self._inputs = [x[1] for x in self._input_bindings]
        self._outputs = [x[1] for x in self._output_bindings]

        def wrap(x, y):
            return ffi.new('plaidml_binding*', [x.as_ptr(), y.as_ptr()])

        inputs = [wrap(x, y) for x, y in self._input_bindings]
        outputs = [wrap(x, y) for x, y in self._output_bindings]
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

    def __call__(self, inputs):
        for buffer, ndarray in zip(self._inputs, inputs):
            # Cast the input data type to match the dtype expected by the placeholder buffer
            ndarray = np.array(ndarray, dtype=buffer.shape.dtype.into_numpy())
            buffer.copy_from_ndarray(ndarray)
        ffi_call(lib.plaidml_executable_run, self.as_ptr())
        return self._outputs


def run(program, inputs, device=None, target=None):
    exe = Executable(program, [x for x, y in inputs], device=device, target=target)
    return [x.as_ndarray() for x in exe([y for x, y in inputs])]
