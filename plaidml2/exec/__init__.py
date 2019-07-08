# Copyright 2019 Intel Corporation.

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
