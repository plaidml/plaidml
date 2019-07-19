#!/usr/bin/env python
"""Creates a plaidml user configuration file."""
from __future__ import print_function

import os
import sys

from six.moves import input

import numpy as np
import plaidml2 as plaidml
import plaidml2.edsl as edsl
import plaidml2.exec as plaidml_exec
import plaidml2.settings as plaidml_settings


def choice_prompt(question, choices, default):
    inp = ""
    while not inp in choices:
        inp = input("{0}? ({1})[{2}]:".format(question, ",".join(choices), default))
        if not inp:
            inp = default
        elif inp not in choices:
            print("Invalid choice: {}".format(inp))
    return inp


def main():
    print("""
PlaidML Setup ({0})

Thanks for using PlaidML!

Some Notes:
  * Bugs and other issues: https://github.com/plaidml/plaidml
  * Questions: https://stackoverflow.com/questions/tagged/plaidml
  * Say hello: https://groups.google.com/forum/#!forum/plaidml-dev
  * PlaidML is licensed under the Apache License 2.0
 """.format(plaidml.__version__))

    devices = sorted(plaidml_exec.list_devices())
    targets = sorted(plaidml_exec.list_targets())

    if not devices:
        print("""
No OpenCL devices found. Check driver installation.
Read the helpful, easy driver installation instructions from our README:
http://github.com/plaidml/plaidml
""")
        sys.exit(-1)

    dev_idx = 0
    if len(devices) > 1:
        print("""
Multiple devices detected (You can override by setting PLAIDML_DEVICE).
Please choose a default device:
""")
        for i, device in enumerate(devices):
            print("   {} : {}".format(i + 1, device))
        choices = [str(i + 1) for i in range(len(devices))]
        dev_idx = int(choice_prompt("\nDefault device", choices, "1"))
    plaidml_settings.set('PLAIDML_DEVICE', devices[dev_idx - 1])
    device = plaidml_settings.get('PLAIDML_DEVICE')

    print()
    print("Selected device:")
    print("    {}".format(device))

    print()
    print("A target determines the compiler configuration and should be matched with your device.")
    print("Please choose a default target:")
    for i, target in enumerate(targets):
        print("   {} : {}".format(i + 1, target))
    choices = [str(i + 1) for i in range(len(targets))]
    tgt_idx = int(choice_prompt("\nDefault target", choices, "1"))
    plaidml_settings.set('PLAIDML_TARGET', targets[tgt_idx - 1])
    target = plaidml_settings.get('PLAIDML_TARGET')

    print()
    print("Selected target:")
    print("    {}".format(target))

    print()
    print("Almost done. Multiplying some matrices...")
    print("Tile code:")
    print("  function (B[X, Z], C[Z, Y]) -> (A) { A[x, y : X, Y] = +(B[x, z] * C[z, y]); }")

    shape = edsl.LogicalShape(plaidml.DType.FLOAT32, [3, 3])
    B = edsl.Tensor(shape)
    C = edsl.Tensor(shape)

    X, Y, Z = edsl.TensorDims(3)
    x, y, z = edsl.TensorIndexes(3)
    B.bind_dims(X, Z)
    C.bind_dims(Z, Y)
    A = edsl.TensorOutput(X, Y)
    A[x, y] += B[x, z] * C[z, y]

    program = edsl.Program('plaidml_setup', [A])

    def run(program, inputs):

        def make_buffer(tensor):
            # convert LogicalShape into TensorShape
            shape = plaidml.TensorShape(tensor.shape.dtype, tensor.shape.int_dims)
            return plaidml.Buffer(device, shape)

        ibindings = [(x, make_buffer(x)) for x, y in inputs]
        obindings = [(x, make_buffer(x)) for x in program.outputs]

        exe = plaidml_exec.Executable(program, device, target, ibindings, obindings)
        return [x.as_ndarray() for x in exe([y for x, y in inputs])]

    run(program, [(B, np.random.rand(3, 3)), (C, np.random.rand(3, 3))])
    print("Whew. That worked.")
    print()

    settings_path = plaidml_settings.get('PLAIDML_SETTINGS')
    save = choice_prompt("Save settings to {0}".format(settings_path), ["y", "n"], "y")
    if save == "y":
        plaidml_settings.save()
    print("Success!")
    print()


if __name__ == "__main__":
    main()
