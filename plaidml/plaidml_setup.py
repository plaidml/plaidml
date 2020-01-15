#!/usr/bin/env python
"""Creates a plaidml user configuration file."""
from __future__ import print_function

import os
import sys

from six.moves import input

import plaidml
import plaidml.exceptions
import plaidml.settings


def main():
    ctx = plaidml.Context()
    plaidml.quiet()

    def choice_prompt(question, choices, default):
        inp = ""
        while not inp in choices:
            inp = input("{0}? ({1})[{2}]:".format(question, ",".join(choices), default))
            if not inp:
                inp = default
            elif inp not in choices:
                print("Invalid choice: {}".format(inp))
        return inp

    print("""
PlaidML Setup ({0})

Thanks for using PlaidML!

The feedback we have received from our users indicates an ever-increasing need
for performance, programmability, and portability. During the past few months,
we have been restructuring PlaidML to address those needs.  To make all the
changes we need to make while supporting our current user base, all development
of PlaidML has moved to a branch — plaidml-v1. We will continue to maintain and
support the master branch of PlaidML and the stable 0.7.0 release.

Read more here: https://github.com/plaidml/plaidml 

Some Notes:
  * Bugs and other issues: https://github.com/plaidml/plaidml/issues
  * Questions: https://stackoverflow.com/questions/tagged/plaidml
  * Say hello: https://groups.google.com/forum/#!forum/plaidml-dev
  * PlaidML is licensed under the Apache License 2.0
 """.format(plaidml.__version__))

    # Placeholder env var
    if os.getenv("PLAIDML_VERBOSE"):
        # change verbose settings to PLAIDML_VERBOSE, or 4 if PLAIDML_VERBOSE is invalid
        try:
            arg_verbose = int(os.getenv("PLAIDML_VERBOSE"))
        except ValueError:
            arg_verbose = 4
        plaidml._internal_set_vlog(arg_verbose)
        print("INFO:Verbose logging has been enabled - verbose level", arg_verbose, "\n")
        if plaidml.settings.default_config:
            (cfg_path, cfg_file) = os.path.split(plaidml.settings.default_config)
        else:
            (cfg_path, cfg_file) = ("Unknown", "Unknown")
        if plaidml.settings.experimental_config:
            (exp_path, exp_file) = os.path.split(plaidml.settings.experimental_config)
        else:
            (exp_path, exp_file) = ("Unknown", "Unknown")

    # Operate as if nothing is set
    plaidml.settings._setup_for_test(plaidml.settings.user_settings)

    plaidml.settings.experimental = False
    devices, _ = plaidml.devices(ctx, limit=100, return_all=True)
    plaidml.settings.experimental = True
    exp_devices, unmatched = plaidml.devices(ctx, limit=100, return_all=True)

    if not (devices or exp_devices):
        if not unmatched:
            print("""
No OpenCL devices found. Check driver installation.
Read the helpful, easy driver installation instructions from our README:
http://github.com/plaidml/plaidml
""")
        else:
            print("""
No supported devices found. Run 'clinfo' and file an issue containing the full output.
""")
        sys.exit(-1)

    if devices and os.getenv("PLAIDML_VERBOSE"):
        print("Default Config File Location:")
        print("   {0}/".format(cfg_path))

    print("\nDefault Config Devices:")
    if not devices:
        print("   No devices.")
    for dev in devices:
        print("   {0} : {1}".format(dev.id.decode(), dev.description.decode()))

    if exp_devices and os.getenv("PLAIDML_VERBOSE"):
        print("\nExperimental Config File Location:")
        print("   {0}/".format(exp_path))

    print("\nExperimental Config Devices:")
    if not exp_devices:
        print("   No devices.")
    for dev in exp_devices:
        print("   {0} : {1}".format(dev.id.decode(), dev.description.decode()))

    print(
        "\nUsing experimental devices can cause poor performance, crashes, and other nastiness.\n")
    exp = choice_prompt("Enable experimental device support", ["y", "n"], "n")
    plaidml.settings.experimental = exp == "y"
    try:
        devices = plaidml.devices(ctx, limit=100)
    except plaidml.exceptions.PlaidMLError:
        print("\nNo devices available in chosen config. Rerun plaidml-setup.")
        sys.exit(-1)

    if devices:
        dev = 1
        if len(devices) > 1:
            print("""
Multiple devices detected (You can override by setting PLAIDML_DEVICE_IDS).
Please choose a default device:
""")
            devrange = range(1, len(devices) + 1)
            for i in devrange:
                print("   {0} : {1}".format(i, devices[i - 1].id.decode()))
            dev = choice_prompt("\nDefault device", [str(i) for i in devrange], "1")
        plaidml.settings.device_ids = [devices[int(dev) - 1].id.decode()]

    print("\nSelected device:\n    {0}".format(plaidml.devices(ctx)[0]))

    print("\nAlmost done. Multiplying some matrices...")
    # Reinitialize to send a usage report
    print("Tile code:")
    print("  function (B[X,Z], C[Z,Y]) -> (A) { A[x,y : X,Y] = +(B[x,z] * C[z,y]); }")
    with plaidml.open_first_device(ctx) as dev:
        matmul = plaidml.Function(
            "function (B[X,Z], C[Z,Y]) -> (A) { A[x,y : X,Y] = +(B[x,z] * C[z,y]); }")
        shape = plaidml.Shape(ctx, plaidml.DType.FLOAT32, 3, 3)
        a = plaidml.Tensor(dev, shape)
        b = plaidml.Tensor(dev, shape)
        c = plaidml.Tensor(dev, shape)
        plaidml.run(ctx, matmul, inputs={"B": b, "C": c}, outputs={"A": a})
    print("Whew. That worked.\n")

    sav = choice_prompt("Save settings to {0}".format(plaidml.settings.user_settings), ["y", "n"],
                        "y")
    if sav == "y":
        plaidml.settings.save(plaidml.settings.user_settings)
    print("Success!\n")


if __name__ == "__main__":
    main()
