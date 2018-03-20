#!/usr/bin/env python
"""Creates a plaidml user configuration file."""
from __future__ import print_function

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

Some Notes:
  * Bugs and other issues: https://github.com/plaidml/plaidml
  * Questions: https://stackoverflow.com/questions/tagged/plaidml
  * Say hello: https://groups.google.com/forum/#!forum/plaidml-dev
  * PlaidML is licensed under the GNU AGPLv3
 """.format(plaidml.__version__))

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

    print("Default Config Devices:")
    if not devices:
        print("   No devices.")
    for dev in devices:
        print("   {0} : {1}".format(dev.id.decode(), dev.description.decode()))

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
    print("""
PlaidML sends anonymous usage statistics to help guide improvements.
We'd love your help making it better.
""")

    tel = choice_prompt("Enable telemetry reporting", ["y", "n"], "y")
    plaidml.settings.telemetry = tel == "y"

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
