#!/usr/bin/env python
"""Creates a plaidml user configuration file."""

import sys

import plaidml
import plaidml.exceptions
import plaidml.settings

def main():

    ctx = plaidml.Context()
    plaidml.quiet()

    def choice_prompt(question, choices, default):
        input = ""
        while not input in choices:
            input = raw_input("{0}? ({1})[{2}]:".format(question, ",".join(choices), default))
            if not input:
                input = default
            elif input not in choices:
                print("Invalid choice: {}".format(input))
        return input
    print("""
PlaidML Setup ({0})

Thanks for using PlaidML!

Some Notes:
  * Bugs and other issues: https://github.com/plaidml/plaidml
  * Questions: https://stackoverflow.com/questions/tagged/plaidml
  * Say hello: https://groups.google.com/forum/#!forum/plaidml-dev
  * PlaidML is licensed under the GNU AGPLv3
 """).format(plaidml.__version__)

    # Operate as if nothing is set
    plaidml.settings._setup_for_test(plaidml.settings.user_settings)

    try:
        plaidml.settings.experimental = False
        devices = plaidml.devices(ctx, limit=100)
    except plaidml.exceptions.PlaidMLError:
        devices = []
    try:
        plaidml.settings.experimental = True
        exp_devices = plaidml.devices(ctx, limit=100)
    except plaidml.exceptions.PlaidMLError:
        exp_devices = []

    if len(devices) == 0 and len(exp_devices) == 0:
        print(
"""
No devices found. Please run 'clinfo' to ensure an OpenCL device is present.
If a device is present, open an issue and include the full output of clinfo.
""")
        sys.exit(-1)

    print("Default Config Devices:")
    for dev in devices:
        print("   {0} : {1}".format(dev.id, dev.description))

    print("\nExperimental Config Devices:")
    for dev in exp_devices:
        print("   {0} : {1}".format(dev.id, dev.description))

    exp = choice_prompt("\nEnable experimental device support", ["y","n"], "n")
    plaidml.settings.experimental = exp == "y"
    devices = plaidml.devices(ctx, limit=100)

    if len(devices) > 1:
        print(
"""
Multiple devices detected (You can override by setting PLAIDML_DEVICE_IDS).
Please choose a default device:
""")
        devrange = range(1, len(devices) + 1)
        for i in devrange:
            print("   {0} : {1}".format(i, devices[i - 1].id))
        dev = choice_prompt("\nDefault device", [str(i) for i in devrange], "1")
        plaidml.settings.device_ids = [devices[int(dev) - 1].id]

    print(
"""
PlaidML sends anonymous usage statistics to help guide improvements.
We'd love your help making it better.
""")

    tel = choice_prompt("Enable telemetry reporting", ["y","n"], "y")
    plaidml.settings.telemetry = tel == "y"

    print("\nAlmost done. Multiplying some matrices...")
    # Reinitialize to send a usage report
    print("Tile code:")
    print("  function (B[X,Z], C[Z,Y]) -> (A) { A[x,y : X,Y] = +(B[x,z] * C[z,y]); }")
    with plaidml.open_first_device(ctx) as dev:
        matmul = plaidml.Function("function (B[X,Z], C[Z,Y]) -> (A) { A[x,y : X,Y] = +(B[x,z] * C[z,y]); }")
        shape = plaidml.Shape(ctx, plaidml.DATA_FLOAT32, 3, 3)
        a = plaidml.Tensor(dev, shape)
        b = plaidml.Tensor(dev, shape)
        c = plaidml.Tensor(dev, shape)
        plaidml.run(ctx, matmul, inputs={"B": b, "C": c}, outputs={"A": a})
    print("Whew. That worked.\n")

    sav = choice_prompt("Save settings to {0}".format(plaidml.settings.user_settings), ["y","n"], "y")
    if sav == "y":
        plaidml.settings.save(plaidml.settings.user_settings)
    print("Success!\n")

if __name__ == "__main__":
    main()