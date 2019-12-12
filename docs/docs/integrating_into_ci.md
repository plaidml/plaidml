# Integrating PlaidML into your project's CI infrastructure

# Instantiating PlaidML Programmatically

If you want to instantiate PlaidML from the command line, you can set the
following environment variables to select the proper configurations for your
device. This is equivalent to running `plaidml-setup` and selecting these
settings when prompted.

  * `PLAIDML_EXPERIMENTAL` - int (0 or 1) which determines whether to enable
     experimental mode in PlaidML 
  * `PLAIDML_DEVICE_IDS` - string which contains the name of the device to use
     with PlaidML (to see a list of devices, run `plaidml-setup`)

Below is an example of how to set the device configuration environment variables
for PlaidML.

```
export PLAIDML_EXPERIMENTAL=1
export PLAIDML_DEVICE_IDS=opencl_intel_uhd_graphics_630.0
```
