Configuration
#############
If you want to use PlaidML from the command line, you can set the
following environment variables to select the proper configurations for your
device. This is equivalent to running `plaidml-setup` and selecting these
settings when prompted.
  * `PLAIDML_EXPERIMENTAL` - (0 or 1) determines whether to enable experimental mode in PlaidML 
  * `PLAIDML_DEVICE_IDS` - (string) the name of the device to use with PlaidML (to see a list of devices, run ``plaidml\-setup``)
Below is an example of how to set the device configuration environment variables
for PlaidML.
.. code-block::
  export PLAIDML_EXPERIMENTAL=1
  export PLAIDML_DEVICE_IDS=opencl_intel_uhd_graphics_630.0

