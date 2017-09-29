# PlaidML
![The PlaidML Platypus](https://github.com/vertexai/plaidml/raw/master/images/plaid-final.png)
*A framework for making machine learning work everywhere.*

PlaidML is a multi-language acceleration framework that: 
  
  * Enables practioners to deploy high-performance neural nets on any device
  * Allows hardware developers to quickly integrate with high-level frameworks
  * Allows framework developers to easily add support for many kinds of hardware

PlaidML is under active development and should be thought of as early-beta quality.

- [Current Limitations](#current-limitation)
- [Supported Hardware](#supported-hardware)
  - [Validated Hardware](#validated-hardware)
  - [Experimental Config](#experimental-config)
- [Validated Networks](#validated-networks)
- [Installation Instructions](#installation-instructions)
- [Reporting Issues](#reporting-issues)

## Current Limitations

This version of PlaidML has some notable limitations which will be addressed soon in upcoming releases:

  * Initial compile times are long, especially for training
  * AMD training performance is unacceptable
  * RNN support is not implemented
  * The Keras backend is not complete

## Supported Hardware

PlaidML currently supports any devices that provides the full profile of OpenCL 1.1 or higher. PlaidML ships with
configuration files and parameters only for devices that we've verified or are confident have an acceptable level of 
performance.

### Validated Hardware
*These cards are tested for correctness and performance on every commit*
  * AMD
    * R9 Nano
    * RX480
  * NVidia
    * K80, GTX780
    * GTX1070

### Experimental Config
If the device you're testing against isn't supported by the default configuration, you can enable the experimental
configuration.

If your device isn't supported, PlaidML will error with this message:
```
ERROR: No devices found, set PLAIDML_EXPERIMENTAL=1 to enable broader device support
```
As noted, set the `PLAIDML_EXPERIMENTAL=1`. This will use a the experimental config from the python package. If your
device still isn't supported, contact [plaidml-dev](https://groups.google.com/forum/#!forum/plaidml-dev) with the output
of `clinfo`. Alternatively, users can attempt to add support themselves by editing the configuration file listed when
PlaidML starts.

If you have multiple supported devices, you'll need to choose between them by setting `PLAIDML_DEVICE_IDS=<id>`. PlaidML
will emit a helpful error message in this case, like:
```
ERROR: Multiple Devices found, set PLAIDML_DEVICE_IDS=<devid> to one of:
  fiji.0
  intel(r)_xeon(r)_cpu_e5-2670_0_@_2.60ghz.0
```

### Validated Networks

We support most of the convolutional Keras application networks. Validated networks are tested for performance and 
correctness as part of our continuous integration system.

 * CNNs
   * inception_v3
   * resnet50
   * vggg*
   * xception
   * mobilenet

## Installation Instructions

### Ubuntu Linux
If necessary, install python's 'pip' tool.
```
sudo add-apt-repository universe && sudo apt update
sudo apt install python-pip
```
Make sure your system has OpenCL.
```
sudo apt install clinfo
clinfo
```
If clinfo reports "Number of platforms" == 0, you must install a driver.

If you have an Nvidia graphics card:
```
sudo add-apt-repository ppa:graphics-drivers/ppa && sudo apt update
sudo apt install nvidia-modprobe nvidia-384 nvidia-opencl-icd-384 libcuda1-384
```
If you have an AMD card, [download the AMDGPU PRO driver and install](http://support.amd.com/en-us/kb-articles/Pages/AMDGPU-PRO-Driver-for-Linux-Release-Notes.aspx) according to AMD's instructions.

Install plaidml system-wide:
```
sudo ./install.sh
```

Install plaidml inside a virtualenv:
```
./install.sh
```

You can test your installation by running mobilenet in plaidbench:
```
cd plaidbench
python plaidbench.py mobilenet
```

You can adapt any Keras code by using the PlaidML backend instead of the TensorFlow, CTNK, or Theano backend that you 
normally use.

Simply insert this code **BEFORE you `import keras`**:
```
# Install the plaidml backend
import plaidml.keras
plaidml.keras.install_backend()
```

### Plaidvision and Plaidbench

We've developed two open source projects: 

  * [plaidvision](https://github.com/vertexai/plaidvision) provides a simple shell for developing vision applications using your webcam
  * [plaidbench](https://github.com/vertexai/plaidbench) is a performance testing suite designed to help users compare the performance
  of different cards and different frameworks
  
**PRE-RELEASE USERS: plaidvision and plaidbench are included in the release tarball.**


### Hello VGG
One of the great things about keras is how easy it is to play with state-of-the-art networks. Here's all the code you
need to run VGG19:
```
#!/usr/bin/env python
import numpy as np
import time

# Install the plaidml backend
import plaidml.keras
plaidml.keras.install_backend()

import keras
import keras.applications as kapp
from keras.datasets import cifar10

(x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
batch_size = 8
x_train = x_train[:batch_size]
x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)
model = kapp.VGG19()
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Running initial batch (compiling tile program)")
y = model.predict(x=x_train, batch_size=batch_size)

# Now start the clock and run 10 batches
print("Timing inference...")
start = time.time()
for i in range(10):
    y = model.predict(x=x_train, batch_size=batch_size)
print("Ran in {} seconds".format(time.time() - start))

```

## Reporting Issues
Contact [plaidml-dev](https://groups.google.com/forum/#!forum/plaidml-dev) 