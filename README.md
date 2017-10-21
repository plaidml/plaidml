# PlaidML
![The PlaidML Platypus](https://github.com/vertexai/plaidml/raw/master/images/plaid-final.png)
*A framework for making deep learning work everywhere.*

PlaidML is a multi-language acceleration framework that: 
  
  * Enables practitioners to deploy high-performance neural nets on any device
  * Allows hardware developers to quickly integrate with high-level frameworks
  * Allows framework developers to easily add support for many kinds of hardware

For background and early benchmarks see our [blog post](http://vertex.ai/blog/announcing-plaidml) announcing the release. PlaidML is under active development and should be thought of as early alpha quality.

- [Current Limitations](#current-limitation)
- [Supported Hardware](#supported-hardware)
  - [Validated Hardware](#validated-hardware)
  - [Experimental Config](#experimental-config)
- [Validated Networks](#validated-networks)
- [Installation Instructions](#installation-instructions)
- [Building PlaidML](#building-plaidml)
- [Reporting Issues](#reporting-issues)

## Current Limitations

This version of PlaidML has some notable limitations which will be addressed soon in upcoming releases:

  * Start-up times can be quite long
  * Training throughput much lower than we'd like
  * RNN support is not implemented

### Validated Hardware

Vertex.AI runs a comprehensive set of tests for each release against these hardware targets:
  * AMD
    * R9 Nano
    * RX 480
  * NVIDIA
    * K80, GTX 780
    * GTX 1070

### Validated Networks
We support all of the Keras application networks from the current version (2.0.8). Validated networks are tested for performance and 
correctness as part of our continuous integration system.

 * CNNs
   * inception_v3
   * resnet50
   * vgg19
   * xception
   * mobilenet

## Installation Instructions

### Ubuntu Linux
If necessary, install Python's 'pip' tool.
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

If you have an NVIDIA graphics card:
```
sudo add-apt-repository ppa:graphics-drivers/ppa && sudo apt update
sudo apt install nvidia-modprobe nvidia-384 nvidia-opencl-icd-384 libcuda1-384
```
If you have an AMD card, [download the AMDGPU PRO driver and install](http://support.amd.com/en-us/kb-articles/Pages/AMDGPU-PRO-Driver-for-Linux-Release-Notes.aspx) according to AMD's instructions.

Install the PlaidML wheels system-wide:
```
sudo pip install -U plaidml-keras
```

Next, setup PlaidML to use your preferred computing device:
```
plaidml-setup
```

You can test your installation by running MobileNet in [plaidbench](https://github.com/plaidml/plaidbench):
```
git clone https://github.com/plaidml/plaidbench.git
cd plaidbench
sudo pip install -r requirements.txt
python plaidbench.py mobilenet
```

You can adapt any Keras code by using the PlaidML backend instead of the TensorFlow, CNTK, or Theano backend that you 
normally use.

Simply insert this code **BEFORE you `import keras`**:
```
# Install the plaidml backend
import plaidml.keras
plaidml.keras.install_backend()
```

### Plaidvision and Plaidbench

We've developed two open source projects: 

  * [plaidvision](https://github.com/plaidml/plaidvision) provides a simple shell for developing vision applications using your webcam
  * [plaidbench](https://github.com/plaidml/plaidbench) is a performance testing suite designed to help users compare the performance
  of different cards and different frameworks
  

### Hello VGG
One of the great things about Keras is how easy it is to play with state of the art networks. Here's all the code you
need to run VGG-19:
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

## Building PlaidML

PlaidML depends on [Bazel](http://bazel.build) v0.6.0 or higher.
```
bazel build -c opt plaidml:wheel plaidml/keras:wheel
sudo pip install -U bazel-bin/plaidml/*whl bazel-bin/plaidml/keras/*whl
```

## License

PlaidML is licensed under the [AGPLv3](https://www.gnu.org/licenses/agpl-3.0.txt). Commercial licenses and support for PlaidML are available from [Vertex.AI](mailto:info@vertex.ai).

## Reporting Issues
Either open a ticket on [GitHub](https://github.com/plaidml/plaidml/issues) or post to [plaidml-dev](https://groups.google.com/forum/#!forum/plaidml-dev).
