# PlaidML
[![Build Status](https://travis-ci.org/plaidml/plaidml.svg?branch=master)](https://travis-ci.org/plaidml/plaidml)

![The PlaidML Platypus](docs/plaid-final.png)
*A framework for making deep learning work everywhere.*

PlaidML is a multi-language acceleration framework that: 
  
  * Enables practitioners to deploy high-performance neural nets on any device
  * Allows hardware developers to quickly integrate with high-level frameworks
  * Allows framework developers to easily add support for many kinds of hardware
  * Works on all major platforms - linux, [macOS](http://vertex.ai/blog/plaidml-mac-preview), [Windows](http://vertex.ai/blog/deep-learning-for-everyone-plaidml-for-windows)
  * Allows developers to create hardware accelerated, novel, performance portable research kernes.

For examples and benchmarks, see our blog [blog](http://vertex.ai/blog).

- [Documentation](https://vertexai-plaidml.readthedocs-hosted.com/)
- [Installation Instructions](#installation-instructions)
- [Building PlaidML](docs/building.md)
- [Contributing](docs/contributing.rst)
- [Reporting Issues](#reporting-issues)

### Recent Release Notes
* PlaidML 0.3.2
  * Now supports ONNX 1.1.0 as a backend through [onnx-plaidml](https://github.com/plaidml/onnx-plaidml)
  * Preliminary support for LLVM. Currently only supports CPUs, and only on Linux and macOS. More soon.
  * Support for LSTMs & RNNs with static loop sizes, such as examples/imdb_lstm.py (from Keras)
    * Training networks with embeddings is especially slow (#96)
    * RNNs are only staticly sized if the input's sequence length is explicitly specified (#97)
    * Fixes bug related to embeddings (#92)
  * Adds a shared generic op library in python to make creating frontends easier
     * plaidml-keras now uses this library
  * Uses [plaidml/toolchain](https://github.com/plaidml/toolchain) for builds
     * Building for ARM is now simple (–-config=linux_arm_32v7)
  * Various fixes for bugs (#89)


### Validated Hardware

Vertex.AI runs a comprehensive set of tests for each release against these hardware targets:
  * AMD
    * R9 Nano
    * RX 480
    * Vega 10
  * NVIDIA
    * K80, GTX 780
    * GTX 1070, 1050
  * Intel
    * HD4000
    * HD Graphics 505

### Validated Networks
We support all of the Keras application networks from current versions of 2.x. Validated networks are tested for performance and 
correctness as part of our continuous integration system.

 * CNNs
   * inception_v3
   * resnet50
   * vgg19
   * xception
   * mobilenet
   * densenet
   * shufflenet

 * LSTM
   * examples/imdb_lstm.py (from keras)

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

Best practices for python include judicious usage of [Virtualenv](https://virtualenv.pypa.io/en/stable/), and we certainly recommend creating one just for plaidml:
```
virtualenv plaidml-venv
source ./plaidml-venv/bin/activate
pip install -U plaidml-keras
```

Alternatively, install the PlaidML wheels system-wide:
```
sudo -H pip install -U plaidml-keras
```

Next, setup PlaidML to use your preferred computing device:
```
plaidml-setup
```

You can test your installation by running MobileNet in [plaidbench](https://github.com/plaidml/plaidbench):
(Remember to use sudo -H if you're not using a Virtualenv)
```
git clone https://github.com/plaidml/plaidbench.git
cd plaidbench
pip install -r requirements.txt
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
```python
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

## License

PlaidML is licensed under the [AGPLv3](https://www.gnu.org/licenses/agpl-3.0.txt). 

Our open source goals include 1) helping students get started with deep learning as easily as possible and 2) helping researchers develop new methods more quickly than is possible with other tools. PlaidML is unique in being fully open source and free of dependence on libraries like cuDNN that carry revocable and redistribution-prohibiting licenses. For situations where an alternate license is preferable please contact [solutions@vertex.ai](mailto:solutions@vertex.ai).

## Reporting Issues
Either open a ticket on [GitHub](https://github.com/plaidml/plaidml/issues) or post to [plaidml-dev](https://groups.google.com/forum/#!forum/plaidml-dev).
