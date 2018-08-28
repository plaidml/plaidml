<img src="docs/images/plaid-final.png" height="200"></img>

*A platform for making deep learning work everywhere.*

**Vertex.AI (the creators of PlaidML) is excited to join Intel's Artificial Intelligence Products Group. PlaidML will soon be re-licensed under Apache 2. Read the announcement [here!](https://vertex.ai)**

[![Build Status](https://travis-ci.org/plaidml/plaidml.svg?branch=master)](https://travis-ci.org/plaidml/plaidml)

PlaidML is the *easiest, fastest* way to learn and deploy deep learning on any device, especially those running macOS or Windows:
  * **Fastest:** PlaidML is often 10x faster (or more) than popular platforms (like TensorFlow CPU) because it supports all GPUs, *independent of make and model*. 
    * PlaidML accelerates deep learning on AMD, Intel, NVIDIA, ARM, and embedded GPUs.
  * **Easiest:** PlaidML is simple to [install](docs/installing.md) and supports multiple frontends (Keras and ONNX currently)
  * **Free:** PlaidML is completely open source and doesn't rely on any vendor libraries with proprietary and restrictive licenses.

For most platforms, getting started with accelerated deep learning is as easy as running a few commands (assuming you have Python (v2 or v3) installed (if this doesn't work, see the [installation instructions](docs/installing.md)):
```
virtualenv plaidml
source plaidml/bin/activate
pip install plaidml-keras plaidbench
```
Choose which accelerator you'd like to use (many computers, especially laptops, have multiple):
```
plaidml-setup
```

Next, try benchmarking MobileNet inference performance:
```
plaidbench keras mobilenet
```
Or, try training MobileNet:
```
plaidbench --batch-size 16 keras --train mobilenet
```


# About PlaidML

PlaidML is a multi-language acceleration platform that: 
  
  * Enables practitioners to deploy high-performance neural nets on any device
  * Allows hardware developers to quickly integrate with high-level frameworks
  * Allows framework developers to easily add support for many kinds of hardware
  * Works on all major platforms — Linux, [macOS](http://vertex.ai/blog/plaidml-mac-preview), [Windows](http://vertex.ai/blog/deep-learning-for-everyone-plaidml-for-windows)
  * Allows developers to create hardware accelerated, novel, performance portable research kernels.

For examples and benchmarks, see our [blog](http://vertex.ai/blog).

- [Documentation](https://vertexai-plaidml.readthedocs-hosted.com/)
- [Installation Instructions](docs/installing.md)
- [Building PlaidML](docs/building.md)
- [Contributing](docs/contributing.rst)
- [Reporting Issues](#reporting-issues)

### Recent Release Notes
* PlaidML 0.3.3 - 0.3.5
  * Support Keras 2.2.0 - 2.2.2
  * Support ONNX 1.2.1
  * Upgrade kernel scheduling
  * Revise documentation
  * Add HALs for CUDA and Metal
  * Various bugfixes and improvements
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
    * K80, GTX 780, GT 640M
    * GTX 1070, 1050
  * Intel
    * HD4000
    * HD Graphics 505

### Validated Networks
We support all of the Keras application networks from current versions of 2.x. Validated networks are tested for performance and 
correctness as part of our continuous integration system.

 * CNNs
   * Inception v3
   * ResNet50
   * VGG19
   * Xception
   * MobileNet
   * DenseNet
   * ShuffleNet

 * LSTM
   * examples/imdb_lstm.py (from keras)

## Installation Instructions

See detailed per platform instructions [here](docs/installing.md).

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
import os
import time

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

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
