<div align=center><a href="https://www.intel.ai/plaidml"><img
src="docs/assets/images/plaid-final.png" height="200"></a><br>

*A platform for making deep learning work everywhere.*

[Documentation] |
[Installation Instructions] |
[Building PlaidML] |
[Contributing] |
[Troubleshooting] |
[Reporting Issues](#reporting-issues)

</div>

[![License]](https://github.com/plaidml/plaidml/blob/master/LICENSE)
[![Build status]](https://buildkite.com/plaidml/plaidml-plaidml)

# To Our Users

First off, we’d like to thank you for choosing PlaidML. Whether you’re a new
user or a multi-year veteran, we greatly appreciate you for the time you’ve
spent tinkering around with our source code, sending us feedback, and improving
our codebase. PlaidML would truly not be the same without you.

The feedback we have received from our users indicates an ever-increasing need
for performance, programmability, and portability.  During the past few months,
we have been restructuring PlaidML to address those needs. Below is a summary of
the biggest changes: 
* We’ve adopted [MLIR], an extensible compiler infrastructure that has gained
  industry-wide adoption since its release in early 2019. MLIR makes it easier
  to integrate new software and hardware into our compiler stack, as well as
  making it easier to write optimizations for our compiler.
* We’ve worked extensively on [Stripe], our low-level intermediate
  representation within PlaidML. Stripe contains optimizations that greatly
  improve the performance of our compiler. While our work on Stripe began before
  we decided to use MLIR, we are in the process of fully integrating Stripe into
  MLIR.
* We created our C++/Python embedded domain-specific language ([EDSL])
  to improve the programmability of PlaidML.

Today, we’re announcing a new branch of PlaidML — `plaidml-v1`. This will act as
our development branch going forward and will allow us to more rapidly prototype
the changes we’re making without breaking our existing user base. As a
precaution, please note that certain features, tests, and hardware targets may
be broken in `plaidml-v1`. 

You can continue to use code on the `master` branch or from our releases on
PyPI. For your convenience, the contents of our `master` branch will be released
as version 0.7.0. We are keeping the `master` branch of PlaidML stable and
maintaining it until `plaidml-v1` is ready for production.

If you’d like to try out some of PlaidML’s newer performance improvements, you
can try running PlaidML with the environment variable `PLAIDML_USE_STRIPE=1`. 
This will act as a precursor to the changes you’ll be seeing in `plaidml-v1`, 
and we’re excited to hear your feedback on Stripe.

Your support means a lot to us. Thank you for being understanding of our new
development process during this new and exciting time for deep learning
compilers.

-----

PlaidML is an advanced and portable tensor compiler for enabling deep learning
on laptops, embedded devices, or other devices where the available computing
hardware is not well supported or the available software stack contains
unpalatable license restrictions.

PlaidML sits underneath common machine learning frameworks, enabling users to
access any hardware supported by PlaidML. PlaidML supports [Keras], [ONNX], and
[nGraph].

As a component within the [nGraph Compiler stack], PlaidML further extends the
capabilities of specialized deep-learning hardware (especially GPUs,) and makes
it both easier and faster to access or make use of subgraph-level optimizations
that would otherwise be bounded by the compute limitations of the device.

As a component under [Keras], PlaidML can accelerate training workloads with
customized or automatically-generated Tile code. It works especially well on
GPUs, and it doesn't require use of CUDA/cuDNN on Nvidia hardware, while
achieving comparable performance.

PlaidML works on all major operating systems: Linux, macOS, and Windows.

If you are using a hardware target not supported by PlaidML by default, such as
Clover, check out the instructions at [building PlaidML] to build a custom
configuration to support your hardware.

## Prerequisites
- Python (v2 supported, v3 recommended)
- OpenCL 1.2 or greater

## Quick Start

See the [troubleshooting] section for solutions to common issues.

```
virtualenv plaidml
source plaidml/bin/activate
pip install plaidml-keras plaidbench
```

Choose which accelerator you'd like to use (many computers, especially laptops,
have multiple):

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

## Installation Instructions

We support a variety of operating systems and installation methods.

* [Ubuntu][install-ubuntu]
* [macOS][install-macos]
* [Windows][install-windows]

## Demos and Related Projects

### Plaidbench

[Plaidbench] is a performance testing suite designed to help users compare the
performance of different cards and different frameworks.

### Hello VGG

One of the great things about Keras is how easy it is to play with state of the
art networks. Here's all the code you need to run VGG-19:

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

## Reporting Issues

Either open a ticket on [GitHub] or join our [slack channel (#plaidml)][slack].

## CI & Validation

### Validated Hardware

A comprehensive set of tests for each release are run against the hardware
targets listed below.

* AMD
  * R9 Nano
  * RX 480
  * Vega 10


* Intel
  * HD4000
  * HD Graphics 505


* NVIDIA
  * K80
  * GT 640M
  * GTX 1050
  * GTX 1070

### Validated Networks

We support all of the Keras application networks from
current versions of 2.x. Validated networks are tested for performance and
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

[nGraph Compiler stack]: https://ngraph.nervanasys.com/docs/latest/
[Keras]: https://keras.io/
[GitHub]: https://github.com/plaidml/plaidml/issues
[plaidml-dev]: https://groups.google.com/forum/#!forum/plaidml-dev
[ONNX]: https://github.com/onnx
[nGraph]: https://github.com/NervanaSystems/ngraph
[slack]: https://join.slack.com/t/ngraph/shared_invite/enQtNjY1Njk4OTczMzEyLWIyZjZkMDNiNzJlYWQ3MGIyZTg2NjRkODAyYWZlZWY5MmRiODdlNzVkMjcxNjNmNWEyZjNkMDVhMTgwY2IzOWQ
[Documentation]: https://plaidml.github.io/plaidml/
[Installation Instructions]: https://plaidml.github.io/plaidml/docs/install
[Building PlaidML]: https://plaidml.github.io/plaidml/docs/building
[Contributing]: https://plaidml.github.io/plaidml/docs/contributing
[Troubleshooting]: https://plaidml.github.io/plaidml/docs/troubleshooting
[License]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[Build status]: https://badge.buildkite.com/5c9add6b89a14fd498e69a5035062368480e688c4c74cbfab3.svg?branch=master
[Plaidbench]: https://github.com/plaidml/plaidml/tree/master/plaidbench
[install-ubuntu]: https://plaidml.github.io/plaidml/docs/install#ubuntu
[install-macos]: https://plaidml.github.io/plaidml/docs/install#macos
[install-windows]: https://plaidml.github.io/plaidml/docs/install#windows
[EDSL]: https://plaidml.github.io/plaidml/docs/edsl
[MLIR]: https://mlir.llvm.org/
[Stripe]: https://arxiv.org/abs/1903.06498
