# PlaidML
*A framework for making machine learning work everywhere.*
![The PlaidML Platypus](https://github.com/vertexai/plaidml/raw/master/images/plaid-final.png)

PlaidML is a framework that uses the *Tile* language to enable machine learning to work on any device, 
from any framework. We'll be releasing more details about how *Tile* works and why it's so amazing in the near future.

## Supported Hardware, Networks, and Operations
We're constantly working on expanding the set of qualifies platforms for running PlaidML. 
To ensure our users have a positive experience with performance, we only support a subset of commonly available hardware,
though dedicated hackers can enable support for any devices that supports OpenCL 1.1 or higher.

### Currently Supported Hardware
*These cards are tested for correctness and performance on every commit*
  * AMD
    * Fiji
      * R9 Nano
    * Ellsemere
      * RX480
  * NVidia
    * Kepler
      * K80
      * GTX780
    * Pascal
      * GTX1070

The default configs we ship with should work with a slightly broader range of AMD and NVidia hardware.

We also provide experimental configurations for all Fiji, Ellsemere, Kepler, and Pascal devices. 
We can't promise they'll work well though. If your card isn't currently supported, helpful instructions will be printed;
you can enable experimental cards by running with the environment variable `PLAIDML_EXPERIMENTAL=1`

If you have multiple supported devices, you'll need to choose between them by setting `PLAIDML_DEVICE_IDX=<id>`. Again,
instructions will be printed.

### Networks
We support most of the convolutional Keras application networks. Supported networks are
tested for performance and correctness as part of our continuous integration system on
our supported hardware. Other networks may work.

 * CNNs
   * inception_v3
   * resnet50
   * vgg19
   * xception
 * RNNs
   * `<coming soon>`

### Operations & Data Formats
We support many but not all of the keras backend operations. We welcome community contributions to cover more operations.

We currently only officially support single precision floating point. *Tile* itself is data type agnostic and we expect to support every meaningful data type in the near future.

Currently we support:
 * Most CNN related operations
 * 1d, 3d, and separable convolutions

## Installation Instructions
Just install the pip and you're ready to go.

The Pre-release bundles the wheels inside the `whl` directory, so you'll install them from there:
`sudo pip install whl/*`

~~`sudo pip install plaidml-keras`~~

You can adapt any Keras code to use PlaidML simply by using the PlaidML backend instead
of the TensorFlow, CTNK, or Theano backend that you normally use.
```
# Install the plaidml backend
import plaidml.keras
plaidml.keras.install_backend()
```

### Plaidvision and Plaidmark

We've developed two open source projects: 
  **PRE-RELEASE USERS: plaidvision and plaidbench are included in the release tarball.**

  * [Plaidvision](https://github.com/vertexai/plaidvision) provides a simple shell for developing vision applications using your webcam
  * [Plaidbench](https://github.com/vertexai/plaidbench) is a performance testing suite designed to help users compare the performance
  of different cards and different frameworks
  

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
Please use GitHub to report any issues you encounter. Please search for duplicates before posting.
