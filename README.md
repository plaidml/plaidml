# PlaidML
*A framwork for making machine learning work everywhere.*
![The PlaidML Platypus](https://raw.githubusercontent.com/vertexai/plaidml/master/images/plaid-final.png)

PlaidML is a framework that uses the *Tile* language to enable machine learning to work on any device, 
from any framework. We'll be releasing more details about how *Tile* works and why it's so amazing in the near future.

## Supported Hardware, Networks, and Operations
We're constantly working on expanding the set of qualifies platforms for running PlaidML. 
To ensure our users have a positive experience with performance, we only support a subset of commonly available hardware, though dedicated hackers can enable support for any devices that supports OpenCL 1.1 or higher.

### Currently Supported Hardware
*These cards are tested for correctness and performance on every commit*
  * AMD
    * Fiji
      * R9 Nano
    * Ellsemere
      * RX480
    * Vega
      * Vega 10
  * NVidia
    * Kepler
      * K80
      * GTX780
    * Pascal
      * GTX 10XX
  * Intel, Qualcomm, ARM, PowerVR
    * <coming soon>

We've provided experimental default configurations for all Fiji, Ellsemere, Kepler, and Pascal devices. We can't promise they'll work well though.

### Networks
We support most the convolutional Keras application networks. RNNs are coming soon. Support in this context means they are tested (for performance and correctness) as part of our continuous integration system on our supported hardware. Other networks may work.

 * CNNs
   * inception_v3
   * resnet50
   * vgg19
   * xception

### Operations & Data Formats
We support many but not all of the keras backend operations. We welcome community contributions. Once you get the hang of writing *Tile* code, you'll be hooked.

We currently only officially support single precision floating point. *Tile* itself is data type agnostic and we expect to support every meaningful data type in the near future.

Currently we support:
 * Most CNN related operations
 * 1d, 3d, and separable convolutions

## Installation Instructions
Just install the pip and you're ready to go.

`sudo pip install plaidml-keras`

### Hello Xception
One of the great things about keras is how easy it is to play with state-of-the-art networks.
```
<Code goes here>
```

You can adapt any Keras code to use PlaidML simply by using the PlaidML backend instead
of the TensorFlow, CTNK, or Theano backend that you normally use.
```
<Code goes here>
```

## Reporting Issues
Please use GitHub to report any issues you encounter. Please search for duplicates before posting.

## Benchmarking New Devices
Developers are welcome to modify the shipped configurations to support their available 
hardware. We hope to add many, many more platforms to our support matrix. If you add support
for a card, please do it in the `community.yml` file and send us a pull request, along with the output of `benchmark.sh`