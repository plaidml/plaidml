.. toctree::
   :maxdepth: 2
   :caption: Contents


Plaidbench
**********

Plaidbench measures the performance of the built-in Keras_ application networks,
using PlaidML_ or TensorFlow_ as a backend. Networks available are InceptionV3,
ResNet50, VGG16, VGG19, Xception, and (with Keras 2.0.6 and later) MobileNet.


Installation
============

Download the repository::

   git clone https://github.com/plaidml/plaidbench.git

Install the dependencies, including PlaidML_::

   cd plaidbench
   pip install -r requirements.txt


Examples
========

Run an inference benchmark for ResNet50 using PlaidML_::

   python plaidbench.py --plaid resnet50

Run a training benchmark for VGG16 using TensorFlow_::

  python plaidbench.py --no-plaid --train vgg16



Usage
=====

.. argparse::
   :module: plaidbench
   :func: make_parser
   :prog: plaidbench.py
   :nodefaultconst:

.. _TensorFlow: https://www.tensorflow.org
.. _PlaidML: https://github.com/vertexai/plaidml
.. _Keras: https://keras.io

