# Intel Corporation Machine Learning Benchmarks

Plaidbench measures the performance of machine-learning networks.

Plaidbench supports:

* Benchmarking across various frontends (the packages that provide ways to represent ML networks)

* Benchmarking across various backends (the packages used by the frontends to actually run the network).

Plaidbench was created to quantify the performance of [PlaidML](http://www.github.com/plaidml/plaidml) relative to other frameworks' backends across a variety of hardware, and to help determine which networks provide acceptable performance in various application deployment scenarios.

## Current Status

[![Build Status](https://travis-ci.org/plaidml/plaidbench.svg?branch=master)](https://travis-ci.org/plaidml/plaidbench)
[![Build status](https://ci.appveyor.com/api/projects/status/307lhqu7kp2m0j0v?svg=true)](https://ci.appveyor.com/project/earhart/plaidbench)

## Installation

To get the basic framework and command-line interface:

    pip install plaidbench

If you know which ML frontends you'll want to use, you can install their pre-requisites ahead of time:

    pip install plaidbench[keras]
    pip install plaidbench[onnx]

You can also install various ML backends -- for example,

    pip install plaidml-keras
    pip install tensorflow
    pip install caffe2
    pip install onnx-plaidml
    pip install onnx-tf

If you don't have a particular package installed, and you run benchmarks that require the package, Plaidbench will try to determine what needs to be installed and tell you how to install it.

If you're using PlaidML as a backend, you'll want to run `plaidml-setup` to configure it correctly for your hardware.

## Usage

Plaidbench provides a simple command-line interface; global flags are provided immediately, and subcommands are used to select the frontend framework and to provide framework-specific options.

For example, to benchmark [ShuffleNet](https://arxiv.org/abs/1707.01083) on [ONNX](https://onnx.ai/) using PlaidML, writing results to the directory `~/shuffle_results`, you can run:

    plaidbench --result ~/shuffle_results onnx --plaid shufflenet

For a complete overview of the supported global flags, use `plaidbench --help`; for the individual subcommand flags, specify `--help` with the subcommand (e.g. `plaidbench keras --help`).

## Supported Configurations

Plaidbench supports:

* Keras

  * Backends: PlaidML and Tensorflow

  * Networks: Inception-V3, ResNet50, Vgg16, Vgg19, Xception, and (in Keras 2.0.6 and later) MobileNet.

  * Training vs. Inference performance, and fp16 vs. fp32 performance.

* ONNX

  * Backends: PlaidML, Caffe2, and Tensorflow

  * Networks: AlexNet, DenseNet, Inception-V1, Inception-V2, Resnet50, ShuffleNet, SqueezeNet, Vgg16, and Vgg19.
