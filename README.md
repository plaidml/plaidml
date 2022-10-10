<div align=center><a href="https://www.intel.ai/plaidml"><img
src="docs/assets/images/plaid-final.png" height="200"></a><br>

*A platform for making deep learning work everywhere.*

</div>

[![License]](https://github.com/plaidml/plaidml/blob/master/LICENSE)
[![Build status](https://badge.buildkite.com/87cb87799399a2e27c6f99b1839a66e9101b6f132b46d36089.svg)](https://buildkite.com/intel/tpp-plaidml)

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
be broken in `plaidml-v1` as is a research project. Right now `plaidml-v1`
only supports Intel and AMD CPUs with AVX2 and AVX512 support.

You can continue to use code on the `master` branch or from our releases on
PyPI. For your convenience, the contents of our `master` branch will be released
as version 0.7.0. There is no further development in this branch. `plaidml-v1` is 
a research project.

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


## Building PlaidML from source 

Due to use of conda PlaidML runs on all major Linux distributions.

```
export PLAIDML_WORKSPACE_DIR=[choose a directory of your choice]

# setting up miniconda env
cd ${PLAIDML_WORKSPACE_DIR}
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
bash Miniconda3-py37_4.12.0-Linux-x86_64.sh -p ${PLAIDML_WORKSPACE_DIR}/miniconda3
eval "$(${PLAIDML_WORKSPACE_DIR}/miniconda3/bin/conda shell.bash hook)"
conda activate

# clone plaidml-v1 and set up env
git clone https://github.com/plaidml/plaidml.git --recursive -b plaidml-v1
cd plaidml
conda env create -f environment.yml -p .cenv/
conda activate .cenv/

# we might need to go into .cenv/bin and create a sym-link 
cd .cenv/bin/
ln -s ninja ninja-build
cd ../../

# preparing PlaidML build
./configure

# buidling PlaidML
cd build-x86_64/Release
ninja && PYTHONPATH=$PWD python plaidml/plaidml_setup.py
```

## Demos and Related Projects

### Plaidbench

[Plaidbench] is a performance testing suite designed to help users compare the
performance of different cards and different frameworks.

```
cd build-x86_64/Release
ninja plaidbench_py && PYTHONPATH=$PWD KMP_AFFINITY=granularity=fine,verbose,compact,1,0 OMP_NUM_THREADS=8 python plaidbench/plaidbench.py -n128 keras resnet50
```

The command above is suited for 8-core Intel/AMD CPUs with hyper-threading enabled. E.g. on an Intel i9-11900K we expect around 8.5ms latency. 


## Reporting Issues

Either open a ticket on [GitHub].

## CI & Validation

### Validated Hardware

A comprehensive set of tests for each release are run against the hardware
targets listed below.

* AMD CPUs with AVX2 and AVX512
* Intel CPUs with AVX2 and AVX512

### Validated Networks

We support all of the Keras application networks from
current versions of 2.x. Validated networks are tested for performance and
correctness as part of our continuous integration system.

* CNNs
  * Inception v3
  * ResNet50
  * VGG19
  * VGG16
  * Xception
  * DenseNet

[LIBXSMM]: https://github.com/libxsmm/libxsmm/ 
[nGraph Compiler stack]: https://ngraph.nervanasys.com/docs/latest/
[Keras]: https://keras.io/
[GitHub]: https://github.com/plaidml/plaidml/issues
[ONNX]: https://github.com/onnx
[nGraph]: https://github.com/NervanaSystems/ngraph
[License]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[Build status]: https://badge.buildkite.com/5c9add6b89a14fd498e69a5035062368480e688c4c74cbfab3.svg?branch=master
[Plaidbench]: https://github.com/plaidml/plaidml/tree/plaidml-v1/plaidbench
[EDSL]: https://plaidml.github.io/plaidml/docs/edsl
[MLIR]: https://mlir.llvm.org/
[Stripe]: https://arxiv.org/abs/1903.06498
