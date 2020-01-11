---
nav_order: 2
---

# Building from source

## Install Anaconda

Install [Anaconda].  You'll want to use a Python 3 version.

After installing Anaconda, you'll need to restart your shell, to pick up its
environment variable modifications (i.e. the path to the conda tool and shell
integrations).

For Microsoft Windows, you'll also need the Visual C++ compiler (2017+) and the
Windows SDK, following the [Bazel-on-Windows] instructions.

## Install bazelisk

The [Bazelisk] tool is a wrapper for `bazel` which provides the ability to
enfore a particular version of Bazel. 

Download the latest version for your platform and place the executable somewhere
in your PATH (e.g. `/usr/local/bin`). You will also need to mark it as
executable. Example:

```
wget https://github.com/bazelbuild/bazelisk/releases/download/v0.0.8/bazelisk-darwin-amd64
mv bazelisk-darwin-amd64 /usr/local/bin
chmod +x /usr/local/bin/bazelisk
```

https://github.com/bazelbuild/bazelisk/releases

## Configure the build

Use the `configure` script to configure your build. Note: the `configure` script
requires Python 3.

By default, running the `configure` script will:
* Create and/or update your conda environment
* Configure pre-commit hooks for development purposes
* Configure bazelisk based on your host OS

```
./configure
```

On Windows, use:

```
python configure
```

Here's an example session:

```
$ ./configure
Configuring PlaidML build environment

conda found at: /usr/local/miniconda3/bin/conda
Creating conda environment from: $HOME/src/plaidml/environment.yml

Searching for pre-commit in: $HOME/src/plaidml/.cenv/bin
pre-commit installed at .git/hooks/pre-commit

bazelisk version
Bazelisk version: v0.0.8
Starting local Bazel server and connecting to it...
Build label: 0.28.1
Build target: bazel-out/darwin-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Fri Jul 19 15:22:50 2019 (1563549770)
Build timestamp: 1563549770
Build timestamp as int: 1563549770

Using variant: macos_x86_64


Your build is configured.
Use the following to run all unit tests:

bazelisk test //...
```

## Build the PlaidML Python wheel

```
bazelisk build //plaidml:wheel
```

## Install the PlaidML Python wheel

```
pip install -U bazel-bin/plaidml/wheel.pkg/dist/*.whl
plaidml-setup
```

# PlaidML with Keras

The PlaidML-Keras Python Wheel contains the code needed for
integration with Keras.

You can get the latest release of the PlaidML-Keras Python Wheel by
running:
```
pip install plaidml-keras
```
You can also build and install the wheel from source.

## Set up a build environment

Follow the setup instructions for [building the PlaidML Python Wheel], above.

## Build the PlaidML-Keras wheel

```
bazelisk build //plaidml/keras:wheel
```

## Install the PlaidML-Keras Python wheel

```
  pip install -U bazel-bin/plaidml/keras/wheel.pkg/dist/*.whl
```

# Testing PlaidML

Unit tests are executed through bazel:
```
bazelisk test //...
```

Unit tests for frontends are marked manual and must be executed individually (requires
running `plaidml-setup` prior to execution)
```
bazelisk run //plaidml/keras:backend_test
```

[Anaconda]:https://www.anaconda.com/download
[Bazel]:http://bazel.build
[Bazelisk]:https://github.com/bazelbuild/bazelisk
[Bazel-on-Windows]:https://docs.bazel.build/versions/master/windows.html
[PlaidML backend]:https://www.ngraph.ai/documentation/backend-support/cpp-api#plaidml
[nGraph build]:https://ngraph.nervanasys.com/docs/latest/buildlb.html
[building the PlaidML Python Wheel]:(#build-the-plaidml-python-wheel)
