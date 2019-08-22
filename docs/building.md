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

Use the `configure` script to configure your built. Note: the `configure` script requires Python 3.

By default, running the `configure` script will:
* Create and/or update your conda environment
* Configure pre-commit hooks for development purposes
* Configure bazelisk based on the device you're using

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
pip install -U bazel-bin/plaidml/*.whl
plaidml-setup
```

# PlaidML with nGraph

Follow these instructions if you are wanting to work with the [PlaidML backend].

Building the nGraph PlaidML backend requires that you've installed the
PlaidML Python wheel.  You can do this by running:
```
pip install plaidml
plaidml-setup
```

or by following the instructions for [building the PlaidML Python Wheel], above.

When the PlaidML wheel is installed, the default nGraph build contains
the PlaidML backend.  From the nGraph source directory, you can run:
```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/ngraph_plaidml_dist -DNGRAPH_PLAIDML_ENABLE=ON -DNGRAPH_CPU_ENABLE=OFF
make
make install
```
Running the build with the above options will install nGraph into
`~/ngraph_plaidml_dist`. When you do not explicitly use
`-DNGRAPH_CPU_ENABLE=FALSE`, the default build enables CPU for operations.

After running `make` and `make install`, be sure to set the environment
variables to the correct location where the libraries were built. Continuing the
above example, this would be as follows for each respective OS:

## Linux

Most Linux distributions support `LD_LIBRARY_PATH`; consult the distribution's
documentation for specifics.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/ngraph_plaidml_dist/lib
export NGRAPH_CPP_BUILD_PATH=~/ngraph_plaidml_dist
```

## macOS

MacOS usually requires use of `DYLD_LIBRARY_PATH`.
```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/ngraph_plaidml_dist/lib
export NGRAPH_CPP_BUILD_PATH=~/ngraph_plaidml_dist
```

## Microsoft Windows

Windows requires that dynamic libraries are on your `PATH`.
```
set PATH=%PATH%:~/ngraph_plaidml_dist
```

## Test

- To run the tests on experimental device #0, try something like
    `~/path/to/ngraph/build/test/unit-test --gtest_filter=PlaidML.*`.

- To run nBench, try something like
    `~/ngraph_plaidml_dist/bin/nbench -b "PlaidML:0" -f ~/test/model_inference_batch1_float32.json`.

  This runs the nGraph model specified in the file given by the `-f` option on
  non-experimental device #0. If you want to use an experimental device, set
  the environment variable `PLAIDML_EXPERIMENTAL=1`.

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
  pip install -U bazel-bin/plaidml/keras/*.whl
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

# Custom Hardware Configuration

Custom configuration can be set by producing a `.json` file. Here is an example
with two device configurations specified:
```json
{
"platform": {
    "@type": "type.vertex.ai/vertexai.tile.local_machine.proto.Platform",
    "hardware_configs": [
    {
        "description": "Intel *HD Graphics GPU config",
        "sel": {
        "and": {
            "sel": [
            {
                "name_regex": ".*HD Graphics.*"
            },
            {
                "platform_regex": "Metal.*"
            }
            ]
        }
        },
        "settings": {
        "threads": 128,
        "vec_size": 4,
        "mem_width": 256,
        "max_mem": 16000,
        "max_regs": 16000,
        "goal_groups": 6,
        "goal_flops_per_byte": 50
        }
    },
    {
        "description": "Intel Iris GPU config",
        "sel": {
        "and": {
            "sel": [
            {
                "name_regex": ".*Iris.*"
            },
            {
                "platform_regex": "Metal.*"
            }
            ]
        }
        },
        "settings": {
        "threads": 128,
        "vec_size": 4,
        "mem_width": 256,
        "max_mem": 16000,
        "max_regs": 16000,
        "goal_groups": 6,
        "goal_flops_per_byte": 50,
        "use_global": true
        }
    }
    ]
}
}
```

The custom configuration file can be read into PlaidML by setting the
enviornment variable `PLAIDML_EXPERIMENTAL_CONFIG` to point to the custom `.json`
file. PlaidML will then read the custom configuration file and list it as a
device when running `plaidml-setup`.

[Anaconda]:https://www.anaconda.com/download
[Bazel]:http://bazel.build
[Bazelisk]:https://github.com/bazelbuild/bazelisk
[Bazel-on-Windows]:https://docs.bazel.build/versions/master/windows.html
[PlaidML backend]:https://www.ngraph.ai/documentation/backend-support/cpp-api#plaidml
[nGraph build]:https://ngraph.nervanasys.com/docs/latest/buildlb.html
[building the PlaidML Python Wheel]:(#build-the-plaidml-python-wheel)
