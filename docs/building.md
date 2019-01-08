# PlaidML Python Wheel

The PlaidML Python Wheel contains the PlaidML Python API, as well as
the PlaidML C++ headers, runtime libraries, and configuration files
needed to use PlaidML from other languages.

You can get the latest release of the PlaidML Python Wheel by running:

    pip install plaidml
    plaidml-setup

You can also build and install the wheel from source, by following
these instructions.

## Install Anaconda

Install [Anaconda].  You'll want to use a Python 3 version.

After installing Anaconda, you'll need to restart your shell, to pick
up its environment variable modifications (i.e. the path to the conda
tool and shell integrations).

For Microsoft Windows\* OS, you'll also need the Visual C++ compiler
and the Windows\* SDK, following the [Bazel-on-Windows] instructions.
Note that Conda will take care of the getting you a working Python and
an msys2 shell (you shouldn't need to set BAZEL_* environment
variables unless you've installed tools in unusual locations).

<<<<<<< HEAD
## Create and activate a Conda environment

macOS\* and Linux\*:

    conda env create -n plaidml
    conda activate plaidml

Microsoft Windows\* OS:
    
=======
## Create and Activate a Conda environment

macOS\* and Linux\*:
```
conda env create -n plaidml
conda activate plaidml
```

Microsoft Windows\* OS:
>>>>>>> edbba01e0f1d67eecfe18324c311584dac24ee9f
    conda env create -n plaidml -f environment-windows.yml
    conda activate plaidml

## Build the PlaidML Python wheel

macOS\*:
<<<<<<< HEAD
      
    bazel build --config macos_x86_64 //plaidml:wheel

Linux\*:
      
    bazel build --config linux_x86_64 //plaidml:wheel

Microsoft Windows\* OS:
      
=======
    bazel build --config macos_x86_64 //plaidml:wheel

Linux\*:
    bazel build --config linux_x86_64 //plaidml:wheel

Microsoft Windows\* OS:
>>>>>>> edbba01e0f1d67eecfe18324c311584dac24ee9f
    bazel build --config windows_x86_64 //plaidml:wheel

## Install the PlaidML Python wheel

    pip install -U bazel-bin/plaidml/*.whl
    plaidml-setup

# PlaidML with nGraph

Follow these instructions if you are wanting to work with the [PlaidML backend].

Building the nGraph PlaidML backend requires that you've installed the
PlaidML Python wheel.  You can do this by running:

    pip install plaidml
    plaidml-setup
<<<<<<< HEAD

or by following the instructions for building the PlaidML Python
Wheel, above.

When the PlaidML wheel is installed, the default nGraph build contains
the PlaidML backend.  From the nGraph source directory, you can run:

=======

or by following the instructions for building the PlaidML Python
Wheel, above.

When the PlaidML wheel is installed, the default nGraph build contains
the PlaidML backend.  From the nGraph source directory, you can run:

>>>>>>> edbba01e0f1d67eecfe18324c311584dac24ee9f
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=~/ngraph_plaidml_dist -DNGRAPH_CPU_ENABLE=OFF
    make
    make install
  
The above options will install nGraph into `~/ngraph_plaidml_dist`.
When you do not explicitly use `-DNGRAPH_CPU_ENABLE=FALSE`, the
default build enables CPU for operations.

After running `make` and `make install`, be sure to set the environment variables 
to the correct location where the libraries were built. Continuing the above 
example, this would be as follows for each respective OS:

## Linux\*

Most Linux distributions support `LD_LIBRARY_PATH`; consult the distribution's
documentation for specifics. 

    export LD_LIBRARY_PATH=~/ngraph_plaidml_dist/lib
    export NGRAPH_CPP_BUILD_PATH=~/ngraph_plaidml_dist

## macOS\* 

MacOS usually requires use of `DYLD_LIBRARY_PATH`.

    export DYLD_LIBRARY_PATH=~/ngraph_plaidml_dist/lib
    export NGRAPH_CPP_BUILD_PATH=~/ngraph_plaidml_dist

## Microsoft Windows\* OS 

Windows requires that dynamic libraries are on your `PATH`.

    set PATH=%PATH%:~/ngraph_plaidml_dist

## Test 

- To run the tests on experimental device #0, try something like
    `~/path/to/ngraph/build/test/unit-test --gtest_filter=PlaidML.*`.

- To run nBench, try something like
    `~/ngraph_plaidml_dist/bin/nbench -b "PlaidML:0" -f ~/test/model_inference_batch1_float32.json`.

  This runs the nGraph model specified in the file given by the `-f` option on 
  experimental device #0.

# PlaidML with Keras\*

<<<<<<< HEAD
The PlaidML-Keras Python wheel contains the code needed for
integration with Keras\*.

You can get the latest release of the PlaidML-Keras Python wheel by
=======
The PlaidML-Keras Python Wheel contains the code needed for
integration with Keras\*.

You can get the latest release of the PlaidML-Keras Python Wheel by
>>>>>>> edbba01e0f1d67eecfe18324c311584dac24ee9f
running:

    pip install plaidml-keras

You can also build and install the wheel from source.

<<<<<<< HEAD
## Set up a build environment
=======
## Setup a Build Environment
>>>>>>> edbba01e0f1d67eecfe18324c311584dac24ee9f

Follow the setup instructions for building the PlaidML Python Wheel,
above.

<<<<<<< HEAD
## Build the PlaidML-Keras wheel

macOS\*:
    `bazel build --config macos_x86_64 //plaidml/keras:wheel`

Linux\*:
    `bazel build --config linux_x86_64 //plaidml/keras:wheel`

Microsoft Windows\* OS:
    `bazel build --config windows_x86_64 //plaidml/keras:wheel`
=======
## Build the PlaidML-Keras Wheel

macOS\*:
    bazel build --config macos_x86_64 //plaidml/keras:wheel

Linux\*:
    bazel build --config linux_x86_64 //plaidml/keras:wheel

Microsoft Windows\* OS:
    bazel build --config windows_x86_64 //plaidml/keras:wheel
>>>>>>> edbba01e0f1d67eecfe18324c311584dac24ee9f

## Install the PlaidML-Keras Python wheel

    pip install -U bazel-bin/plaidml/keras/*.whl

# Testing PlaidML

Unit tests are executed through bazel:

    bazel test //...

Unit tests for frontends are marked manual and must be executed individually (requires 
running `plaidml-setup` prior to execution)

    bazel test //plaidml/keras:backend_test

[Anaconda]:https://www.anaconda.com/download
[Bazel]:http://bazel.build
[Bazel-on-Windows]:https://docs.bazel.build/versions/master/windows.html
[PlaidML backend]:https://ngraph.nervanasys.com/docs/latest/programmable/index.html#plaidml
[nGraph build]:https://ngraph.nervanasys.com/docs/latest/buildlb.html 
