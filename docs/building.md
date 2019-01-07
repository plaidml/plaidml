
# PlaidML with nGraph

Follow these instructions if you are wanting to work with the [PlaidML backend].

Note that the [nGraph build] must be compiled with, at minimum, the 
`-DNGRAPH_PLAIDML_ENABLE=TRUE` flag. If your `clinfo` driver is already installed 
for Nvidia* or AMD* GPUs, try something like:  

    cmake .. -DCMAKE_INSTALL_PREFIX=~/ngraph_plaidml_dist -DNGRAPH_PLAIDML_ENABLE=TRUE -DNGRAPH_CPU_ENABLE=FALSE
  
The above options will build nGraph with PlaidML into `~/ngraph_plaidml_dist`. 
When you do not explicitly use `-DNGRAPH_CPU_ENABLE=FALSE`, the default build 
enables CPU for operations.

After running `make` and `make install`, be sure to set the environment variables 
to the correct location where the libraries were built. Continuing the above 
example, this would be as follows for each respective OS:  

### Linux\*

Most Linux distributions support `LD_LIBRARY_PATH`; consult the distribution's
documentation for specifics. 

    export LD_LIBRARY_PATH=~/ngraph_plaidml_dist/lib
    export NGRAPH_CPP_BUILD_PATH=~/ngraph_plaidml_dist
    export PlaidML_DIR=$PlaidML_DIR

### macOS\* 

MacOS usually requires use of `DYLD_LIBRARY_PATH`.

    export DYLD_LIBRARY_PATH=~/ngraph_plaidml_dist/lib
    export NGRAPH_CPP_BUILD_PATH=~/ngraph_plaidml_dist
    export PlaidML_DIR=$PlaidML_DIR

### Microsoft Windows\* OS 

Consult the Microsoft documentation for that version of Windows for how to set 
`$PATH` variables.


## Test 

- To run the tests on experimental device #0, try something like:
  
        PLAIDML_DEFAULT_CONFIG=~/path/to/plaidml/configs/experimental.json ~/path/to/ngraph/build/test/unit-test --gtest_filter=PlaidML.*


- To run nBench, try something like

        PLAIDML_DEFAULT_CONFIG=~/path/to/plaidml/plaidml/configs/experimental.json ~/ngraph_plaidml_dist/bin/nbench -b "PlaidML:0" -f ~/test/model_inference_batch1_float32.json

  This runs the nGraph model specified in the file given by the `-f` option on 
  experimental device #0.


# PlaidML with Keras\*

PlaidML with Keras depends on [Bazel] v0.12.0 or higher.

## Building PlaidML

### Install Requirements

    pip install -r requirements.txt

### Linux

    bazel build --config linux_x86_64 plaidml:wheel plaidml/keras:wheel
    sudo pip install -U bazel-bin/plaidml/*whl bazel-bin/plaidml/keras/*whl

### macOS

The versions of bison and flex that are provided with xcode are too old to build 
PlaidML. It's easiest to use homebrew to install all the prerequisites:

    brew install bazel bison flex

Then, use bazel to build and specify the correct config from ``tools/bazel.rc``: 

    bazel build --config macos_x86_64 plaidml:wheel plaidml/keras:wheel
    sudo pip install -U bazel-bin/plaidml/*whl bazel-bin/plaidml/keras/*whl

## Testing PlaidML

Unit tests are executed through bazel:

    bazel test ...

Unit tests for frontends are marked manual and must be executed individually (requires 
running `plaidml-setup` prior to execution)

    bazel test plaidml/keras:backend_test


[Bazel]:http://bazel.build
[PlaidML backend]:https://ngraph.nervanasys.com/docs/latest/programmable/index.html#plaidml
[nGraph build]:https://ngraph.nervanasys.com/docs/latest/buildlb.html 
