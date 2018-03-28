# Building and Testing PlaidML

PlaidML depends on [Bazel](http://bazel.build) v0.11.0 or higher.

## Building PlaidML

### Install Requirements
pip install -r requirements.txt

### Linux

```
bazel build --config linux_x86_64 plaidml:wheel plaidml/keras:wheel
sudo pip install -U bazel-bin/plaidml/*whl bazel-bin/plaidml/keras/*whl
```

### macOS

The versions of bison and flex that are provided with xcode are too old to build PlaidML.
It's easiest to use homebrew to install all the prerequisites:

```
brew install bazel bison flex
```

Then, use bazel to build, sepecifying the correct config from tools/bazel.rc: 

```
bazel build --config macos_x86_64 plaidml:wheel plaidml/keras:wheel
sudo pip install -U bazel-bin/plaidml/*whl bazel-bin/plaidml/keras/*whl
```

## Testing PlaidML

Unit tests are executed through bazel:

```
bazel test ...
```

Unit tests for frontends are marked manual and must be executed individually (requires running `plaidml-setup` prior to execution)

```
bazel test plaidml/keras:backend_test
```



