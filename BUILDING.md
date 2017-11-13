# Building PlaidML

PlaidML depends on [Bazel](http://bazel.build) v0.6.0 or higher.

## Linux


```
bazel build plaidml:wheel plaidml/keras:wheel
sudo pip install -U bazel-bin/plaidml/*whl bazel-bin/plaidml/keras/*whl
```

## macOS

The versions of bison and flex that are provided with xcode are too old to build PlaidML.
It's easiest to use homebrew to install all the prerequisites:

```
brew install bazel bison flex
```

Then, use bazel to build, sepecifying the correct config from tools/bazel.rc: 

```
bazel build plaidml:wheel plaidml/keras:wheel --config macos-10.12
sudo pip install -U bazel-bin/plaidml/*whl bazel-bin/plaidml/keras/*whl
```
