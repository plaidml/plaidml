# Troubleshooting

Having trouble getting PlaidML to work? Well, you're in the right place!

Before you open a new issue on GitHub, please
[take a look at the common issues](#common-issues),
[enable verbose logging in PlaidML](#enable-verbose-logging), and
[run backend tests](#run-backend-tests). These steps will help enable us to
provide you with better support on your issue.

## Common Issues

### PlaidML Setup Errors

### Memory Errors

`OSError: exception: access violation reading 0x0000000000000030`

This error might be caused by a memory allocation failure, and it fails
silently. You can fix this error by decreasing your batch size and trying again.

`plaidml.exceptions.ResourceExhausted: Out of memory` 

This error is caused by incorrect Tile syntax.

### Bazel Issues

For any Bazel-specific issues you're encountering, we recommend that you first
visit [Bazel's installation
documentation](https://docs.bazel.build/versions/master/install.html) which has
a comprehensive overview of Bazel on various platforms. Any issues commonly
encountered by PlaidML users are documented below.

`Encountered error while reading extension file 'workspace.bzl': no such package
'@toolchain//'`

On MacOS devices, `toolchain` errors often indicate that the user does not have
Xcode properly installed. Even if you have Xcode Command Line Tools installed,
you may not have a proper installation of Xcode itself.

To check your installation of Xcode, first print the path of the active
developer directory:

`xcode-select -p`

The resulting path should be `/Applications/Xcode.app/Contents/Developer`. If
that is not the path you are seeing when you run `xcode-select -p`, please go to
the App Store and download Xcode.

After verifying that Xcode is properly installed, you will need to reset your
Bazel instance before running Bazel again:

`bazelisk clean --expunge`

### PlaidML Exceptions

`Applying function, tensor with mismatching dimensionality`

This error may be caused by a known issue with the `BatchDot` operation, where 
results are inconsistent across backends. The [Keras documentation for 
BatchDot](https://keras.io/backend/#batch_dot) matches the Theano backend's 
implemented behavior and the _default_ behavior within PlaidML. The TensorFlow 
backend implements BatchDot in a different way, and this causes a mismatch in 
the expected output shape (there is an [open issue against 
TensorFlow](https://github.com/tensorflow/tensorflow/issues/30846) to get this 
fixed).

If you have existing Keras code that was written for the TensorFlow backend, 
and it is running into this issue, you can enable experimental support for 
TensorFlow-like `BatchDot` behavior by setting the environment variable 
`PLAIDML_BATCHDOT_TF_BEHAVIOR` to `True`.

`ERROR:plaidml:syntax error, unexpected -, expecting "," or )`

This error may be caused by special characters, such as `-`, that are used in
variable names within your code. Please try removing and/or replacing special
characters in your variable names, and try running again.

## Run Backend Tests

Backend Tests provide us with useful information that we can use to help solve
your issue. To run backend tests on PlaidML, follow these steps:

1. Verify that you have the PlaidML Python Wheel built as specified in
[building.md](building.md)
1. Run the backend tests through Bazel
```
bazel test --config macos_x86_64 @com_intel_plaidml//plaidml/keras:backend_test
```

## Enable Verbose Logging

You can enable verbose logging through the environment variable
`PLAIDML_VERBOSE`.

`PLAIDML_VERBOSE` should be set to an integer specifying the level of verbosity
(valid levels are 0-4 inclusive, where 0 is not verbose and 4 is the most
verbose).

For instance, the following command would set a verbosity level of 1.

```
export PLAIDML_VERBOSE=1
```
