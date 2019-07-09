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

### PlaidML Exceptions

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

If you are importing the `plaidml` package in your code, you can enable verbose
logging with the function `plaidml_set_internal_vlog`. The function takes a
single integer argument specifying the level of verbosity (available levels are
1-4 inclusive).
