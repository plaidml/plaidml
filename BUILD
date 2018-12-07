# Copyright 2018 Intel Corporation
#
# To build PlaidML for MacOS or Linux, it's easiest to start by
# installing Anaconda, from <https://www.anaconda.com/download>.
# You'll want to use a Python 3 version.
#
# Note that after installing Anaconda, you'll need to restart your
# shell, to pick up its environment variable modifications (i.e. the
# path to the conda tool and shell integrations).
#
# Once you have Anaconda installed, you can install the build tools by
# creating a Conda environemnt with them:
#   ) cd to the top-level PlaidML directory
#   ) Run "conda env create -n plaidml" to create the "plaidml" environment
#   ) Run "conda activate plaidml" to make "plaidml" your current environment
#
# To build PlaidML, run Bazel with the appropriate configuration setting:
#
#   ) MacOS: "bazel build --config=macos_x86_64 //..."
#   ) Linux: "bazel build --config=linux_x86_64 //..."
#
# We recommend adding a file named ".bazelrc" to your HOME directory,
# containing a line like "build --config=macos_x86_64", in order to
# automatically set the appropriate default configuration for your
# machine.

package(default_visibility = ["//visibility:public"])
