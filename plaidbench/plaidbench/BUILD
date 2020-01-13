# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

py_library(
    name = "plaidbench",
    srcs = glob(["**/*.py"]),
    data = glob([
        "cifar16.npy",
        "imdb16.npy",
        "golden/**/*.npy",
        "networks/**/*.h5",
    ]),
    deps = [
        "//plaidml/bridge/keras",
    ],
)
