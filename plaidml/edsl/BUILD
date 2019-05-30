load("//bzl:conda.bzl", "conda_test")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_py_library",
)

plaidml_cc_library(
    name = "edsl",
    srcs = [
        "ffi.cc",
        "tile.cc",
    ],
    hdrs = [
        "tile.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//tile/lang",
    ],
    alwayslink = 1,
)

plaidml_py_library(
    name = "py",
    srcs = [
        "__init__.py",
        "_ffi.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml:py",
    ],
)

plaidml_cc_test(
    name = "cc_test",
    srcs = ["tile_test.cc"],
    deps = [":edsl"],
)

conda_test(
    name = "py_test",
    srcs = ["test.py"],
    env = "@com_intel_plaidml_conda//:env",
    main = "test.py",
    deps = [
        ":py",
    ],
)
