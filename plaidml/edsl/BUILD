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
        "edsl.cc",
        "ffi.cc",
    ],
    hdrs = [
        "edsl.h",
        "ffi.h",
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
    srcs = ["edsl_test.cc"],
    deps = [":edsl"],
)

conda_test(
    name = "py_test",
    srcs = ["edsl_test.py"],
    env = "@com_intel_plaidml_conda//:env",
    main = "edsl_test.py",
    deps = [
        ":py",
    ],
)
