load("//bzl:conda.bzl", "conda_test")
load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_py_library",
)

exports_files([
    "edsl.h",
    "ffi.h",
])

plaidml_cc_library(
    name = "ffi",
    srcs = ["ffi.cc"],
    hdrs = [
        "ffi.h",
        "internal.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml/base:api",
        "//tile/lang",
    ],
    alwayslink = 1,
    defines = ["PLAIDML_EDSL_DLL"],
)

plaidml_cc_library(
    name = "public",
    hdrs = [
        "edsl.h",
        "ffi.h",
    ],
    visibility = ["//visibility:public"],
    deps = ["//plaidml/base:api"],
)

plaidml_cc_library(
    name = "edsl",
    visibility = ["//visibility:public"],
    deps = [
        ":ffi",
        ":public",
    ],
)

plaidml_py_library(
    name = "py",
    srcs = [
        "__init__.py",
        "_ffi.py",
    ],
    visibility = ["//visibility:public"],
    deps = ["//plaidml:py"],
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
    deps = [":py"],
)
