load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
)

plaidml_cc_test(
    name = "test",
    srcs = [
        "math_test.cc",
        "test.cc",
    ],
    deps = [
        ":math",
    ],
)

plaidml_cc_library(
    name = "math",
    srcs = [
        "basis.cc",
        "bignum.cc",
        "matrix.cc",
        "polynomial.cc",
    ],
    hdrs = [
        "basis.h",
        "bignum.h",
        "matrix.h",
        "polynomial.h",
    ],
    copts = select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["-Wno-error"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "@boost",
    ],
)

