load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "math",
    srcs = [
        "basis.cc",
        "bignum.cc",
        "matrix.cc",
        "polynomial.cc",
        "util.cc",
    ],
    hdrs = [
        "basis.h",
        "bignum.h",
        "matrix.h",
        "polynomial.h",
        "util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//pmlc/util",
        "@boost",
        "@llvm-project//llvm:support",
    ],
)

# plaidml_cc_test(
#     name = "test",
#     srcs = [
#         "math_test.cc",
#         "test.cc",
#     ],
#     tags = ["windows_fail"],
#     deps = [
#         ":math",
#     ],
# )
