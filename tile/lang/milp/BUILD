load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "milp",
    srcs = [
        "ilp_solver.cc",
        "ilp_solver.h",
        "tableau.cc",
        "tableau.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util:util_cc",
        "//external:gtest",
        "//tile/lang:math",
    ],
)

plaidml_cc_test(
    name = "test",
    srcs = [
        "test.cc",
    ],
    deps = [
        ":milp",
        "//external:gtest",
    ],
)
