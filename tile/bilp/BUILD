load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

plaidml_cc_library(
    name = "bilp",
    srcs = [
        "ilp_solver.cc",
        "ilp_solver.h",
        "tableau.cc",
        "tableau.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//tile/math",
        "@gmock//:gtest",
    ],
)

plaidml_cc_test(
    name = "test",
    srcs = [
        "test.cc",
    ],
    # tags = ["large"],
    timeout = "eternal",
    deps = [
        ":bilp",
    ],
)
