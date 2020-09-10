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
        "//pmlc/util",
        "//pmlc/util/math",
        "@boost",
        "@com_google_googletest//:gtest",
    ],
)

plaidml_cc_test(
    name = "test",
    # tags = ["large"],
    timeout = "eternal",
    srcs = [
        "test.cc",
    ],
    deps = [
        ":bilp",
    ],
)
