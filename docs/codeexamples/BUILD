load("@rules_python//python:defs.bzl", "py_binary")
load("//bzl:plaidml.bzl", "plaidml_cc_test")

filegroup(
    name = "code_files",
    srcs = [
        "writing_edsl.cc",
        "writing_edsl.py",
    ],
    visibility = ["//visibility:public"],
)

plaidml_cc_test(
    name = "cc_writing_edsl",
    srcs = ["writing_edsl.cc"],
    deps = [
        "//plaidml:testenv",
        "//plaidml/edsl",
        "//plaidml/edsl:api",
        "//plaidml/exec",
    ],
)

py_test(
    name = "py_writing_edsl",
    srcs = ["writing_edsl.py"],
    main = "writing_edsl.py",
    deps = [
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)

py_binary(
    name = "monte_carlo_pi",
    srcs = ["monte_carlo_pi.py"],
    deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)

py_test(
    name = "monte_carlo_pi_tests",
    srcs = [
        "monte_carlo_pi.py",
        "monte_carlo_pi_tests.py",
    ],
    main = "monte_carlo_pi_tests.py",
    deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
    ],
)
