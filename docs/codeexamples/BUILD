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
