load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_py_library",
)

# plaidml_cc_library(
#     name = "op",
#     hdrs = [
#         "__init__.h",
#     ],
#     visibility = ["//visibility:public"],
# )

plaidml_py_library(
    name = "py",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml2:py",
    ],
)
