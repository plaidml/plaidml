load("//bzl:plaidml.bzl", "plaidml_cc_library")

exports_files([
    "cpu.json.j2",
    "cpu.yml",
])

plaidml_cc_library(
    name = "cpu",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    copts = [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
    ],
    tags = ["llvm"],
    visibility = ["//visibility:public"],
    deps = [
        "//tile/stripe",
        "@half",
        "@llvm_shim//:llvm",
    ],
)
