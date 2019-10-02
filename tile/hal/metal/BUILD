load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
)

# objc_library has many problems:
#   - linker errors
#   - deps aren't transitive (to other cc_librarys that depend on an objc_library)

plaidml_cc_library(
    name = "metal",
    srcs = [
        "emit.cc",
        "hal.cc",
        "hal.h",
    ],
    copts = [
        "-ObjC++",
        "-fobjc-arc",
    ],
    linkopts = ["-framework Metal"],
    tags = ["metal"],
    visibility = ["//visibility:public"],
    deps = [
        "//tile/base:hal",
        "//tile/hal/opencl:proto_cc",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "platform_test",
    srcs = ["platform_test.cc"],
    tags = [
        "metal",
        "not_ci",
    ],
    deps = [
        ":metal",
        "//tile/base:platform_test",
        "//tile/platform/local_machine",
    ],
)
