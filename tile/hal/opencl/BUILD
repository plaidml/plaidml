# Copyright 2017, Vertex.AI.

load("//bzl:plaidml.bzl", "plaidml_proto_library", "plaidml_cc_library", "plaidml_cc_test")

plaidml_proto_library(
    name = "proto",
    srcs = [
        "opencl.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/context:proto",
        "//tile/proto:hal",
    ],
)

OPENCL_1_0_IMPL = [
    "shared_memory_disabled.cc",
    "zero_kernel_svm_disabled.cc",
]

OPENCL_2_0_IMPL = [
    "shared_memory.cc",
    "zero_kernel_svm.cc",
]

plaidml_cc_library(
    name = "opencl",
    srcs = [
        "buffer.cc",
        "buffer.h",
        "cl_mem_arena.cc",
        "cl_mem_arena.h",
        "cl_mem_buffer.cc",
        "cl_mem_buffer.h",
        "cl_opt.cc",
        "cl_opt.h",
        "compiler.cc",
        "compiler.h",
        "device.cc",
        "device.h",
        "device_memory.cc",
        "device_memory.h",
        "device_set.cc",
        "device_set.h",
        "device_state.cc",
        "device_state.h",
        "driver.cc",
        "driver.h",
        "emitocl.cc",
        "event.cc",
        "event.h",
        "executor.cc",
        "executor.h",
        "exprtype.cc",
        "exprtype.h",
        "host_memory.cc",
        "host_memory.h",
        "info.cc",
        "kernel.cc",
        "kernel.h",
        "library.cc",
        "library.h",
        "ocl.cc",
        "ocl.h",
        "opencl.cc",
        "result.cc",
        "result.h",
        "zero_kernel.cc",
        "zero_kernel.h",
    ] + select({
        "//bzl:darwin": OPENCL_1_0_IMPL,
        "//conditions:default": OPENCL_2_0_IMPL,
    }),
    hdrs = [
        "emitocl.h",
        "info.h",
        "opencl.h",
    ],
    copts = select({
        "//bzl:arm7-ubuntu-14.04": [],
        "//bzl:x64_windows": [],
        "//conditions:default": ["-Wno-ignored-attributes"],
    }),
    linkopts = select({
        "//bzl:darwin": [
            "-framework OpenCL",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//base/util",
        "@boost_archive//:filesystem",
        "@boost_archive//:regex",
        "//tile/base",
        "//tile/base:hal",
        "//tile/hal/util:selector",
        "//tile/lang",
        "//tile/proto:proto_cc",
        "//tile/proto:support",
    ] + select({
        "//bzl:darwin": [],
        "//conditions:default": [
            "@opencl_icd_loader_repo//:lib",
        ],
    }),
    alwayslink = 1,
)

plaidml_cc_test(
    name = "exprtype_test",
    srcs = [
        "exprtype_test.cc",
    ],
    deps = [
        ":opencl",
    ],
)
