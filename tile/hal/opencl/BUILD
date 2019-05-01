# Copyright 2017-2018 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test", "plaidml_proto_library")

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
        "compute_kernel.cc",
        "compute_kernel.h",
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
        "executable.cc",
        "executable.h",
        "executor.cc",
        "executor.h",
        "host_memory.cc",
        "host_memory.h",
        "info.cc",
        "kernel.h",
        "library.cc",
        "library.h",
        "ocl.cc",
        "ocl.h",
        "opencl.cc",
        "result.cc",
        "result.h",
        "shared_memory.cc",
        "zero_kernel.cc",
        "zero_kernel.h",
    ],
    hdrs = [
        "emitocl.h",
        "info.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//base/shim/opencl",
        "//base/util",
        "@boost//:filesystem",
        "@boost//:regex",
        "//tile/base",
        "//tile/base:hal",
        "//tile/hal/util:selector",
        "//tile/lang",
        "//tile/proto:proto_cc",
        "//tile/proto:support",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "platform_test",
    srcs = ["platform_test.cc"],
    deps = [
        ":opencl",
        "//tile/base:platform_test",
        "//tile/platform/local_machine",
    ],
)
