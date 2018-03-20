# Copyright 2017, Vertex.AI. CONFIDENTIAL

load("//bzl:plaidml.bzl", "plaidml_proto_library", "plaidml_cc_library", "plaidml_cc_test")

plaidml_proto_library(
    name = "proto",
    srcs = [
        "cpu.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tile/proto:hal",
    ],
)

plaidml_cc_library(
    name = "cpu",
    srcs = [
        "arena.cc",
        "arena.h",
        "buffer.cc",
        "buffer.h",
        "compiler.cc",
        "compiler.h",
        "cpu.cc",
        "device.cc",
        "device.h",
        "device_set.cc",
        "device_set.h",
        "driver.cc",
        "driver.h",
        "emitllvm.cc",
        "emitllvm.h",
        "event.cc",
        "event.h",
        "executor.cc",
        "executor.h",
        "kernel.cc",
        "kernel.h",
        "library.cc",
        "library.h",
        "memory.cc",
        "memory.h",
        "result.cc",
        "result.h",
        "runtime.cc",
        "runtime.h",
    ],
    hdrs = [
        "cpu.h",
    ],
    copts = [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
    ] + select({
        "@toolchain//:linux_arm_32v7": [],
        "@toolchain//:linux_arm_64v8": [],
        "//conditions:default": ["-DENABLE_CPU_DEVICE"],
    }),
    linkopts = select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["-ldl"],
    }),
    tags = ["llvm"],
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//base/util",
        "//lib:llvm",
        "//tile/base",
        "//tile/base:hal",
        "//tile/hal/util:selector",
        "//tile/lang",
        "//tile/proto:proto_cc",
        "//tile/proto:support",
        "@half_repo//:half",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "llvm_test",
    srcs = ["llvm_test.cc"],
    copts = [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
    ],
    tags = ["llvm"],
    deps = [
        ":cpu",
        "//base/util",
        "//tile/base:platform_test",
    ],
)
