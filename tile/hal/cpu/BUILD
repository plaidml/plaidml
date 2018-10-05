# Copyright 2017-2018 Intel Corporation. 

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test")

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
        "executable.cc",
        "executable.h",
        "executor.cc",
        "executor.h",
        "library.cc",
        "library.h",
        "memory.cc",
        "memory.h",
        "result.cc",
        "result.h",
        "runtime.cc",
        "runtime.h",
    ],
    copts = [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
    ],
    linkopts = select({
        "@toolchain//:windows_x86_64": [],
        "//conditions:default": ["-ldl"],
    }),
    tags = ["llvm"],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//lib:llvm",
        "//tile/base",
        "//tile/base:hal",
        "//tile/hal/util:selector",
        "//tile/lang",
        "//tile/proto:proto_cc",
        "//tile/proto:support",
        "@half",
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
    ],
)

plaidml_cc_test(
    name = "platform_test",
    srcs = ["platform_test.cc"],
    tags = ["llvm"],
    deps = [
        ":cpu",
        "//tile/base:platform_test",
        "//tile/platform/local_machine",
    ],
)
