# Copyright 2017-2018 Intel Corporation.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test", "plaidml_proto_library")
load("@cm//:build_defs.bzl", "if_cm_is_configured")

plaidml_proto_library(
    name = "proto",
    srcs = [
        "cm.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/context:proto",
        "//tile/proto:hal",
    ],
)

plaidml_cc_library(
    name = "cm",
    srcs = if_cm_is_configured([
    	"buffer.cc",
        "buffer.h",
        "mem_arena.cc",
        "mem_arena.h",
        "mem_buffer.cc",
        "mem_buffer.h",
        "compiler.cc",
        "compiler.h",
        "compute_kernel.cc",
        "compute_kernel.h",
        "device.cc",
        "device.h",
        "device_set.cc",
        "device_set.h",
        "device_state.cc",
        "device_state.h",
        "driver.cc",
        "driver.h",
        "event.cc",
        "event.h",
        "executable.cc",
        "executable.h",
        "executor.cc",
        "executor.h",
        "host_memory.cc",
        "host_memory.h",
        "result.cc",
        "result.h", 
        "library.cc",
        "library.h", 
        "info.cc",   
        "info.h",
        "err.cc", 
        "err.h", 
        "zero_kernel.cc",
        "zero_kernel.h",
        "travel.h",
        "travel.cc",
        "emitcm.cc",
        "emitcm.h",
        "kernel.h",
        "cm.cc",
        "shared_memory.cc",
    ]),
    hdrs = [
    	"runtime.h",
    ]+ if_cm_is_configured([
        "@libva//:configure-make",
    ]),
    copts = [
  		'-Wno-overflow',
	],
	linkopts=if_cm_is_configured([
		'-rdynamic external/cm_headers/runtime/lib/x64/libigfxcmrt.so',
		'-rdynamic external/libva/va/.libs/libva.so.2',
	]),
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//base/util",
        "@boost//:filesystem",
        "@boost//:regex",
        "//tile/base",
        "//tile/base:hal",
        "//tile/hal/util:selector",
        "//tile/lang",
        "//tile/proto:proto_cc",
        "//tile/proto:support",
    ]+ if_cm_is_configured([
        "@cm_headers//:inc",
        "@libva//:inc",
    ]),
    alwayslink = 1,
)

plaidml_cc_test(
    name = "platform_test",
    srcs = if_cm_is_configured(["platform_test.cc"]),
    deps = [
        ":cm",
        "//tile/base:platform_test",
        "//tile/platform/local_machine",
    ],
)
