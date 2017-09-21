load("//bzl:plaidml.bzl", "plaidml_proto_library", "plaidml_cc_library", "plaidml_cc_test")

plaidml_proto_library(
    name = "proto",
    srcs = [
        "local_machine.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tile/proto:hal",
    ],
)

plaidml_cc_library(
    name = "local_machine",
    srcs = [
        "buffer.cc",
        "buffer.h",
        "copy_mem_strategy.cc",
        "copy_mem_strategy.h",
        "devinfo.h",
        "direct_mem_strategy.cc",
        "direct_mem_strategy.h",
        "factory.cc",
        "mem_cache.cc",
        "mem_cache.h",
        "mem_chunk.h",
        "mem_deps.cc",
        "mem_deps.h",
        "mem_strategy.h",
        "platform.cc",
        "platform.h",
        "program.cc",
        "program.h",
        "tmp_mem_strategy.cc",
        "tmp_mem_strategy.h",
    ],
    hdrs = [
        "local_machine.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":proto_cc",
        "//tile/base",
        "//tile/base:hal",
        "//tile/hal/util:selector",
        "//tile/hal/util:settings",
        "//tile/proto:hal_cc",
        "//tile/proto:support",
    ],
    alwayslink = 1,
)

plaidml_cc_test(
    name = "platform_test",
    srcs = ["platform_test.cc"],
    data = ["//testing:configs"],
    deps = [
        ":local_machine",
        "//base/config",
        "//plaidml:proto_cc",
        "//testing:plaidml_config",
        "//tile/base:platform_test",
        "//tile/hal/opencl",
    ],
)
