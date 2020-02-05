package(default_visibility = ["//visibility:public"])

#load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_binary")

filegroup(
    name = "NORMAL_LOADER_SRCS",
    srcs = [
        "loader/cJSON.c",
        "loader/debug_utils.c",
        "loader/extension_manual.c",
        "loader/loader.c",
        "loader/murmurhash.c",
        "loader/trampoline.c",
        "loader/wsi.c",
    ],
)

filegroup(
    name = "OPT_LOADER_SRCS",
    srcs = [
        "loader/dev_ext_trampoline.c",
        "loader/phys_dev_ext.c",
    ],
)

#for no asm path
filegroup(
    name = "OPT_LOADER_SRCS_NO_ASM",
    srcs = [
        "loader/unknown_ext_chain.c",
    ],
)

#for asm path
cc_binary(
    name = "asm_offset",
    srcs = [
        "asm_offset.c",
    ],
)

#shall use asm_offset to create gen_defines.asm
#genrule(
#    name = "gen_asm",
#    srcs = [
#        ":asm_offset",
#    ],
#    outputs = ["gen_defines.asm"],
#    cmd = "asm_offset",
#)

cc_library(
    name = "vulkan1",
    srcs = [
        ":NORMAL_LOADER_SRCS",
        ":OPT_LOADER_SRCS",
        ":OPT_LOADER_SRCS_NO_ASM",
    ],
    hdrs = [
        "loader/generated/vk_loader_extensions.c",
    ],
    includes = [
        "loader",
        "loader/generated",
    ],
    linkopts = [
        "-lpthread",
        "-lm",
    ],
    deps = [
        "@vulkan_headers//:inc",
    ],
)
