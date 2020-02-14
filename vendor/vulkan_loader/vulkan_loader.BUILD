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

config_setting(
    name = "ASSEMBLER_WORKS",
    values = {
        "define": "MODE=ASM",
    },
)

#for asm path
cc_binary(
    name = "asm_offset",
    srcs = [
        "loader/asm_offset.c",
    ],
    includes = [
        "loader",
        "loader/generated",
    ],
    deps = [
        "@vulkan_headers//:inc",
    ],
)

#shall use asm_offset to create gen_defines.asm
genrule(
    name = "gen_asm",
    outs = ["gen_defines.asm"],
    cmd = "\n".join([
        " $(location :asm_offset) GAS ",
        " cp gen_defines.asm $@ ",
    ]),
    tools = [
        ":asm_offset",
    ],
)

cc_library(
    name = "vulkan_loader",
    srcs = [
        ":NORMAL_LOADER_SRCS",
        ":OPT_LOADER_SRCS",
    ] + select({
        ":ASSEMBLER_WORKS": ["loader/unknown_ext_chain_gas.S"],
        "//conditions:default": [":OPT_LOADER_SRCS_NO_ASM"],
    }),
    hdrs = [
        "loader/generated/vk_loader_extensions.c",
    ] + select({
        "ASSEMBLER_WORKS": [":gen_asm"],
        "//conditions:default": [],
    }),
    copts = [
        "-I loader",
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
