load("@rules_cc//cc:defs.bzl", "cc_toolchain")
load(":build_defs.bzl", "gcc_toolchain_config")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "empty",
    srcs = [],
)

cc_toolchain(
    name = "cc-linux_x86_64-gcc_8.3.0",
    all_files = "//toolchain/crosstool_ng/linux_x86_64-gcc_8.3.0:all_files",
    compiler_files = "//toolchain/crosstool_ng/linux_x86_64-gcc_8.3.0:compiler_files",
    dwp_files = ":empty",
    linker_files = "//toolchain/crosstool_ng/linux_x86_64-gcc_8.3.0:linker_files",
    objcopy_files = "//toolchain/crosstool_ng/linux_x86_64-gcc_8.3.0:objcopy",
    strip_files = "//toolchain/crosstool_ng/linux_x86_64-gcc_8.3.0:strip",
    supports_param_files = 1,
    tags = ["linux"],
    toolchain_config = ":cc-linux_x86_64-gcc_8.3.0-config",
    toolchain_identifier = "gcc-linux_x86_64",
)

gcc_toolchain_config(
    name = "cc-linux_x86_64-gcc_8.3.0-config",
    libc_version = "glibc_2.12.1",
    tags = ["linux"],
    target = "linux_x86_64",
    version = "8.3.0",
)

toolchain(
    name = "cc-toolchain-linux",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    tags = ["linux"],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    toolchain =
        select({
            "//:gcc": ":cc-linux_x86_64-gcc_8.3.0",
            "//:clang": "@llvm_toolchain//:cc-clang-linux",
            "//conditions:default": ":cc-linux_x86_64-gcc_8.3.0",
        }),
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
