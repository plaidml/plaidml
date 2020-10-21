load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# This file contains cmake-generated definitions.  We use local defines on the cc_library rule instead.
write_file(
    name = "icd_cmake_config_gen",
    out = "loader/icd_cmake_config.h",
    content = [],
    visibility = ["//visibility:private"],
)

# Linux specific configuration
opencl_icd_loader_lnx_srcs = glob([
    "loader/linux/*.c",
])

opencl_icd_loader_lnx_lnks = [
    "-ldl",
    "-lpthread",
]

# Windows specific configuration
opencl_icd_loader_win_srcs = glob([
    "loader/windows/*.c",
    "loader/windows/*.cpp",
    "loader/windows/*.h",
])

opencl_icd_loader_win_lnks = [
    "cfgmgr32.lib",
    "runtimeobject.lib",
    "Advapi32.lib",
    "ole32.lib",
]

opencl_icd_loader_srcs = glob([
    "loader/*.c",
    "loader/*.h",
]) + [
    "loader/icd_cmake_config.h",
] + select({
    "@bazel_tools//src/conditions:windows": opencl_icd_loader_win_srcs,
    "//conditions:default": opencl_icd_loader_lnx_srcs,
})

opencl_icd_loader_lnks = select({
    "@bazel_tools//src/conditions:windows": opencl_icd_loader_win_lnks,
    "//conditions:default": opencl_icd_loader_lnx_lnks,
})

cc_library(
    name = "opencl_icd_loader",
    srcs = opencl_icd_loader_srcs,
    copts = ["-w"],
    defines = ["CL_TARGET_OPENCL_VERSION=220"],
    includes = ["loader"],
    linkopts = opencl_icd_loader_lnks,
    linkstatic = 1,
    local_defines = select({
        "@com_intel_plaidml//:clang": ["HAVE_SECURE_GETENV"],
        "//conditions:default": [],
    }),
    deps = ["@opencl_headers"],
)
