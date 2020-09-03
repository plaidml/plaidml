package(default_visibility = ["//visibility:public"])

# This file is used for detecting secure_getenv, in which case it will contain
# #define HAVE_SECURE_GETENV.
# For now leave blank to fallback to getenv.
genrule(
    name = "icd_cmake_config_gen",
    srcs = [],
    outs = ["loader/icd_cmake_config.h"],
    cmd = "echo '' > \"$@\"",
    cmd_ps = "echo '' > \"$@\"",
    visibility = ["//visibility:private"],
)

opencl_icd_loader_srcs = glob([
    "loader/*.c",
    "loader/*.h",
]) + [
    "loader/icd_cmake_config.h",
]

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

# Select configuration based on system
opencl_icd_loader_srcs += select({
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
    includes = ["loader"],
    linkopts = opencl_icd_loader_lnks,
    linkstatic = 1,
    deps = ["@opencl_headers"],
)
