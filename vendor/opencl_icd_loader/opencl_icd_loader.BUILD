package(default_visibility = ["//visibility:public"])

opencl_icd_loader_srcs = glob([
    "loader/*.c",
    "loader/*.h",
])

opencl_icd_loader_win_srcs = glob([
    "loader/windows/*.c",
    "loader/windows/*.cpp",
    "loader/windows/*.h",
])

opencl_icd_loader_win_hdrs = [
    "loader/windows/OpenCL.rc",
]

opencl_icd_loader_lnx_srcs = glob([
    "loader/linux/*.c",
])

opencl_icd_loader_lnx_hdrs = [
    "loader/linux/icd_exports.map.in",
]

opencl_icd_loader_srcs += select({
    "@bazel_tools//src/conditions:windows": opencl_icd_loader_win_srcs,
    "//conditions:default": opencl_icd_loader_lnx_srcs,
})

opencl_icd_loader_hdrs = select({
    "@bazel_tools//src/conditions:windows": opencl_icd_loader_win_hdrs,
    "//conditions:default": opencl_icd_loader_lnx_hdrs,
})

opencl_icd_loader_lnks = [
    "cfgmgr32.lib",
    "runtimeobject.lib",
    "Advapi32.lib",
    "ole32.lib",
]

cc_library(
    name = "opencl_icd_loader",
    srcs = opencl_icd_loader_srcs,
    hdrs = opencl_icd_loader_hdrs,
    includes = ["loader"],
    linkopts = opencl_icd_loader_lnks,
    deps = ["@opencl_headers"],
)
