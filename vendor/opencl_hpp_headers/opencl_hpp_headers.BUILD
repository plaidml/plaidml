package(default_visibility = ["//visibility:public"])

cc_library(
    name = "opencl_hpp_headers",
    includes = ["include"],
    textual_hdrs = glob(["include/CL/*.hpp"]),
    deps = ["@opencl_headers"],
)
