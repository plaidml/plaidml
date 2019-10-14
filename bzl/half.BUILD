package(default_visibility = ["@//visibility:public"])

exports_files(["LICENSE.txt"])

cc_library(
    name = "half",
    hdrs = ["include/half.hpp"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "license",
    srcs = ["LICENSE.txt"],
    outs = ["half-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
