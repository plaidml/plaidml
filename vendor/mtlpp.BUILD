genrule(
    name = "mm2cc",
    srcs = ["mtlpp.mm"],
    outs = ["mtlpp.cc"],
    cmd = "cp $(location mtlpp.mm) $@",
)

cc_library(
    name = "mtlpp_cc",
    srcs = ["mtlpp.cc"],
    hdrs = ["mtlpp.hpp"],
    copts = [
        "-ObjC++",
        "-w",
    ],
    linkopts = ["-framework Metal"],
    visibility = ["//visibility:public"],
)

objc_library(
    name = "mtlpp_objc",
    hdrs = ["mtlpp.hpp"],
    non_arc_srcs = ["mtlpp.mm"],
    sdk_frameworks = ["Metal"],
    visibility = ["//visibility:public"],
)
