package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license

cc_library(
   name = "xsmm",
   srcs = ["include/libxsmm_source.h"],
   hdrs = glob(["include/*"]),
   includes=["include"],
)

genrule(
    name = "license",
    srcs = ["documentation/LICENSE.md"],
    outs = ["xsmm-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
