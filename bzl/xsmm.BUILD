package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license

cc_library(
   name = "xsmm",
   srcs = [ ],
   hdrs = glob(["**"]),
)

genrule(
    name = "license",
    srcs = ["documentation/LICENSE.md"],
    outs = ["xsmm-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
