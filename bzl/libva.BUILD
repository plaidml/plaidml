package(default_visibility = ["@//visibility:public"])

cc_library(
    name = "inc",
    hdrs = glob(["va/*.h"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)

genrule(
    name = "configure-make",
    srcs = ["va/va.h"],
    outs = ["libva_placeholder"],
    cmd = "cd ../../external/libva\n" + "./configure\n" + "make\n" + "cd ../../execroot/com_intel_plaidml\n" + "cp $(SRCS) $@",
)
