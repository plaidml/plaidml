cc_library(
    name = "llvm",
    srcs = select({
#        "@toolchain//:macos_x86_64": [
#            ":lib/macos_x86_64/liblib.lo",
#            ":lib/macos_x86_64/libsupport.lo",
#            ":lib/macos_x86_64/libtargets.lo",
#        ],
        "//conditions:default": [],
    }),
    tags = ["llvm"],
    visibility = ["//visibility:public"],
    deps = select({
#        "@toolchain//:macos_x86_64": ["@llvm//:inc"],
        "//conditions:default": ["@llvm"],
    }),
)
