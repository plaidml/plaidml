cc_import(
    name = "llvm_base",
    static_library = select({
        "@toolchain//:macos_x86_64": "macos_x86_64/libllvm_base.a",
        "@toolchain//:linux_x86_64": "linux_x86_64/libllvm_base.a",
        "//conditions:default": "fake.a",
    }),
    alwayslink = 1,
)

cc_import(
    name = "llvm_lib",
    static_library = select({
        "@toolchain//:macos_x86_64": "macos_x86_64/libllvm_lib.a",
        "@toolchain//:linux_x86_64": "linux_x86_64/libllvm_lib.a",
        "//conditions:default": "fake.a",
    }),
    alwayslink = 1,
)

cc_import(
    name = "llvm_targets",
    static_library = select({
        "@toolchain//:macos_x86_64": "macos_x86_64/libllvm_targets.a",
        "@toolchain//:linux_x86_64": "linux_x86_64/libllvm_targets.a",
        "//conditions:default": "fake.a",
    }),
    alwayslink = 1,
)

cc_library(
    name = "llvm",
    tags = ["llvm"],
    visibility = ["//visibility:public"],
    deps = select({
        "@toolchain//:macos_x86_64": [
            ":llvm_base",
            ":llvm_lib",
            ":llvm_targets",
            "@llvm_archive//:llvm_inc",
        ],
        "//conditions:default": ["@llvm_archive//:llvm"],
    }),
)
