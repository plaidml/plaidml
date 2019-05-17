package(default_visibility = ["@//visibility:public"])

load("@com_intel_plaidml//bzl:mlir.bzl", "mlir_tblgen")

PLATFORM_COPTS = select({
    "@toolchain//:macos_x86_64": [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
        "-w",
    ],
    "//conditions:default": [
        "-fPIC",
        "-w",
    ],
})

cc_binary(
    name = "mlir-tblgen",
    srcs = glob([
        "tools/mlir-tblgen/*.cpp",
        "lib/TableGen/**/*.cpp",
        "lib/TableGen/**/*.c",
        "lib/TableGen/**/*.h",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    linkopts = ["-lm"],
    deps = ["@llvm_shim//:llvm", "@llvm_shim//:tblgen-lib"],
)

mlir_tblgen(
    name = "gen-op-decls",
    src = "include/mlir/StandardOps/Ops.td",
    out = "include/mlir/StandardOps/Ops.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-op-defs",
    src = "include/mlir/StandardOps/Ops.td",
    out = "include/mlir/StandardOps/Ops.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

cc_library(
    name = "ir",
    hdrs = glob([
        "lib/Parser/**/*.def",
    ]) + [
        ":gen-op-decls",
    ],
    srcs = glob([
        "lib/Parser/**/*.h",
        "lib/Parser/**/*.cpp",
        "lib/Support/**/*.h",
        "lib/Support/**/*.cpp",
        "lib/IR/**/*.h",
        "lib/IR/**/*.cpp",
        "lib/StandardOps/**/*.h",
        "lib/StandardOps/**/*.cpp",
    ]) + [
        ":gen-op-defs",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = ["@llvm_shim//:llvm"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
