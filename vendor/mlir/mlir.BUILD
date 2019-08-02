package(default_visibility = ["@//visibility:public"])

load("@com_intel_plaidml//vendor/mlir:mlir.bzl", "mlir_tblgen")

PLATFORM_COPTS = select({
    "@com_intel_plaidml//toolchain:macos_x86_64": [
        "-D__STDC_LIMIT_MACROS",
        "-D__STDC_CONSTANT_MACROS",
        "-w",
    ],
    "@com_intel_plaidml//toolchain:windows_x86_64": [
        "/w",
        "/wd4244",
        "/wd4267",
    ],
    "//conditions:default": [
        "-fPIC",
        "-w",
    ],
})

cc_library(
    name = "TableGen",
    srcs = glob([
        "lib/TableGen/**/*.cpp",
        "lib/TableGen/**/*.h",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm//:support",
        "@llvm//:tablegen",
    ],
)

cc_binary(
    name = "mlir-tblgen",
    srcs = glob([
        "tools/mlir-tblgen/*.cpp",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    linkopts = select({
        "@com_intel_plaidml//toolchain:windows_x86_64": [],
        "@com_intel_plaidml//toolchain:macos_x86_64": [],
        "//conditions:default": [
            "-pthread",
            "-ldl",
            "-lm",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":Support",
        ":TableGen",
    ],
)

mlir_tblgen(
    name = "gen-standard-op-decls",
    src = "include/mlir/StandardOps/Ops.td",
    out = "include/mlir/StandardOps/Ops.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-standard-op-defs",
    src = "include/mlir/StandardOps/Ops.td",
    out = "include/mlir/StandardOps/Ops.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-affine-op-decls",
    src = "include/mlir/AffineOps/AffineOps.td",
    out = "include/mlir/AffineOps/AffineOps.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-affine-op-defs",
    src = "include/mlir/AffineOps/AffineOps.td",
    out = "include/mlir/AffineOps/AffineOps.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-loop-op-decls",
    src = "include/mlir/Dialect/LoopOps/LoopOps.td",
    out = "include/mlir/Dialect/LoopOps/LoopOps.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-loop-op-defs",
    src = "include/mlir/Dialect/LoopOps/LoopOps.td",
    out = "include/mlir/Dialect/LoopOps/LoopOps.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-llvm-op-decls",
    src = "include/mlir/LLVMIR/LLVMOps.td",
    out = "include/mlir/LLVMIR/LLVMOps.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-llvm-enum-decls",
    src = "include/mlir/LLVMIR/LLVMOps.td",
    out = "include/mlir/LLVMIR/LLVMOpsEnums.h.inc",
    action = "-gen-enum-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-llvm-op-defs",
    src = "include/mlir/LLVMIR/LLVMOps.td",
    out = "include/mlir/LLVMIR/LLVMOps.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-llvm-enum-defs",
    src = "include/mlir/LLVMIR/LLVMOps.td",
    out = "include/mlir/LLVMIR/LLVMOpsEnums.cpp.inc",
    action = "-gen-enum-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-llvm-conversions",
    src = "include/mlir/LLVMIR/LLVMOps.td",
    out = "include/mlir/LLVMIR/LLVMConversions.inc",
    action = "-gen-llvmir-conversions",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-nvvm-op-decls",
    src = "include/mlir/LLVMIR/NVVMOps.td",
    out = "include/mlir/LLVMIR/NVVMOps.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-nvvm-op-defs",
    src = "include/mlir/LLVMIR/NVVMOps.td",
    out = "include/mlir/LLVMIR/NVVMOps.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-nvvm-conversions",
    src = "include/mlir/LLVMIR/NVVMOps.td",
    out = "include/mlir/LLVMIR/NVVMConversions.inc",
    action = "-gen-llvmir-conversions",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-gpu-op-decls",
    src = "include/mlir/GPU/GPUOps.td",
    out = "include/mlir/GPU/GPUOps.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-gpu-op-defs",
    src = "include/mlir/GPU/GPUOps.td",
    out = "include/mlir/GPU/GPUOps.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

cc_library(
    name = "StandardOps",
    srcs = glob([
        "lib/StandardOps/**/*.h",
        "lib/StandardOps/**/*.cpp",
    ]) + [
        ":gen-standard-op-defs",
    ],
    hdrs = [
        ":gen-standard-op-decls",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        "@llvm//:core",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "Translation",
    srcs = glob([
        "lib/Translation/*.cpp",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        "@llvm//:support",
    ],
)

cc_library(
    name = "EDSC",
    srcs = glob([
        "lib/EDSC/*.cpp",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":StandardOps",
        ":TransformUtils",
        ":VectorOps",
    ],
)

cc_library(
    name = "Support",
    srcs = glob([
        "lib/Support/FileUtilities.cpp",
        "lib/Support/StorageUniquer.cpp",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        "@llvm//:core",
        "@llvm//:support",
    ],
)

cc_library(
    name = "OptMain",
    srcs = glob([
        "lib/Support/MlirOptMain.cpp",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        "@llvm//:support",
    ],
)

cc_library(
    name = "LLVMIR",
    srcs = glob([
        "lib/LLVMIR/IR/LLVMDialect.cpp",
    ]) + [
        ":gen-llvm-op-defs",
        ":gen-llvm-enum-defs",
        # ":gen-llvm-conversions",
    ],
    hdrs = [
        ":gen-llvm-enum-decls",
        ":gen-llvm-op-decls",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        "@llvm//:asm_parser",
        "@llvm//:core",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "NVVMIR",
    srcs = glob([
        "lib/LLVMIR/IR/NVVMDialect.cpp",
    ]) + [
        ":gen-nvvm-op-defs",
    ],
    hdrs = [
        ":gen-nvvm-op-decls",
        # ":gen-nvvm-conversions",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        "@llvm//:asm_parser",
        "@llvm//:core",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "TargetLLVMIRModuleTranslation",
    srcs = glob([
        "lib/Target/LLVMIR/ModuleTranslation.cpp",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":LLVMIR",
        ":Translation",
        "@llvm//:asm_parser",
        "@llvm//:core",
        "@llvm//:support",
        "@llvm//:transform_utils",
    ],
)

cc_library(
    name = "TargetLLVMIR",
    srcs = glob([
        "lib/Target/LLVMIR/ConvertToLLVMIR.cpp",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":TargetLLVMIRModuleTranslation",
    ],
)

cc_library(
    name = "GPU",
    srcs = glob([
        "lib/GPU/**/*.cpp",
    ]) + [
        ":gen-gpu-op-defs",
    ],
    hdrs = [
        ":gen-gpu-op-decls",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":IR",
        ":StandardOps",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "TargetNVVMIR",
    srcs = glob([
        "lib/Target/LLVMIR/ConvertToNVVMIR.cpp",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":GPU",
        ":IR",
        ":NVVMIR",
        ":TargetLLVMIRModuleTranslation",
    ],
)

cc_library(
    name = "IR",
    srcs = glob([
        "lib/IR/*.cpp",
        "lib/IR/*.h",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":Support",
        "@llvm//:support",
    ],
)

cc_library(
    name = "AffineOps",
    srcs = glob([
        "lib/AffineOps/*.cpp",
        "lib/AffineOps/*.h",
    ]) + [
        ":gen-affine-op-defs",
    ],
    hdrs = [
        ":gen-affine-op-decls",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":IR",
        ":StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "Analysis",
    srcs = glob([
        "lib/Analysis/*.cpp",
        "lib/Analysis/*.h",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":LoopOps",
    ],
)

cc_library(
    name = "Dialect",
    srcs = glob([
        "lib/Dialect/*.cpp",
        "lib/Dialect/*.h",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":IR",
    ],
)

cc_library(
    name = "LoopOps",
    srcs = glob([
        "lib/Dialect/LoopOps/*.cpp",
        "lib/Dialect/LoopOps/*.h",
    ]) + [
        ":gen-loop-op-defs",
    ],
    hdrs = [
        ":gen-loop-op-decls",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":StandardOps",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "Pass",
    srcs = glob([
        "lib/Pass/*.cpp",
        "lib/Pass/*.h",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        "@llvm//:support",
    ],
)

cc_library(
    name = "VectorOps",
    srcs = glob([
        "lib/VectorOps/*.cpp",
        "lib/VectorOps/*.h",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "Transforms",
    srcs = glob([
        "lib/Transforms/*.cpp",
        "lib/Transforms/*.h",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":Analysis",
        ":EDSC",
        ":LoopOps",
        ":Pass",
        ":TransformUtils",
        ":VectorOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "TransformUtils",
    srcs = glob([
        "lib/Transforms/Utils/*.cpp",
        "lib/Transforms/Utils/*.h",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":Analysis",
        ":LoopOps",
        ":Pass",
        ":StandardOps",
    ],
)

cc_library(
    name = "Parser",
    srcs = glob([
        "lib/Parser/*.cpp",
        "lib/Parser/*.h",
    ]),
    hdrs = glob([
        "lib/Parser/**/*.def",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
    ],
)

cc_library(
    name = "TestTransforms",
    srcs = glob([
        "test/lib/Transforms/**/*.cpp",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":AffineOps",
        ":Analysis",
        ":LoopOps",
        ":Pass",
        ":TransformUtils",
        ":VectorOps",
    ],
    alwayslink = 1,
)
