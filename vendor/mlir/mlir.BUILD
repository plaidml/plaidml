package(default_visibility = ["@//visibility:public"])

load("@com_intel_plaidml//vendor/mlir:mlir.bzl", "mlir_tblgen")

exports_files(["LICENSE.TXT"])

PLATFORM_COPTS = select({
    "@com_intel_plaidml//toolchain:macos_x86_64": [
        "-std=c++14",
        "-w",
    ],
    "@com_intel_plaidml//toolchain:windows_x86_64": [
        "/w",
        "/DWIN32_LEAN_AND_MEAN",
        "/std:c++17",  # This MUST match all other compilation units
    ],
    "//conditions:default": [
        "-std=c++14",
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
        "tools/mlir-tblgen/*.h",
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
    name = "gen-call-interfaces-decls",
    src = "include/mlir/Analysis/CallInterfaces.td",
    out = "include/mlir/Analysis/CallInterfaces.h.inc",
    action = "-gen-op-interface-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-call-interfaces-defs",
    src = "include/mlir/Analysis/CallInterfaces.td",
    out = "include/mlir/Analysis/CallInterfaces.cpp.inc",
    action = "-gen-op-interface-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-infer-type-op-interface-decls",
    src = "include/mlir/Analysis/InferTypeOpInterface.td",
    out = "include/mlir/Analysis/InferTypeOpInterface.h.inc",
    action = "-gen-op-interface-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-infer-type-op-interface-defs",
    src = "include/mlir/Analysis/InferTypeOpInterface.td",
    out = "include/mlir/Analysis/InferTypeOpInterface.cpp.inc",
    action = "-gen-op-interface-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-standard-op-decls",
    src = "include/mlir/Dialect/StandardOps/Ops.td",
    out = "include/mlir/Dialect/StandardOps/Ops.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-standard-op-defs",
    src = "include/mlir/Dialect/StandardOps/Ops.td",
    out = "include/mlir/Dialect/StandardOps/Ops.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-affine-op-decls",
    src = "include/mlir/Dialect/AffineOps/AffineOps.td",
    out = "include/mlir/Dialect/AffineOps/AffineOps.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-affine-op-defs",
    src = "include/mlir/Dialect/AffineOps/AffineOps.td",
    out = "include/mlir/Dialect/AffineOps/AffineOps.cpp.inc",
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
    src = "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    out = "include/mlir/Dialect/LLVMIR/LLVMOps.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-llvm-enum-decls",
    src = "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    out = "include/mlir/Dialect/LLVMIR/LLVMOpsEnums.h.inc",
    action = "-gen-enum-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-llvm-op-defs",
    src = "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    out = "include/mlir/Dialect/LLVMIR/LLVMOps.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-llvm-enum-defs",
    src = "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    out = "include/mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc",
    action = "-gen-enum-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-llvm-conversions",
    src = "include/mlir/Dialect/LLVMIR/LLVMOps.td",
    out = "include/mlir/Dialect/LLVMIR/LLVMConversions.inc",
    action = "-gen-llvmir-conversions",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-nvvm-op-decls",
    src = "include/mlir/Dialect/LLVMIR/NVVMOps.td",
    out = "include/mlir/Dialect/LLVMIR/NVVMOps.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-nvvm-op-defs",
    src = "include/mlir/Dialect/LLVMIR/NVVMOps.td",
    out = "include/mlir/Dialect/LLVMIR/NVVMOps.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-nvvm-conversions",
    src = "include/mlir/Dialect/LLVMIR/NVVMOps.td",
    out = "include/mlir/Dialect/LLVMIR/NVVMConversions.inc",
    action = "-gen-llvmir-conversions",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-gpu-op-decls",
    src = "include/mlir/Dialect/GPU/GPUOps.td",
    out = "include/mlir/Dialect/GPU/GPUOps.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-gpu-op-defs",
    src = "include/mlir/Dialect/GPU/GPUOps.td",
    out = "include/mlir/Dialect/GPU/GPUOps.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-vector-op-decls",
    src = "include/mlir/Dialect/VectorOps/VectorOps.td",
    out = "include/mlir/Dialect/VectorOps/VectorOps.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-vector-op-defs",
    src = "include/mlir/Dialect/VectorOps/VectorOps.td",
    out = "include/mlir/Dialect/VectorOps/VectorOps.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-loop-like-interface-decls",
    src = "include/mlir/Transforms/LoopLikeInterface.td",
    out = "include/mlir/Transforms/LoopLikeInterface.h.inc",
    action = "-gen-op-interface-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-loop-like-interface-defs",
    src = "include/mlir/Transforms/LoopLikeInterface.td",
    out = "include/mlir/Transforms/LoopLikeInterface.cpp.inc",
    action = "-gen-op-interface-defs",
    incs = ["include"],
)

cc_library(
    name = "StandardOps",
    srcs = glob([
        "lib/Dialect/StandardOps/*.cpp",
        "lib/Dialect/StandardOps/*.h",
    ]),
    hdrs = [
        "include/mlir/Dialect/StandardOps/Ops.cpp.inc",
        "include/mlir/Dialect/StandardOps/Ops.h.inc",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":gen-call-interfaces-decls",
        ":gen-standard-op-decls",
        ":gen-standard-op-defs",
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
        ":IR",
        ":Parser",
        ":StandardOps",
        ":Support",
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
        "lib/Support/ToolUtilities.cpp",
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
        ":Analysis",
        "@llvm//:support",
    ],
)

cc_library(
    name = "LLVMIR",
    srcs = glob([
        "lib/Dialect/LLVMIR/IR/LLVMDialect.cpp",
    ]),
    hdrs = [
        "include/mlir/Dialect/LLVMIR/LLVMOps.cpp.inc",
        "include/mlir/Dialect/LLVMIR/LLVMOps.h.inc",
        "include/mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc",
        "include/mlir/Dialect/LLVMIR/LLVMOpsEnums.h.inc",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        # ":gen-llvm-conversions",
        ":gen-llvm-enum-decls",
        ":gen-llvm-enum-defs",
        ":gen-llvm-op-decls",
        ":gen-llvm-op-defs",
        ":Analysis",
        "@llvm//:asm_parser",
        "@llvm//:core",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "NVVMIR",
    srcs = glob([
        "lib/Dialect/LLVMIR/IR/NVVMDialect.cpp",
    ]),
    hdrs = [
        "include/mlir/Dialect/LLVMIR/NVVMOps.cpp.inc",
        "include/mlir/Dialect/LLVMIR/NVVMOps.h.inc",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":gen-nvvm-op-defs",
        ":gen-nvvm-op-decls",
        # ":gen-nvvm-conversions",
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
        "lib/Dialect/GPU/**/*.cpp",
    ]),
    hdrs = [
        "include/mlir/Dialect/GPU/GPUOps.cpp.inc",
        "include/mlir/Dialect/GPU/GPUOps.h.inc",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":IR",
        ":StandardOps",
        ":gen-gpu-op-decls",
        ":gen-gpu-op-defs",
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
        ":gen-call-interfaces-decls",
        ":gen-call-interfaces-defs",
        "@llvm//:support",
    ],
)

cc_library(
    name = "AffineOps",
    srcs = glob([
        "lib/Dialect/AffineOps/*.cpp",
        "lib/Dialect/AffineOps/*.h",
    ]),
    hdrs = [
        "include/mlir/Dialect/AffineOps/AffineOps.cpp.inc",
        "include/mlir/Dialect/AffineOps/AffineOps.h.inc",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":IR",
        ":StandardOps",
        ":gen-affine-op-decls",
        ":gen-affine-op-defs",
        ":gen-loop-like-interface-decls",
    ],
    alwayslink = 1,
)

cc_library(
    name = "Analysis",
    srcs = glob([
        "lib/Analysis/*.cpp",
        "lib/Analysis/*.h",
    ]),
    hdrs = [
        "include/mlir/Analysis/CallInterfaces.h.inc",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":AffineOps",
        ":LoopOps",
        ":VectorOps",
        ":gen-call-interfaces-decls",
        ":gen-call-interfaces-defs",
        ":gen-infer-type-op-interface-decls",
        ":gen-infer-type-op-interface-defs",
        "@llvm//:support",
    ],
    alwayslink = 1,
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
    ]),
    hdrs = [
        "include/mlir/Dialect/LoopOps/LoopOps.cpp.inc",
        "include/mlir/Dialect/LoopOps/LoopOps.h.inc",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":StandardOps",
        ":gen-loop-like-interface-decls",
        ":gen-loop-op-decls",
        ":gen-loop-op-defs",
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
        "lib/Dialect/VectorOps/*.cpp",
        "lib/Dialect/VectorOps/*.h",
    ]),
    hdrs = [
        "include/mlir/Dialect/VectorOps/VectorOps.cpp.inc",
        "include/mlir/Dialect/VectorOps/VectorOps.h.inc",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":IR",
        ":gen-vector-op-decls",
        ":gen-vector-op-defs",
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
        ":gen-loop-like-interface-decls",
        ":gen-loop-like-interface-defs",
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

mlir_tblgen(
    name = "gen-test-ops-decls",
    src = "test/lib/TestDialect/TestOps.td",
    out = "test/lib/TestDialect/TestOps.h.inc",
    action = "-gen-op-decls",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-test-ops-defs",
    src = "test/lib/TestDialect/TestOps.td",
    out = "test/lib/TestDialect/TestOps.cpp.inc",
    action = "-gen-op-defs",
    incs = ["include"],
)

mlir_tblgen(
    name = "gen-test-ops-rewriters",
    src = "test/lib/TestDialect/TestOps.td",
    out = "test/lib/TestDialect/TestPatterns.inc",
    action = "-gen-rewriters",
    incs = ["include"],
)

cc_library(
    name = "TestDialect",
    srcs = glob([
        "test/lib/TestDialect/*.cpp",
        "test/lib/TestDialect/*.h",
    ]),
    copts = PLATFORM_COPTS,
    includes = ["test/lib/TestDialect"],
    deps = [
        ":Analysis",
        ":Dialect",
        ":IR",
        ":Pass",
        ":TransformUtils",
        ":Transforms",
        ":gen-test-ops-decls",
        ":gen-test-ops-defs",
        ":gen-test-ops-rewriters",
    ],
    alwayslink = 1,
)

cc_library(
    name = "TestTransforms",
    srcs = glob([
        "test/lib/Transforms/**/*.cpp",
    ]),
    copts = PLATFORM_COPTS,
    deps = [
        ":TestDialect",
        ":TransformUtils",
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "TranslateClParser",
    srcs = ["lib/Support/TranslateClParser.cpp"],
    hdrs = ["include/mlir/Support/TranslateClParser.h"],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    deps = [
        ":Analysis",
        ":IR",
        ":Parser",
        ":Support",
        ":Translation",
        "@llvm//:support",
    ],
)

cc_library(
    name = "MlirTranslateMain",
    srcs = ["tools/mlir-translate/mlir-translate.cpp"],
    copts = PLATFORM_COPTS,
    deps = [
        ":IR",
        ":Parser",
        ":Support",
        ":TranslateClParser",
        ":Translation",
        "@llvm//:support",
    ],
)

cc_binary(
    name = "mlir-opt",
    srcs = glob([
        "tools/mlir-opt/*.cpp",
        "tools/mlir-opt/*.h",
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
        ":Analysis",
        ":OptMain",
        ":Parser",
        ":TestTransforms",
        ":Transforms",
    ],
)
