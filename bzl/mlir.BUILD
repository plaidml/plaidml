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
    visibility = ["//visibility:public"],
    deps = ["@llvm//:TableGen"],
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
    name = "ir",
    srcs = glob([
        "lib/AffineOps/**/*.h",
        "lib/AffineOps/**/*.cpp",
        "lib/Analysis/**/*.h",
        "lib/Analysis/**/*.cpp",
        "lib/Conversion/StandardToLLVM/**/*.h",
        "lib/Conversion/StandardToLLVM/**/*.cpp",
        "lib/Conversion/ControlFlowToCFG/**/*.h",
        "lib/Conversion/ControlFlowToCFG/**/*.cpp",
        "lib/Dialect/LoopOps/**/*.h",
        "lib/Dialect/LoopOps/**/*.cpp",
        "lib/Dialect/LoopOps/**/*.h",
        "lib/Dialect/LoopOps/**/*.cpp",
        "lib/EDSC/**/*.h",
        "lib/EDSC/**/*.cpp",
        "lib/ExecutionEngine/**/*.h",
        "lib/ExecutionEngine/**/*.cpp",
        "lib/GPU/**/*.h",
        "lib/GPU/**/*.cpp",
        "lib/IR/**/*.h",
        "lib/IR/**/*.cpp",
        "lib/LLVMIR/**/*.h",
        "lib/LLVMIR/**/*.cpp",
        "lib/Pass/**/*.h",
        "lib/Pass/**/*.cpp",
        "lib/Parser/**/*.h",
        "lib/Parser/**/*.cpp",
        "lib/StandardOps/**/*.h",
        "lib/StandardOps/**/*.cpp",
        "lib/Support/**/*.h",
        "lib/Support/**/*.cpp",
        "lib/Target/**/*.h",
        "lib/Target/**/*.cpp",
        "lib/Transforms/**/*.h",
        "lib/Transforms/**/*.cpp",
        "lib/Translation/**/*.h",
        "lib/Translation/**/*.cpp",
        "lib/VectorOps/**/*.cpp",
    ]) + [
        ":gen-standard-op-defs",
        ":gen-affine-op-defs",
        ":gen-loop-op-defs",
        ":gen-llvm-op-defs",
        ":gen-llvm-enum-defs",
        ":gen-llvm-conversions",
        ":gen-nvvm-op-defs",
        ":gen-nvvm-conversions",
        ":gen-gpu-op-defs",
    ],
    hdrs = glob([
        "lib/Parser/**/*.def",
    ]) + [
        ":gen-standard-op-decls",
        ":gen-affine-op-decls",
        ":gen-loop-op-decls",
        ":gen-llvm-op-decls",
        ":gen-llvm-enum-decls",
        ":gen-nvvm-op-decls",
        ":gen-gpu-op-decls",
    ],
    copts = PLATFORM_COPTS,
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm//:Core",
        "@llvm//:ExecutionEngine",
        "@llvm//:OrcJIT",
        "@llvm//:Support",
        "@llvm//:TransformUtils",
        "@llvm//:X86",
        "@llvm//:ipo",
    ],
)

cc_library(
    name = "test_transforms",
    srcs = glob([
        "test/lib/Transforms/**/*.cpp",
    ]),
    copts = PLATFORM_COPTS,
    visibility = ["//visibility:public"],
    deps = [":mlir"],
    alwayslink = 1,
)
