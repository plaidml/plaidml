# Copyright 2020 Intel Corporation

load("//bzl:plaidml.bzl", "plaidml_cc_binary", "plaidml_cc_library", "plaidml_cc_test")

filegroup(
    name = "lit_files",
    srcs = glob(["lit*cfg.py"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "lit_tests",
    srcs = [
        "//pmlc/conversion/pxa_to_affine/tests:all",
        "//pmlc/conversion/stdx_to_llvm/tests:all",
        "//pmlc/conversion/tile_to_pxa/tests:all",
        "//pmlc/conversion/gpu_to_spirv/tests:all",
        "//pmlc/dialect/comp/tests:all",
        "//pmlc/dialect/eltwise/tests:all",
        "//pmlc/dialect/oplib/tests:all",
        "//pmlc/dialect/pxa/tests:all",
        "//pmlc/dialect/stdx/tests:all",
        "//pmlc/dialect/tile/tests:all",
        "//pmlc/dialect/xsmm/tests:all",
        "//pmlc/rt/tests:all",
        "//pmlc/rt/vulkan/tests:all",
        "//pmlc/target/intel_gen/tests:all",
        "//pmlc/target/x86/tests:all",
    ],
    visibility = ["//visibility:public"],
)

plaidml_cc_test(
    name = "test",
    srcs = ["test.cc"],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
    ],
)

plaidml_cc_library(
    name = "hdrs",
    hdrs = [
        "all_dialects.h",
        "all_passes.h",
    ],
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "all_passes_and_dialects",
    srcs = ["all_dialects.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":hdrs",
        "//pmlc/conversion/gpu",
        "//pmlc/conversion/pxa_to_affine",
        "//pmlc/conversion/stdx_to_llvm",
        "//pmlc/conversion/tile_to_pxa",
        "//pmlc/conversion/gpu_to_spirv",
        "//pmlc/dialect/comp",
        "//pmlc/dialect/eltwise",
        "//pmlc/dialect/pxa",
        "//pmlc/dialect/stdx",
        "//pmlc/dialect/tile",
        "//pmlc/dialect/xsmm",
        "//pmlc/target/demo",
        "//pmlc/target/intel_gen",
        "//pmlc/target/x86",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
    ],
)

# The following rules are helpers to make it quicker and easier to launch the
# pmlc tools manually.  Because they include all lit test files in their
# runfiles, relative paths can be used from the command line.
# Example:
#   bazelisk run //pmlc:opt -- pmlc/dialect/tile/tests/constant-fold.mlir

plaidml_cc_binary(
    name = "jit",
    data = [":lit_tests"],
    deps = ["//pmlc/tools/pmlc-jit:lib"],
)

plaidml_cc_binary(
    name = "opt",
    data = [":lit_tests"],
    deps = ["//pmlc/tools/pmlc-opt:lib"],
)

plaidml_cc_binary(
    name = "tblgen",
    data = [":lit_tests"],
    deps = ["//pmlc/tools/pmlc-tblgen:lib"],
)

plaidml_cc_binary(
    name = "translate",
    data = [":lit_tests"],
    deps = ["//pmlc/tools/pmlc-translate:lib"],
)

plaidml_cc_binary(
    name = "vulkan",
    data = [":lit_tests"],
    deps = ["//pmlc/tools/pmlc-vulkan-runner:lib"],
)
