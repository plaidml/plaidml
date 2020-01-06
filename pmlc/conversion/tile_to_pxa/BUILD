# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

# filegroup(
#     name = "td_files",
#     srcs = [
#         "//pmlc/dialect/pxa:td_files",
#         "//pmlc/dialect/tile:td_files",
#         "@llvm-project//mlir:OpBaseTdFiles",
#     ],
# )

# gentbl(
#     name = "gen-rewrites",
#     tbl_outs = [
#         (
#             "-gen-rewriters",
#             "rewrites.cc.inc",
#         ),
#     ],
#     td_file = "rewrites.td",
#     td_srcs = [":td_files"],
# )

plaidml_cc_library(
    name = "tile_to_pxa",
    srcs = [
        "pass.cc",
        # "rewrites.cc",
    ],
    hdrs = [
        "pass.h",
        # "rewrites.h",
    ],
    copts = COPTS,
    tags = ["llvm"],
    deps = [
        # ":gen-rewrites",
        "//pmlc/dialect/pxa",
        "//pmlc/dialect/tile",
    ],
    alwayslink = 1,
)
