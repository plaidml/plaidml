# Copyright 2020 Intel Corporation

load("//vendor/mlir:tblgen.bzl", "COPTS", "gentbl")
load("//bzl:plaidml.bzl", "plaidml_cc_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "td_files",
    srcs = ["enums.td"],
)

gentbl(
    name = "enums-gen",
    tbl_outs = [
        ("-gen-enum-decls", "enums.h.inc"),
        ("-gen-enum-defs", "enums.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "enums.td",
    td_srcs = ["@llvm-project//mlir:OpBaseTdFiles"],
)

plaidml_cc_library(
    name = "util",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = COPTS,
    deps = [
        ":enums-gen",
        "@easylogging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)

plaidml_cc_library(
    name = "env",
    srcs = ["env.cc"],
    hdrs = ["env.h"],
)

plaidml_cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = ["@easylogging"],
)
