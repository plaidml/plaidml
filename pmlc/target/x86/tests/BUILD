# Copyright 2019 Intel Corporation.

load("//pmlc:lit.bzl", "glob_lit_tests")

glob_lit_tests(
    tags_override = {
        # FIXME: We get the following error on windows:
        # LIBXSMM ERROR: application must be linked against LAPACK/BLAS sgemm!
        "xsmm_call.mlir": ["skip_windows"],
        "brgemm_call.mlir": ["skip_windows"],
        "brgemm_call2.mlir": ["skip_windows"],
        "xsmm_gemm_call.mlir": ["skip_windows"],
    },
)
