# Copyright 2019 Intel Corporation

load("@bazel_latex//packages:packages.bzl", "latex_package")

# Externally usable library targets.

latex_package(
    name = "minted",
    srcs = [
        # ":collectbox",
        # ":graphicx",
        # ":pgf",
        # ":xkeyval",
        "@texlive_texmf__texmf-dist__tex__latex__minted",
    ],
    tests = ["minted_test.tex"],
)
