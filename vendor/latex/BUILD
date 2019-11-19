# Copyright 2019 Intel Corporation

load("@bazel_latex//packages:packages.bzl", "latex_package")

# Externally usable library targets.

latex_package(
    name = "minted",
    srcs = [
        "ifplatform",
        ":fancyvrb",
        ":fvextra",
        "@bazel_latex//packages:etoolbox",
        "@bazel_latex//packages:float",
        "@bazel_latex//packages:framed",
        "@bazel_latex//packages:lineno",
        "@bazel_latex//packages:upquote",
        "@bazel_latex//packages:xstring",
        "@texlive_texmf__texmf-dist__tex__latex__minted",
    ],
    tests = ["minted_test.tex"],
)

latex_package(
    name = "fvextra",
    srcs = [
        "@texlive_texmf__texmf-dist__tex__latex__fvextra",
    ],
    tests = ["fvextra_test.tex"],
)

latex_package(
    name = "fancyvrb",
    srcs = [
        "@texlive_texmf__texmf-dist__tex__latex__fancyvrb",
    ],
    tests = ["fancyvrb_test.tex"],
)

latex_package(
    name = "ifplatform",
    srcs = [
        "@texlive_texmf__texmf-dist__tex__latex__ifplatform",
    ],
    tests = ["ifplatform_test.tex"],
)
