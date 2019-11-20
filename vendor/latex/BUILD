# Copyright 2019 Intel Corporation

load("@bazel_latex//packages:packages.bzl", "latex_package")

# Externally usable library targets.

latex_package(
    name = "minted",
    srcs = [
        ":fancyvrb",
        ":fvextra",
        ":ifplatform",
        "@bazel_latex//packages:xcolor",
        "@bazel_latex//packages:etoolbox",
        "@bazel_latex//packages:float",
        "@bazel_latex//packages:framed",
        "@bazel_latex//packages:lineno",
        "@bazel_latex//packages:upquote",
        "@bazel_latex//packages:xstring",
        "@texlive_texmf__texmf-dist__tex__latex__minted",
    ],
    #fix: need to pass on ci build
    #tests = ["minted_test.tex"],
)

latex_package(
    name = "fancyvrb",
    srcs = [
        "@bazel_latex//packages:keyval",
        "@texlive_texmf__texmf-dist__tex__latex__fancyvrb",
    ],
    #fix: need to pass on ci build 
    #tests = ["fancyvrb_test.tex"],
)

latex_package(
    name = "fvextra",
    srcs = [
        ":fancyvrb",
        "@bazel_latex//packages:etoolbox",
        "@bazel_latex//packages:lineno",
        "@bazel_latex//packages:upquote",
        "@texlive_texmf__texmf-dist__tex__latex__fvextra",
    ],
    #fix: need to pass on ci build 
    #tests = ["fvextra_test.tex"],
)

latex_package(
    name = "ifplatform",
    srcs = [
        # Need to make tools visible
        #"@bazel_latex//packages:tools", 
        "@bazel_latex//packages:ifpdf",
        "@texlive_texmf__texmf-dist__tex__latex__ifplatform",
    ],
    # need to make tools package visible in bazel-latex 
    #tests = ["ifplatform_test.tex"],
)

