load("@bazel_latex//:latex.bzl", "latex_document")

latex_document(
    name = "docs",
    srcs =[
        "@bazel_latex//packages:graphicx",
        "@bazel_latex//packages:xcolor",
        "@bazel_latex//packages:amsmath",
        "@bazel_latex//packages:hyperref",
        "@bazel_latex//packages:listings",
        "toroidal_shell_moment.jpg",
        "uw_toroidal_shell.tex",
    ],
    main = "uw_toroidal_shell.tex",
)
