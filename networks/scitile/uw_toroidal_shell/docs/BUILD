load("@bazel_latex//:latex.bzl", "latex_document")

latex_document(
    name = "docs",
    srcs = [
        "toroidal_shell_moment.jpg",
        "uw_toroidal_shell_doc.tex",
        "//vendor/latex:minted",
        "@bazel_latex//packages:amsmath",
        "@bazel_latex//packages:graphicx",
        "@bazel_latex//packages:hyperref",
        "@bazel_latex//packages:listings",
        "@bazel_latex//packages:xcolor",
    ],
    main = "uw_toroidal_shell_doc.tex",
    tags = ["latex"],
)
