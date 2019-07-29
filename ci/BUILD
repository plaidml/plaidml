package(default_visibility = ["//visibility:public"])

load("@com_intel_plaidml//bzl:conda.bzl", "conda_binary")

# conda_binary(
#     name = "analysis",
#     srcs = ["analysis.py"],
#     data = glob([
#         "golden/**/*",
#         "plan.yml",
#         "templates/*",
#     ]),
#     env = "@com_intel_plaidml_conda_analysis//:env",
#     tags = ["manual"],
#     deps = ["//tools/analysis"],
# )

conda_binary(
    name = "bkpoke",
    srcs = ["bkpoke.py"],
    env = "@com_intel_plaidml_conda_buildkite//:env",
    tags = ["manual"],
)
