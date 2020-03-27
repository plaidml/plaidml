package(default_visibility = ["//visibility:public"])

exports_files(["env"])

filegroup(
    name = "all",
    srcs = glob(["env/**/*"]),
)

filegroup(
    name = "python",
    srcs = ["env/python.exe"],
)

filegroup(
    name = "conda",
    srcs = ["env/Scripts/conda.exe"],
)
