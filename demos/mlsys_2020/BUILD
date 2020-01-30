package(default_visibility = ["//visibility:public"])

py_binary(
    name = "demo",
    srcs = ["demo.py"],
    data = ["demo.ipynb"],
    deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
    ],
)
