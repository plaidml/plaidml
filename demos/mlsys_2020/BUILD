package(default_visibility = ["//visibility:public"])

py_binary(
    name = "demo",
    srcs = [
        "demo.py",
        "demo_source.py",
    ],
    data = [
        "demo.ipynb",
        "lowering_demo.ipynb",
    ],
    deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
    ],
)
