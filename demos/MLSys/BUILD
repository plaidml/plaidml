package(default_visibility = ["//visibility:public"])

py_binary(
  name = "demo",
  srcs = ["demo.py"],
  deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
    ],
  data = ["demo.ipynb"],
)
