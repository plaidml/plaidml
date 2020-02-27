package(default_visibility = ["//visibility:public"])

py_binary(
    name = "demo",
    srcs = [
        "demo.py",
        "demo_handler.py",
        "demo_source.py",
        "lowering_demo_app.py",
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

# sh_binary(
#     name = "voila",
#     srcs = ["@com_intel_plaidml_conda_unix//:voila"],
#     args = ["$(location :lowering_demo.ipynb)"],
#     data = [
#         "lowering_demo.ipynb",
#         ":demo",
#     ],
# )

py_binary(
    name = "voila",
    srcs = glob([
        "*.py",
    ]),
    args = [
        "--no-browser",
        "$(location :lowering_demo.ipynb)",
    ],
    data = glob([
        "*.ipynb",
    ]),
    deps = [
        "//plaidml:py",
        "//plaidml/edsl:py",
        "//plaidml/exec:py",
        "//plaidml/op:py",
    ],
)
