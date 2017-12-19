load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_cc_test", "plaidml_ast")

py_binary(
    name = "astgen",
    srcs = ["astgen.py"],
    data = ["base.template"],
    visibility = ["//visibility:public"],
)

plaidml_ast(
    name = "make_test_ast",
    ast = "test_ast.yaml",
    output = "test_ast.h",
)

plaidml_cc_test(
    name = "ast_test",
    srcs = [
        "test_ast.cc",
        ":make_test_ast",
    ],
    deps = [
        "//base/util:util_cc",
    ],
)
