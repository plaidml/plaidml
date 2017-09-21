package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_py_binary", "plaidml_py_library", "plaidml_py_test")

plaidml_py_binary(
    name = "analysis",
    srcs = ["__init__.py"],
    main = "__init__.py",
    deps = [
        ":util",
        "//external:protobuf_python",
        "//tile/hal/opencl:proto_py",
        "//tile/platform/local_machine:proto_py",
        "//tile/proto:hal_py",
    ],
)

plaidml_py_test(
    name = "analysis_test",
    srcs = ["analysis_test.py"],
    data = [
        "Analysis.ipynb",
        "testdata/mnist_mlp_log.gz",
    ],
    tags = ["manual"],
    deps = [
        ":analysis",
    ],
)

plaidml_py_binary(
    name = "kernel_times",
    srcs = ["kernel_times.py"],
    deps = [":analysis"],
)

plaidml_py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//base/context:proto_py",
        "//base/eventing/file:proto_py",
    ],
)

plaidml_py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    data = ["testdata/eventlog.gz"],
    deps = [
        ":util",
        "//external:protobuf_python",
    ],
)

plaidml_py_binary(
    name = "to_csv",
    srcs = ["to_csv.py"],
    deps = [":util"],
)

plaidml_py_binary(
    name = "to_excel",
    srcs = ["to_excel.py"],
    deps = [":util"],
)

plaidml_py_binary(
    name = "to_graph",
    srcs = ["to_graph.py"],
    deps = [":util"],
)

plaidml_py_binary(
    name = "stats",
    srcs = ["stats.py"],
    deps = [":util"],
)
