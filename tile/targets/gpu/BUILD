# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_jsonnet//jsonnet:jsonnet.bzl", "jsonnet_to_json")

jsonnet_to_json(
    name = "amd",
    src = "amd.jsonnet",
    outs = ["amd.json"],
)

jsonnet_to_json(
    name = "intel_gen9_metal",
    src = "intel_gen9_metal.jsonnet",
    outs = ["intel_gen9_metal.json"],
)

jsonnet_to_json(
    name = "intel_gen9_opencl",
    src = "intel_gen9_opencl.jsonnet",
    outs = ["intel_gen9_opencl.json"],
)
