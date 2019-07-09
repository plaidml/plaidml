# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_jsonnet//jsonnet:jsonnet.bzl", "jsonnet_to_json")

jsonnet_to_json(
    name = "amd",
    src = "amd.jsonnet",
    outs = ["amd.json"],
)

jsonnet_to_json(
    name = "nvidia",
    src = "nvidia.jsonnet",
    outs = ["nvidia.json"],
)

jsonnet_to_json(
    name = "intel_gen9",
    src = "intel_gen9.jsonnet",
    outs = ["intel_gen9.json"],
)

