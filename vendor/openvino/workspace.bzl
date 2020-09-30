# Copyright 2020 Intel Corporation

load("//vendor/bazel:repo.bzl", "http_archive")

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def openvino_workspace():
    http_archive(
        name = "ade",
        url = "https://github.com/opencv/ade/archive/cbe2db61a659c2cc304c3837406f95c39dfa938e.zip",
        strip_prefix = "ade-cbe2db61a659c2cc304c3837406f95c39dfa938e",
        sha256 = "6660e1b66bd3d8005026155571a057765ace9b0fdd9899aaa5823eca12847896",
        build_file = clean_dep("//vendor/openvino:ade.BUILD"),
    )

    http_archive(
        name = "openvino",
        url = "https://github.com/plaidml/openvino/archive/62963cfd8cbd0466c6bba56407250747193993e3.zip",
        strip_prefix = "openvino-62963cfd8cbd0466c6bba56407250747193993e3",
        sha256 = "507982d08519f53b17bd0c819d3cda1e837567e20a5d9aa0cc24304e0892de12",
        build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
