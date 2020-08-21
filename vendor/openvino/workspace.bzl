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
        url = "https://github.com/PlaidML/openvino/archive/60bd0feb01b3abea217a1fb62bfbd3b80d2cc4e7.zip",
        strip_prefix = "openvino-60bd0feb01b3abea217a1fb62bfbd3b80d2cc4e7",
        sha256 = "de4985988d9c5378a317e8323b03ee8e40b1bbc63e20fd12e80680823cb70137",
        build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
