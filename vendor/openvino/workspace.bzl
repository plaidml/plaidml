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
        url = "https://github.com/plaidml/openvino/archive/c6b9f944be8dc7eac479d05ca4e5aa934dbbd34c.zip",
        strip_prefix = "openvino-c6b9f944be8dc7eac479d05ca4e5aa934dbbd34c",
        sha256 = "d99ef8c46f2e091cd73e77f6b96ea0bf1025305948aa2d8d401ee93534329873",
        build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
