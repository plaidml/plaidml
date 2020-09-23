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
        url = "https://github.com/plaidml/openvino/archive/0cad515350629375125db878a373d15f6feb7853.zip",
        strip_prefix = "openvino-0cad515350629375125db878a373d15f6feb7853",
        sha256 = "7773484029459b57c2300435a65a7084c7f6b4a2a6e9398495692b24d682e166",
        build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
