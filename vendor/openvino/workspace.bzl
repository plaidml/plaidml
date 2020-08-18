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
        name = "ngraph",
        url = "https://github.com/NervanaSystems/ngraph/archive/edc65ca0111f86a7e63a98f62cb17d153cc2535c.zip",
        strip_prefix = "ngraph-edc65ca0111f86a7e63a98f62cb17d153cc2535c",
        sha256 = "34c0c0ec372514b79105db8896381826595565ff0c4cd76a2d2a5f1e7eb82c19",
        build_file = clean_dep("//vendor/openvino:ngraph.BUILD"),
    )

    http_archive(
        name = "openvino",
        url = "https://github.com/PlaidML/openvino/archive/fa832406b3c0d053aab4042ff53ac7f6dccde16b.zip",
        strip_prefix = "openvino-fa832406b3c0d053aab4042ff53ac7f6dccde16b",
        sha256 = "1c308163a4ef381c49ea203f6dda1820a233b64e2b22a8aa1750fc46d61256d0",
        build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
