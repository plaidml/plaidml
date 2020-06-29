# Copyright 2020 Intel Corporation

load("//vendor/bazel:repo.bzl", "http_archive")

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def openvino_workspace():
    http_archive(
        name = "ade",
        sha256 = "6660e1b66bd3d8005026155571a057765ace9b0fdd9899aaa5823eca12847896",
        url = "https://github.com/opencv/ade/archive/cbe2db61a659c2cc304c3837406f95c39dfa938e.zip",
        strip_prefix = "ade-cbe2db61a659c2cc304c3837406f95c39dfa938e",
        build_file = clean_dep("//vendor/openvino:ade.BUILD"),
    )

    http_archive(
        name = "ngraph",
        build_file = "//vendor/openvino:ngraph.BUILD",
        sha256 = "34c0c0ec372514b79105db8896381826595565ff0c4cd76a2d2a5f1e7eb82c19",
        strip_prefix = "ngraph-edc65ca0111f86a7e63a98f62cb17d153cc2535c",
        url = "https://github.com/NervanaSystems/ngraph/archive/edc65ca0111f86a7e63a98f62cb17d153cc2535c.zip",
    )

    http_archive(
        name = "openvino",
        sha256 = "493a989ebdcd0b90d606261555ef9762a51086a9608973a14dc1dc005576b657",
        strip_prefix = "openvino-e178fec7701ecf563378a951a5af6e303dc1f8e3",
        url = "https://github.com/PlaidML/openvino/archive/e178fec7701ecf563378a951a5af6e303dc1f8e3.zip",
        build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
