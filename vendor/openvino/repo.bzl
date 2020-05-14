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
        sha256 = "515426cf7be052871f8b4a29bb73e8a960adbe9f1fbf9d37c4fd0cc992cdbb6f",
        url = "https://github.com/NervanaSystems/ngraph/archive/b0bb801c91f091a125a6aeb38c51f10fc49f3425.zip",
        strip_prefix = "ngraph-b0bb801c91f091a125a6aeb38c51f10fc49f3425",
        build_file = clean_dep("//vendor/openvino:ngraph.BUILD"),
    )

    http_archive(
        name = "nlo_json",
        url = "https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip",
        sha256 = "87b5884741427220d3a33df1363ae0e8b898099fbc59f1c451113f6732891014",
        build_file = clean_dep("//vendor/openvino:nlohmann_json.BUILD"),
    )

    http_archive(
        name = "openvino",
        sha256 = "0d641b158fff868ae7a969c02bf73462e335c15c40b42e3fe65ac6ea7cffea66",
        strip_prefix = "openvino-2020.1",
        url = "https://github.com/opencv/dldt/archive/2020.1.zip",
        build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
