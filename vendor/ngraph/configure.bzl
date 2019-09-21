load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def configure_ngraph():
    http_archive(
        name = "ngraph",
        url = "https://github.com/NervanaSystems/ngraph/archive/e2f55f83ce9c765f8acb05d7a7fa9f5a82e5ec97.zip",
        sha256 = "b784ffa2d692b03f4dd5696da2abddcb4f72a6153bdfcd7d07757aff748a54eb",
        strip_prefix = "ngraph-e2f55f83ce9c765f8acb05d7a7fa9f5a82e5ec97",
        build_file = "//vendor/ngraph:ngraph.BUILD",
    )

    http_archive(
        name = "nlohmann_json_lib",
        build_file = "//vendor/ngraph:nlohmann_json.BUILD",
        sha256 = "e0b1fc6cc6ca05706cce99118a87aca5248bd9db3113e703023d23f044995c1d",
        strip_prefix = "json-3.5.0",
        urls = [
            "https://mirror.bazel.build/github.com/nlohmann/json/archive/v3.5.0.tar.gz",
            "https://github.com/nlohmann/json/archive/v3.5.0.tar.gz",
        ],
    )
