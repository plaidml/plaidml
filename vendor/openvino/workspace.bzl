# Copyright 2020 Intel Corporation

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def openvino_workspace():
    new_git_repository(
        name = "openvino",
        commit = "c32a7f47e9df54bbeef31ff90e1b9248bc30070e",
        init_submodules = True,
        remote = "https://github.com/plaidml/openvino",
        build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
