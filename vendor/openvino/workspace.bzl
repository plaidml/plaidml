# Copyright 2020 Intel Corporation

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Sanitize a dependency so that it works correctly from code that includes it as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def openvino_workspace():
    new_git_repository(
        name = "openvino",
        tag = "pml-current",
        init_submodules = True,
        remote = "https://github.com/plaidml/openvino",
        build_file = clean_dep("//vendor/openvino:openvino.BUILD"),
    )
