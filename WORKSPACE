# Bazel Workspace for PlaidML
workspace(name = "com_intel_plaidml")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# define this first in case any repository rules want to use it
http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

load("//bzl:workspace.bzl", "plaidml_workspace")

plaidml_workspace()

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

register_toolchains("//:py_toolchain")

load("@bazel_latex//:repositories.bzl", "latex_repositories")

latex_repositories()

http_archive(
    name = "com_grail_bazel_toolchain",
    sha256 = "17e12dfba5ae4fc735dded73500e24546ca74459bac68b8d6f68c7785e3e4500",
    strip_prefix = "bazel-toolchain-7abc442059ea5b08f9c9bc7f0c1450b914aca090",
    urls = ["https://github.com/plaidml/bazel-toolchain/archive/7abc442059ea5b08f9c9bc7f0c1450b914aca090.zip"],
)

load("@com_grail_bazel_toolchain//toolchain:deps.bzl", "bazel_toolchain_dependencies")

bazel_toolchain_dependencies()

load("@com_grail_bazel_toolchain//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "8.0.0",
)

load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")

llvm_register_toolchains()

load("//vendor/openvino:workspace.bzl", "openvino_workspace")

openvino_workspace()
