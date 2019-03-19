# Bazel Workspace for PlaidML
workspace(name = "com_intel_plaidml")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

new_local_repository(
    name = "llvm_shim",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "license",
    srcs = ["@llvm//:license"],
)

cc_library(
    name = "llvm",
    deps = ["@llvm"],
)
    """,
    path = "vendor/llvm/shim",
    workspace_file_content = """
workspace(name = "llvm_shim")
    """,
)

git_repository(
    name = "toolchain",
    commit = "a487bf9f2cc4edc47d376606abaaf29d85fffcd8",
    remote = "https://github.com/plaidml/toolchain",
)

load(
    "@toolchain//:workspace.bzl",
    toolchain_repositories = "repositories",
)

toolchain_repositories()

local_repository(
    name = "opengl_repo",
    path = "vendor/opengl",
)

load("//bzl:workspace.bzl", "plaidml_workspace")

plaidml_workspace()
