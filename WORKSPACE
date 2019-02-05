# Bazel Workspace for PlaidML
workspace(name = "com_intel_plaidml")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "toolchain",
    remote = "https://github.com/plaidml/toolchain",
    commit = "a487bf9f2cc4edc47d376606abaaf29d85fffcd8",
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
