# Bazel Workspace for PlaidML
workspace(name = "com_intel_plaidml")

git_repository(
    name = "toolchain",
    remote = "https://github.com/plaidml/toolchain",
    tag = "0.1.2",
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
