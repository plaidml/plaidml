# Bazel Workspace for Vertex.AI
workspace(name = "vertexai_plaidml")

local_repository(
    name = "opengl_repo",
    path = "vendor/opengl",
)

load("//bzl:deps.bzl", "plaidml_deps")
plaidml_deps()

load("//bzl:protobuf.bzl", "with_protobuf")
with_protobuf()

load("//bzl:plaidml.bzl", "with_plaidml")
with_plaidml()

git_repository(
    name = "io_bazel_rules_python",
    remote = "https://github.com/earhart/rules_python.git",
    commit = "0a4cd713349294a4ef28f1f61bf62851c83ccbe5",
)

load("@io_bazel_rules_python//python:pip.bzl", "pip_repositories", "pip_import")
pip_repositories()

pip_import(
    name = "vertexai_plaidml_pip_deps",
    requirements = "//:requirements.txt",
)

load("@vertexai_plaidml_pip_deps//:requirements.bzl", "pip_install")
pip_install()
