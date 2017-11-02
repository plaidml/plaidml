# Bazel Workspace for Vertex.AI
workspace(name = "vertexai_plaidml")

local_repository(
    name = "opengl_repo",
    path = "external/opengl",
)

load("//bzl:deps.bzl", "plaidml_deps")
plaidml_deps()

load("//bzl:protobuf.bzl", "with_protobuf")
with_protobuf()
