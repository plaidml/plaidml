# Bazel Workspace for Vertex.AI
workspace(name = "vertexai_plaidml")

load("//bzl:deps.bzl", "plaidml_deps")
plaidml_deps()

load("//bzl:protobuf.bzl", "with_protobuf")
with_protobuf()

load("//bzl:plaidml.bzl", "with_plaidml")
with_plaidml()