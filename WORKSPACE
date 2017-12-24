# Bazel Workspace for Vertex.AI
workspace(name = "vertexai_plaidml")

local_repository(
    name = "opengl_repo",
    path = "vendor/opengl",
)

load("//bzl:workspace.bzl", "plaidml_workspace")
plaidml_workspace()
