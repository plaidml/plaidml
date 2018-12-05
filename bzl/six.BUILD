package(default_visibility = ["@//visibility:public"])

py_library(
    name = "six",
    srcs = ["six.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

genrule(
    name = "license",
    srcs = ["LICENSE"],
    outs = ["six-LICENSE"],
    cmd = "cp $(SRCS) $@",
)
