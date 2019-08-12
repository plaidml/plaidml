package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cm_headers",
    hdrs = [
        %{cm_headers}
    ],
)

%{cm_include_genrules}
