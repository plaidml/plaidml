load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        #":executable",
        ":platform_id",
        "@org_tensorflow//tensorflow/compiler/xla:xla_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        #"@org_tensorflow//tensorflow/stream_executor",
        "@com_google_absl//absl/memory",
        "//plaidml",
    ],
)

# cc_library(
#     name = "executable_base",
#     srcs = ["executable_base.cc"],
#     hdrs = ["executable_base.h"],
#     deps = [
#         "//plaidml/edsl",
#         "//plaidml/exec",
#         "//plaidml/op",
#         "@com_google_absl//absl/types:optional",
#         "@org_tensorflow//tensorflow/compiler/xla:literal",
#         "@org_tensorflow//tensorflow/compiler/xla:shape_tree",
#         "@org_tensorflow//tensorflow/compiler/xla:shape_util",
#         "@org_tensorflow//tensorflow/compiler/xla:statusor",
#         "@org_tensorflow//tensorflow/compiler/xla:xla_proto_cc",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:dynamic_dimension_inference",
#             # "@org_tensorflow//tensorflow/compiler/xla/service:executable",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:hlo_execution_profile",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:maybe_owning_device_memory",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:shaped_buffer",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:transfer_manager",
#         # "@org_tensorflow//tensorflow/stream_executor:event",
#         # "@org_tensorflow//tensorflow/stream_executor:stream",
#         # "@org_tensorflow//tensorflow/stream_executor/lib",
#     ],
# )

# cc_library(
#     name = "executable",
#     srcs = ["executable.cc"],
#     hdrs = ["executable.h"],
#     deps = [
#         ":executable_base",
#         ":executor",
#         "//plaidml/edsl",
#         "//plaidml/exec",
#         "//plaidml/op",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/types:span",
#         "@org_tensorflow//tensorflow/compiler/xla:literal",
#         "@org_tensorflow//tensorflow/compiler/xla:shape_util",
#         "@org_tensorflow//tensorflow/compiler/xla:status_macros",
#         "@org_tensorflow//tensorflow/compiler/xla:statusor",
#         # "@org_tensorflow//tensorflow/compiler/xla:types",
#         "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_cc",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:executable",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:hlo_cost_analysis",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:hlo_evaluator",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:hlo_execution_profile",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:hlo_module_config",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:maybe_owning_device_memory",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:shaped_buffer",
#         # "@org_tensorflow//tensorflow/compiler/xla/service:transfer_manager",
#         # "@org_tensorflow//tensorflow/core:lib",
#         # "@org_tensorflow//tensorflow/core:stream_executor_no_cuda",
#         # "@org_tensorflow//tensorflow/core/platform:macros",
#         # "@org_tensorflow//tensorflow/core/platform:mutex",
#         # "@org_tensorflow//tensorflow/core/platform:types",
#     ],
# )

# cc_library(
#     name = "executor",
#     srcs = ["executor.cc"],
#     hdrs = ["executor.h"],
#     deps = [
#         "@com_google_absl//absl/types:span",
#         # "@org_tensorflow//:host_stream",
#         # "@org_tensorflow//:host_timer",
#         "@org_tensorflow//tensorflow/compiler/xla:shape_util",
#         "@org_tensorflow//tensorflow/compiler/xla:status_macros",
#         "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_cc",
#         "@org_tensorflow//tensorflow/core:lib",
#         "@org_tensorflow//tensorflow/core:stream_executor_headers_lib",
#     ],
# )

cc_library(
    name = "platform_id",
    srcs = ["platform_id.cc"],
    hdrs = ["platform_id.h"],
    deps = [
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_headers",
        "@org_tensorflow//tensorflow/core:stream_executor_headers_lib",
    ],
)

# cc_library(
#     name = "platform",
#     srcs = ["platform.cc"],
#     hdrs = ["platform.h"],
#     deps = [
#         ":executor",
#         ":platform_id",
#         "@org_tensorflow//tensorflow/core:stream_executor_headers_lib",
#     ],
#     alwayslink = True,  # Registers itself with the MultiPlatformManager.
# )
