# Copyright 2020 Intel Corporation

package(default_visibility = ["//visibility:public"])

# Test that query for availiable devices executes properly.
# This is only untill real tests for OpenCL runtime are availiable.
cc_test(
    name = "clprobe_test",
    srcs = ["clprobe.cc"],
    deps = ["@opencl_icd_loader", "@opencl_hpp_headers"],
)
