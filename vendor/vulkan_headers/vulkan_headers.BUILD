# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(default_visibility = ["//visibility:public"])

# Exports all headers but defining VK_NO_PROTOTYPES to disable the
# inclusion of C function prototypes. Useful if dynamically loading
# all symbols via dlopen/etc.
# Not all headers are hermetic, so they are just included as textual
# headers to disable additional validation.
cc_library(
    name = "vulkan_headers_no_prototypes",
    defines = ["VK_NO_PROTOTYPES"],
    includes = ["include"],
    textual_hdrs = glob(["include/vulkan/*.h"]),
)

# Exports all headers, including C function prototypes. Useful if statically
# linking against the Vulkan SDK.
# Not all headers are hermetic, so they are just included as textual
# headers to disable additional validation.
cc_library(
    name = "vulkan_headers",
    includes = ["include"],
    textual_hdrs = glob(["include/vulkan/*.h"]),
)
