message("Fetching opencl_headers")
FetchContent_Declare(
  opencl_headers
  URL      https://github.com/KhronosGroup/OpenCL-Headers/archive/v2020.06.16.zip
  URL_HASH SHA256=518703d3c3a6333bcf8e4f80758e4e98f7af30fbd72a09fe8c2673da1628d80c
)
FetchContent_MakeAvailable(opencl_headers)

add_library(opencl_headers INTERFACE)
target_include_directories(opencl_headers INTERFACE ${opencl_headers_SOURCE_DIR})
