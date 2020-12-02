FetchContent_Declare(
  vulkan_headers
  URL      https://github.com/KhronosGroup/Vulkan-Headers/archive/v1.2.132.zip
  URL_HASH SHA256=e6b5418e3d696ffc7c97991094ece7cafc4c279c8a88029cc60e587bc0c26068
)
FetchContent_MakeAvailable(vulkan_headers)

add_library(vulkan_headers INTERFACE)
target_include_directories(vulkan_headers INTERFACE ${vulkan_headers_SOURCE_DIR})
target_include_directories(vulkan_headers INTERFACE ${vulkan_headers_SOURCE_DIR}/include)
