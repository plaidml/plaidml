message("Fetching volk")
FetchContent_Declare(
  volk
  URL      https://github.com/zeux/volk/archive/2638ad1b2b40f1ad402a0a6ac55b60bc51a23058.zip
  URL_HASH SHA256=4a5fb828e05d8c86f696f8754e90302d6446b950236256bcb4857408357d2b60
)
FetchContent_MakeAvailable(volk)

target_include_directories(volk PRIVATE ${vulkan_headers_SOURCE_DIR}/include)
