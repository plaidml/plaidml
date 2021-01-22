message("Fetching opencl_icd_loader")
FetchContent_Declare(
  opencl_icd_loader
  URL      https://github.com/KhronosGroup/OpenCL-ICD-Loader/archive/v2020.06.16.zip
  URL_HASH SHA256=e4c27a5adcef4dbc0fee98864af203dc78dfc967ca7287c9bad9add030e7516e
)
FetchContent_MakeAvailable(opencl_icd_loader)

file(COPY ${opencl_headers_SOURCE_DIR}/CL DESTINATION ${opencl_icd_loader_SOURCE_DIR}/inc)
