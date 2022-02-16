message("Fetching mlperf")
FetchContent_Declare(
  mlperf
  URL      https://github.com/mlcommons/inference/archive/d29092298f5075b234eee21352a85c094a636e71.zip
  URL_HASH SHA256=55bc744a5b2725ed1ba4bb9634b8d2078cad70eca0e05cd7cce731750f50aa55
)

FetchContent_GetProperties(mlperf)
if(NOT mlperf_POPULATED)
  FetchContent_Populate(mlperf)
endif()

execute_process(
  COMMAND ${PYTHON_EXECUTABLE}
    ${mlperf_SOURCE_DIR}/loadgen/version_generator.py
    ${mlperf_SOURCE_DIR}/loadgen/version_generated.cc
    ${mlperf_SOURCE_DIR}/loadgen
)
