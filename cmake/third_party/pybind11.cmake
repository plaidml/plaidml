message("Fetching pybind11")
FetchContent_Declare(
  pybind11
  URL      https://github.com/pybind/pybind11/archive/refs/tags/v2.10.1.tar.gz
  URL_HASH SHA256=111014b516b625083bef701df7880f78c2243835abdb263065b6b59b960b6bad
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
