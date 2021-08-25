message("Fetching pybind11")
FetchContent_Declare(
  pybind11
  URL      https://github.com/pybind/pybind11/archive/refs/tags/v2.7.1.tar.gz
  URL_HASH SHA256=616d1c42e4cf14fa27b2a4ff759d7d7b33006fdc5ad8fd603bb2c22622f27020
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
