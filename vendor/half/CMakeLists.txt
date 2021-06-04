add_library(half INTERFACE)

target_include_directories(half INTERFACE ".")

install(
  FILES half.hpp
  DESTINATION "include/"
  COMPONENT devkit
)
