add_subdirectory(ir)
#add_subdirectory(tests)
add_subdirectory(transforms)

pml_cc_library(
  NAME
    tile
  DEPS
    ::ir
    ::transforms
)