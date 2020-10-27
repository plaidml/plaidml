add_subdirectory(ir)
#add_subdirectory(tests)
add_subdirectory(transforms)

pml_cc_library(
  NAME
    comp
  DEPS
    ::ir
    ::transforms
)