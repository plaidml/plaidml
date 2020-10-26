add_subdirectory(ir)
#add_subdirectory(tests)

pml_cc_library(
  NAME
    tile
  DEPS
    pmlc_dialect_ir_tile
)