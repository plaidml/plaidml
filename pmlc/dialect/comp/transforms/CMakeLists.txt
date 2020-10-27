add_mlir_doc(ops -gen-transforms-doc comp_transforms comp/)
pmlc_add_mlir_passes(comp)

add_mlir_dialect_library(pmlc_comp_transforms_lib
  execenv_coalescing.cc
  minimize_allocations.cc

  ADDITIONAL_HEADER_DIRS

  DEPENDS
    MLIROpAsmInterfaceIncGen
    pmlc_comp_passes_gen
    pmlc_comp_dialect_gen
  LINK_LIBS 
    PUBLIC 
    MLIRIR
    MLIRSideEffectInterfaces
 )
target_compile_options(pmlc_comp_transforms_lib PUBLIC ${PMLC_DEFAULT_COPTS})

