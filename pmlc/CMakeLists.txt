# Declare a dialect in the include directory
function(pmlc_add_mlir_dialect namespace)
  set(LLVM_TARGET_DEFINITIONS ops.td)
  mlir_tablegen(ops.h.inc -gen-op-decls)
  mlir_tablegen(ops.cc.inc -gen-op-defs)
  mlir_tablegen(dialect.h.inc -gen-dialect-decls -dialect=${namespace})
  add_public_tablegen_target(pmlc_${namespace}_dialect_gen)
  add_dependencies(mlir-headers pmlc_${namespace}_dialect_gen)
endfunction()

function(pmlc_add_mlir_interfaces namespace)
  set(LLVM_TARGET_DEFINITIONS interfaces.td)
  mlir_tablegen(interfaces.h.inc -gen-op-interface-decls)
  mlir_tablegen(interfaces.cc.inc -gen-op-interface-defs)
  add_public_tablegen_target(pmlc_${namespace}_interfaces_gen)
  add_dependencies(mlir-headers pmlc_${namespace}_interfaces_gen)
endfunction()

function(pmlc_add_mlir_passes namespace)
endfunction()

add_subdirectory(util)
add_subdirectory(dialect)
add_subdirectory(tools)

pml_cc_library(
  NAME
    all_passes_and_dialects
  DEPS
    pmlc_dialect_tile
    
    MLIRAffinePassIncGen
    MLIRConversionPassIncGen
    MLIRGPUPassIncGen
    MLIRLinalgPassIncGen
    MLIRLLVMPassIncGen
    MLIRTransformsPassIncGen
    MLIRStandardTransformsIncGen
)