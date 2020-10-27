pmlc_add_mlir_dialect(tile)
pmlc_add_mlir_interfaces(tile)
add_mlir_doc(interfaces -gen-dialect-doc tile_interfaces tile/)
add_mlir_doc(ops -gen-op-doc tile_ops tile/)

add_mlir_dialect_library(pmlc_dialect_ir_tile
   dialect.cc
   eltwise.cc
   ops.cc
   util.cc

   ADDITIONAL_HEADER_DIRS

   DEPENDS
     MLIROpAsmInterfaceIncGen
     pmlc_tile_dialect_gen
     pmlc_tile_interfaces_gen
     pmlc_util_enums_gen

   LINK_LIBS 
     PUBLIC 
     MLIRIR
     MLIRSideEffectInterfaces
 )
target_compile_options(pmlc_dialect_ir_tile PUBLIC ${PMLC_DEFAULT_COPTS})

