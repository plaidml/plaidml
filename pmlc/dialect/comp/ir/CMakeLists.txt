pml_add_mlir(
  NAME
    comp
  DEPS
    util
)

pmlc_add_mlir_dialect(comp)
add_mlir_doc(ops -gen-op-doc comp_ops comp/)
pmlc_add_mlir_interfaces(comp)

add_mlir_dialect_library(pmlc_comp_ir_lib
        ops.cc
        types.cc
        interfaces.cc

        ADDITIONAL_HEADER_DIRS

        DEPENDS
                MLIROpAsmInterfaceIncGen
                pmlc_util_enums_gen
        LINK_LIBS 
                PUBLIC 
                MLIRIR
                MLIRSideEffectInterfaces
 )
target_compile_options(pmlc_pxa_ir_lib PUBLIC ${PMLC_DEFAULT_COPTS})

