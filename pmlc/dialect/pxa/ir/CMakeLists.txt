pmlc_add_mlir_dialect(PXA)
add_mlir_doc(ops -gen-op-doc PXAOps PXA/)

add_mlir_dialect_library(PMLCPXA
        ops.cc

        ADDITIONAL_HEADER_DIRS

        DEPENDS
                MLIROpAsmInterfaceIncGen
                PMLCPXADialectGen
                PMLCUtilEnumsGen
                PMLCUtilInterfacesGen
        LINK_LIBS 
                PUBLIC 
                MLIRIR
                MLIRSideEffectInterfaces
                MLIRStandardOps
 )
target_compile_options(PMLCPXA PUBLIC ${PMLC_DEFAULT_COPTS})

