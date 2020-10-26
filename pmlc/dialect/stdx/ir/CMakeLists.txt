# In order to include source and headers we have to do it this way.
pmlc_add_mlir_dialect(StdX)
add_mlir_doc(ops -gen-op-doc StdXOps StdX/)

add_mlir_dialect_library(PMLCStdX
        ops.cc

        ADDITIONAL_HEADER_DIRS

        DEPENDS
                MLIROpAsmInterfaceIncGen
                PMLCStdXDialectGen
        LINK_LIBS 
                PUBLIC 
                MLIRIR
                MLIRSideEffectInterfaces
 )
target_compile_options(PMLCStdX PUBLIC ${PMLC_DEFAULT_COPTS})

