pmlc_add_mlir_dialect(Eltwise)
pmlc_add_mlir_interfaces(Eltwise)
add_mlir_doc(interfaces -gen-dialect-doc EltwiseInterfaces Eltwise/)
add_mlir_doc(ops -gen-op-doc EltwiseOps Eltwise/)

add_mlir_dialect_library(PMLCEltwise
        dialect.cc
        ops.cc

        ADDITIONAL_HEADER_DIRS

        DEPENDS
                MLIROpAsmInterfaceIncGen
                PMLCEltwiseDialectGen
                PMLCEltwiseInterfacesGen
                PMLCUtilInterfacesGen
        LINK_LIBS 
                PUBLIC 
                MLIRIR
                MLIRSideEffectInterfaces
                MLIRStandardOps
 )
target_compile_options(PMLCEltwise PUBLIC ${PMLC_DEFAULT_COPTS})

