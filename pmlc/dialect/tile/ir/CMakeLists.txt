# In order to include source and headers we have to do it this way.
pmlc_add_mlir_dialect(ops eltwise)
pmlc_add_mlir_dialect(interfaces eltwise)
add_mlir_doc(interfaces -gen-dialect-doc EltwiseInterfaces Eltwise/)
add_mlir_doc(ops -gen-op-doc EltwiseOps Eltwise/)

add_mlir_dialect_library(Eltwise
        dialect.cc
        ops.cc

        ADDITIONAL_HEADER_DIRS

        DEPENDS
          MLIROpAsmInterfaceIncGen
          PMLC-eltwise-opsIncGen
          PMLC-eltwise-interfacesIncGen
          PMLC-util-interfacesIncGen
        LINK_LIBS 
          PUBLIC 
          MLIRIR
          MLIRSideEffectInterfaces
          MLIRStandardOps
          PMLCUtil
  )
