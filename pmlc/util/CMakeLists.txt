pmlc_add_mlir_dialect(interfaces util)
add_mlir_dialect_library(PMLCUtil
        interfaces.cc

        ADDITIONAL_HEADER_DIRS

        DEPENDS
          MLIROpAsmInterfaceIncGen
        LINK_LIBS 
          PUBLIC 
          MLIRIR
          MLIRSideEffectInterfaces
          MLIRStandardOps
  )