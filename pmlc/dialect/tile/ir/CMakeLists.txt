pmlc_add_mlir_dialect(Tile)
pmlc_add_mlir_interfaces(Tile)
add_mlir_doc(interfaces -gen-dialect-doc TileInterfaces Tile/)
add_mlir_doc(ops -gen-op-doc TileOps Tile/)

add_mlir_dialect_library(pmlc_dialect_ir_tile
        dialect.cc
        eltwise.cc
        ops.cc
        util.cc

        ADDITIONAL_HEADER_DIRS

        DEPENDS
                MLIROpAsmInterfaceIncGen
                PMLCTileDialectGen
                PMLCTileInterfacesGen
                PMLCUtilEnumsGen
        LINK_LIBS 
                PUBLIC 
                MLIRIR
                MLIRSideEffectInterfaces
 )
target_compile_options(pmlc_dialect_ir_tile PUBLIC ${PMLC_DEFAULT_COPTS})

