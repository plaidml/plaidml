include(CMakeParseArguments)


# pml_mlir_ir()
#
# Creates a <PACKAGE>_mlir_ir target
#
# Parameters:
# DEPS: List of other libraries to be linked in to the binary targets
function(pml_mlir_ir)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME" # have to have this here in order to skim dialect name - could be renamed
    "SRCS;DEPS"
    ${ARGN}
  )
  
  # Replace dependencies passed by ::name with ::pml::package::name
  # Prefix the library with the package name, so we get: pml_package_name.
  pml_package_ns(_PACKAGE_NS)

  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  pml_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}")

  set(LLVM_TARGET_DEFINITIONS ops.td)
  mlir_tablegen(ops.h.inc -gen-op-decls)
  mlir_tablegen(ops.cc.inc -gen-op-defs)
  mlir_tablegen(dialect.h.inc -gen-dialect-decls -dialect=${_RULE_NAME})
  add_mlir_doc(ops -gen-op-doc ${_RULE_NAME}_ops ${_RULE_NAME}/)
  add_public_tablegen_target(${_NAME}_gen)
  add_dependencies(mlir-headers ${_NAME}_gen)
  list(APPEND _GEN_DEPS ${_NAME}_gen)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/interfaces.td")
    set(LLVM_TARGET_DEFINITIONS interfaces.td)
    mlir_tablegen(interfaces.h.inc -gen-op-interface-decls)
    mlir_tablegen(interfaces.cc.inc -gen-op-interface-defs)
    add_mlir_doc(ops -gen-interfaces-doc "${_PACKAGE_NAME}_interfaces" /=${RULE_NAME}/)
    add_public_tablegen_target(${_NAME}_interfaces_gen)
    add_dependencies(mlir-headers ${_NAME}_interfaces_gen)
    list(APPEND _GEN_DEPS ${_NAME}_interfaces_gen)
  endif()

  add_mlir_dialect_library(${_NAME}
      ${_RULE_SRCS}
    DEPENDS
      ${_GEN_DEPS}
      ${_RULE_DEPS}
      MLIROpAsmInterfaceIncGen
    LINK_LIBS PUBLIC
      MLIRIR
      MLIRSideEffectInterfaces
  )

  add_library(${_PACKAGE_NS} ALIAS ${_NAME})
  target_compile_options(${_NAME} PUBLIC ${PMLC_DEFAULT_COPTS})
  pml_package_dir(_PACKAGE_DIR)
endfunction()


# pml_mlir_transforms()
#
# Creates a <PACKAGE>_mlir_transforms target
#
# Parameters:
# DEPS: List of other libraries to be linked in to the binary targets
function(pml_mlir_transforms)
  cmake_parse_arguments(
    _RULE
    ""
    ""
    "SRCS;DEPS"
    ${ARGN}
  )

  pml_package_ns(_PACKAGE_NS)

  # Replace dependencies passed by ::name with ::pml::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  # Prefix the library with the package name, so we get: pml_package_name.
  pml_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}")
 
  set(LLVM_TARGET_DEFINITIONS passes.td)
  mlir_tablegen(passes.h.inc -gen-pass-decls)
  add_mlir_doc(ops -gen-passes-doc "${_PACKAGE_NAME}_passes" stdx/)
  add_public_tablegen_target(${_NAME}_gen)
  add_dependencies(mlir-headers ${_NAME}_gen)
  add_mlir_dialect_library(${_NAME}
      ${_RULE_SRCS}
    DEPENDS
      ${_NAME}_gen
      ${_RULE_DEPS}
      MLIRLLVMIR
    LINK_LIBS PUBLIC
      MLIRIR
      MLIROpenMP
      MLIRSideEffectInterfaces
  )

  # Alias the pml_package_name library to pml::package::name.
  add_library(${_PACKAGE_NS} ALIAS ${_NAME})
  pml_package_dir(_PACKAGE_DIR)
endfunction()
