configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(PML_TEST_DEPENDS
  FileCheck count not
  pmlc-opt
)

# add_lit_testsuite(check-pml "Running the PML regression tests"
#   ${CMAKE_CURRENT_BINARY_DIR}
#   DEPENDS ${PML_TEST_DEPENDS}
# )

# add_lit_testsuites(
#   STANDALONE ${CMAKE_CURRENT_SOURCE_DIR}
#   DEPENDS ${PML_TEST_DEPENDS}
# )

add_custom_target(check-pml
  COMMAND ${llvm-project_BINARY_DIR}/bin/llvm-lit pmlc -v
  COMMENT "Running the PML regression tests"
  USES_TERMINAL
)
add_dependencies(check-pml ${PML_TEST_DEPENDS})
set_target_properties(check-pml PROPERTIES FOLDER "Tests")
