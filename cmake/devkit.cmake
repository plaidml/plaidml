include(CMakePackageConfigHelpers)

# Set general cpack variables for package description
set(CPACK_GENERATOR "TGZ")
set(CPACK_ARCHIVE_COMPONENT_INSTALL 1)
set(CPACK_PACKAGE_NAME "PlaidML")
set(CPACK_PACKAGE_VENDOR "Intel Corp")
set(CPACK_PACKAGE_CONTACT "Intel")
set(CPACK_PACKAGE_VERSION ${PLAIDML_VERSION})
set(CPACK_COMPONENTS_IGNORE_GROUPS 1)
set(CPACK_COMPONENTS_ALL devkit)
include(CPack)

set_target_properties(plaidml_plaidml
  PROPERTIES
    EXPORT_NAME PlaidML
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS
    plaidml_plaidml
  EXPORT ${PROJECT_NAME}_Targets
  DESTINATION "lib"
  COMPONENT devkit
)

install(TARGETS
    omp
    mlir_runner_utils
  DESTINATION "lib"
  COMPONENT devkit
)

write_basic_package_version_file(
  "${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${CPACK_PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION "cmake"
  COMPONENT devkit
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "cmake"
)

install(
  FILES
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/third_party/gflags.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/third_party/googletest.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/third_party/half.cmake"
  DESTINATION "cmake"
  COMPONENT devkit
)

install(
  FILES
    "${PROJECT_SOURCE_DIR}/devkit/CMakeLists.txt"
  DESTINATION "."
  COMPONENT devkit
)

install(
  FILES
    "${PROJECT_SOURCE_DIR}/plaidml/edsl/tests/edsl_test.cc"
    "${PROJECT_SOURCE_DIR}/plaidml/testenv.cc"
  DESTINATION "src"
  COMPONENT devkit
)

# TODO: Install any other source code, README, etc
