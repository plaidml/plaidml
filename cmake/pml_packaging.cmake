

# Set general Cpack Variables for package description
set(CPACK_GENERATOR "TGZ")
set(CPACK_ARCHIVE_COMPONENT_INSTALL 1)
set(CPACK_PACKAGE_NAME "PlaidML")
set(CPACK_PACKAGE_VENDOR "Intel Corp")
set(CPACK_PACKAGE_CONTACT "Intel")
set(CPACK_PACKAGE_VERSION ${PLAIDML_VERSION})
set(CPACK_COMPONENTS_IGNORE_GROUPS 1)
set(CPACK_COMPONENTS_ALL devkit)

set_target_properties(mlir_runner_utils PROPERTIES INTERFACE_LINK_LIBRARIES "")

install(TARGETS plaidml_plaidml
                omp
                mlir_runner_utils
        EXPORT ${PROJECT_NAME}_Targets
        DESTINATION lib
        COMPONENT devkit)

get_target_property(OMP_INC_DIRS omp INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(omp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${OMP_INC_DIRS}>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

include(CMakePackageConfigHelpers) 
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${CPACK_PACKAGE_VERSION}
                                 COMPATIBILITY SameMajorVersion)

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake
        COMPONENT devkit)

configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/packaging/pml_config.cmake.in"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake)

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake
        COMPONENT devkit)

install(FILES "${PROJECT_SOURCE_DIR}/cmake/third_party/CPM.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/third_party/easylogging.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/third_party/gflags.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/third_party/googletest.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/third_party/half.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/third_party/llvm-project.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/third_party
        COMPONENT devkit)

install(FILES "${PROJECT_SOURCE_DIR}/cmake/packaging/CMakeLists.txt"
              "${PROJECT_SOURCE_DIR}/plaidml/edsl/tests/edsl_test.cc"
              "${PROJECT_SOURCE_DIR}/pmlc/testing/gtest_main.cc"
        DESTINATION example_project
        COMPONENT devkit)

# TODO: Install any other source code, README, etc

include(CPack)