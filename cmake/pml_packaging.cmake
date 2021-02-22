

# Set general Cpack Variables for package description
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_NAME "PlaidML")
set(CPACK_PACKAGE_VENDOR "Intel Corp")
set(CPACK_PACKAGE_CONTACT "Intel")
set(CPACK_PACKAGE_VERSION ${PLAIDML_VERSION})

# Installs a list of targets, recursively installing anything listed in INTERFACE_LINK_LIBRARIES
function(recursive_lib_install INSTALL_LIST)
  foreach(TP ${INSTALL_LIST})
    if(TARGET ${TP})
      # Switch to true target if necessary
      get_target_property(_ALIASED_TARGET ${TP} ALIASED_TARGET)
      if(_ALIASED_TARGET)
        set(TP ${_ALIASED_TARGET})
      endif()

      get_property(INSTALLED_TARGETS GLOBAL PROPERTY installed_targets_property)

      if(NOT ${TP} MATCHES ".*::.*" AND NOT ${TP} IN_LIST INSTALLED_TARGETS)

        # Add install_interface include directory property to target
        get_target_property(TP_INC_DIRS ${TP} INTERFACE_INCLUDE_DIRECTORIES)
        if(TP_INC_DIRS AND NOT "${TP_INC_DIRS}" MATCHES ".*INSTALL_INTERFACE.*")
          set_target_properties(${TP} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${TP_INC_DIRS}>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
        endif()

        # Install target
        install(TARGETS ${TP}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
        list(APPEND INSTALLED_TARGETS "${TP}")
        set_property(GLOBAL PROPERTY installed_targets_property "${INSTALLED_TARGETS}")

        # Recurse through dependencies
        get_target_property(NEXT_DEPS ${TP} INTERFACE_LINK_LIBRARIES)
        if(NEXT_DEPS)
          recursive_lib_install("${NEXT_DEPS}")
        endif()
      endif()
    endif()
  endforeach()
endfunction()

get_property(INSTALL_TARGETS GLOBAL PROPERTY install_targets_property)
recursive_lib_install("${INSTALL_TARGETS}")


# Install the 3rd party header files such that a copy of edsl_tests is able to run
# It *might* be possible to automatically detect and install all necessary headers in recursive_lib_install
install(DIRECTORY "${PROJECT_BINARY_DIR}/_deps/llvm-project-src/llvm/utils/unittest/googlemock/include/gmock"
        DESTINATION include)
install(DIRECTORY "${PROJECT_BINARY_DIR}/_deps/llvm-project-src/llvm/utils/unittest/googletest/include/gtest"
        DESTINATION include)
install(DIRECTORY "${PROJECT_BINARY_DIR}/_deps/half-src/include"
        DESTINATION .)
install(DIRECTORY "${PROJECT_BINARY_DIR}/_deps/llvm-project-src/llvm/include/llvm"
        DESTINATION include)
install(DIRECTORY "${PROJECT_BINARY_DIR}/_deps/llvm-project-src/llvm/include/llvm-c"
        DESTINATION include)
install(DIRECTORY "${PROJECT_BINARY_DIR}/_deps/llvm-project-build/include/llvm"
        DESTINATION include)
install(DIRECTORY "${PROJECT_BINARY_DIR}/_deps/openvino-build/inference-engine/samples/thirdparty/gflags/include/gflags"
        DESTINATION include)


include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${CPACK_PACKAGE_VERSION}
                                 COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/pml_config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

# Install cmake for Boost (Cannot figure out how to install Boost target directly)
install(FILES "${PROJECT_SOURCE_DIR}/cmake/third_party/boost.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/third_party/CPM.cmake"
        DESTINATION cmake/third_party)

# TODO: Install any desired standalone executables or source code for reference

include(CPack)
