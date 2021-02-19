set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_NAME "PlaidML")
set(CPACK_PACKAGE_VENDOR "Intel Corp")
set(CPACK_PACKAGE_CONTACT "Intel")
set(CPACK_PACKAGE_VERSION ${PLAIDML_VERSION})

get_property(gen_dep_list GLOBAL PROPERTY gen_list_property)
get_property(exp_list GLOBAL PROPERTY exp_list_property)
get_property(tp_list GLOBAL PROPERTY tp_list_property)

foreach(GEN_DEP ${gen_dep_list})
  if(NOT ${GEN_DEP} IN_LIST exp_list)
    install(TARGETS ${GEN_DEP}
      EXPORT ${PROJECT_NAME}_Targets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endforeach()

function(tp_install INSTALL_LIST)
  foreach(TP ${INSTALL_LIST})
    get_property(exp_list GLOBAL PROPERTY exp_list_property)
    if(NOT ${TP} MATCHES ".*::.*" AND TARGET ${TP} AND NOT ${TP} IN_LIST exp_list)
      get_target_property(TP_INC_DIRS ${TP} INTERFACE_INCLUDE_DIRECTORIES)
      set(DONT_FIX "FALSE")
      foreach(INCDIR ${TP_INC_DIRS})
        if(${INCDIR} MATCHES ".*INSTALL_INTERFACE.*" OR ${INCDIR} MATCHES ".*NOTFOUND.*")
          set(DONT_FIX "TRUE")
        endif()
      endforeach()
      if(${DONT_FIX} MATCHES "FALSE")
        set_target_properties(${TP} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${TP_INC_DIRS}>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
      endif()
      install(TARGETS ${TP}
      EXPORT ${PROJECT_NAME}_Targets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
      list(APPEND exp_list "${TP}")
      set_property(GLOBAL PROPERTY exp_list_property "${exp_list}")
      get_target_property(NEXT_DEPS ${TP} INTERFACE_LINK_LIBRARIES)
      if(NOT "${NEXT_DEPS}" MATCHES ".*NOTFOUND.*")
        tp_install("${NEXT_DEPS}")
      endif()
    endif()
  endforeach()
endfunction()

tp_install("${tp_list}")

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

install(FILES "${PROJECT_SOURCE_DIR}/cmake/third_party/boost.cmake"
"${PROJECT_SOURCE_DIR}/cmake/third_party/CPM.cmake"
DESTINATION cmake/third_party)

# include(GetPrerequisites)

# # Install binary target
# set(BIN plaidml_edsl_tests_cc_test)
# install(
#     TARGETS ${BIN}
#     DESTINATION bin
#     COMPONENT DEVKIT
# )

# set(BINARY_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/bin/${BIN}")
# get_prerequisites(${BINARY_LOCATION} DEPENDENCIES 0 0 "" "")
# set(SYSDEPS "")
# foreach(DEPENDENCY_FILE ${DEPENDENCIES})
#   # Install non-system dependencies
#   if(${DEPENDENCY_FILE} MATCHES "${CMAKE_CURRENT_SOURCE_DIR}/*")
#     install(
#       PROGRAMS ${DEPENDENCY_FILE}
#       DESTINATION lib
#       COMPONENT DEVKIT
#     )
#   # Add system dependencies to dependency list
#   else()
#     execute_process(COMMAND dpkg -S ${DEPENDENCY_FILE}  
#                     OUTPUT_VARIABLE DEP_PROVIDER)
#     string(REGEX REPLACE ":.*" "" DEPENDENCY ${DEP_PROVIDER})
#     if (NOT ${DEPENDENCY} IN_LIST SYSDEPS)
#       string(LENGTH "${SYSDEPS}" DEP_STR_LEN)
#       if(NOT "${DEP_STR_LEN}" EQUAL "0")
#         set(SYSDEPS "${SYSDEPS}, ")
#       endif()
#       set(SYSDEPS "${SYSDEPS}${DEPENDENCY}")
#     endif()
#   endif()
# endforeach()

# set(CPACK_DEBIAN_PACKAGE_DEPENDS ${SYSDEPS})

# # Add package RPath to binary target
# set_target_properties(${BIN} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:\$LD_LIBRARY_PATH")

include(CPack)
