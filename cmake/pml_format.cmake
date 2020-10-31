find_program(CLANG_FORMAT_PROGRAM clang-format)
find_program(GIT_PROGRAM git)
find_package(Python)

if(CLANG_FORMAT_PROGRAM AND Python_FOUND)
  set(_CLANG_FORMAT_COMMAND
    ${Python_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/cmake/git-clang-format.py
    --binary=${CLANG_FORMAT_PROGRAM}
  )

  add_custom_target(
    clang-format
    COMMAND ${_CLANG_FORMAT_COMMAND} --diff ${MAIN_BRANCH}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT clang-format
  )

  add_custom_target(
    check-clang-format
    COMMAND ${_CLANG_FORMAT_COMMAND} --ci ${MAIN_BRANCH}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT check-clang-format
  )

  add_custom_target(
    fix-clang-format
    COMMAND ${_CLANG_FORMAT_COMMAND} ${MAIN_BRANCH} -f
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT fix-clang-format
  )
else()
  message(STATUS "clang-format and/or python not found, adding dummy targets")

  set(_CLANG_FORMAT_COMMAND
    # show error message
    COMMAND ${CMAKE_COMMAND} -E echo
    "cannot run because clang-format and/or python not found"
    # fail build
    COMMAND ${CMAKE_COMMAND} -E false
  )

  add_custom_target(clang-format ${_CLANG_FORMAT_COMMAND})
  add_custom_target(check-clang-format ${_CLANG_FORMAT_COMMAND})
  add_custom_target(fix-clang-format ${_CLANG_FORMAT_COMMAND})
endif()
